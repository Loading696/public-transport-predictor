"""HTTP client for the ML service, and the only inference path the backend has.

The backend used to load CatBoost in-process.  It no longer does: every
prediction now crosses a network boundary to the ``predictor-ml`` service.  This
module owns that boundary and, just as importantly, owns what happens when the
other side is not there.

Failure handling, by requirement rather than by accident
--------------------------------------------------------
The ML service is a separate container: it can be restarting, still loading its
model, briefly unreachable, or answering with something unexpected.  None of that
may take the backend down with it, because the backend also serves the NDTP
receiver, the replay stream and the cascade.  So:

* **timeout** -- connect and read timeouts are set separately, so a hung service
  is abandoned in seconds rather than pinning a request thread;
* **retry** -- one retry with a short backoff, because a container restart looks
  exactly like a transient connection refusal;
* **circuit breaker** -- after ``failure_threshold`` consecutive failures the
  client stops dialling for ``cooldown_s`` and fails fast from a local verdict.
  Without it, a dead ML service turns every request into a full connect timeout;
* **bad response** -- anything that is not the documented shape, or a non-finite
  prediction, is an error rather than a silently wrong number.

Every failure surfaces as :class:`MlServiceError`.  Callers decide the policy:
``/predict`` answers 503, while the read-only views keep serving last known
state with an explicit ``ml_status`` marker.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx

DEFAULT_BASE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8001")
DEFAULT_TIMEOUT_S = float(os.getenv("ML_TIMEOUT_S", "10"))
DEFAULT_CONNECT_TIMEOUT_S = float(os.getenv("ML_CONNECT_TIMEOUT_S", "2"))
FAILURE_THRESHOLD = int(os.getenv("ML_FAILURE_THRESHOLD", "3"))
COOLDOWN_S = float(os.getenv("ML_COOLDOWN_S", "10"))
RETRY_BACKOFF_S = float(os.getenv("ML_RETRY_BACKOFF_S", "0.25"))


class MlServiceError(RuntimeError):
    """Any failure to obtain a usable answer from the ML service.

    ``kind`` distinguishes the cases a caller may want to treat differently:
    ``unavailable`` (transport, breaker open), ``timeout``, ``status`` (non-2xx),
    ``malformed`` (unparseable or wrong-shaped body), ``loading`` (503 from a
    service whose model is not ready yet).
    """

    def __init__(self, message: str, *, kind: str = "unavailable", status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


class _Breaker:
    """Consecutive-failure gate.

    Closed -> calls pass through.  ``failure_threshold`` consecutive failures
    open it for ``cooldown_s``; the first success after that closes it again.
    Time-based, not attempt-based, so a service that stays down is not dialled
    once per request.
    """

    def __init__(self, threshold: int, cooldown_s: float) -> None:
        self.threshold = max(1, int(threshold))
        self.cooldown_s = float(cooldown_s)
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if (time.monotonic() - self.opened_at) >= self.cooldown_s:
            # Cooldown elapsed: let one call through to probe.
            self.opened_at = None
            self.failures = 0
            return False
        return True

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def state(self) -> dict[str, Any]:
        return {
            "open": self.is_open,
            "consecutive_failures": self.failures,
            "threshold": self.threshold,
            "cooldown_s": self.cooldown_s,
        }


class MlClient:
    """Async client for the ML service.

    One instance per backend process, reused across requests: the underlying
    ``httpx.AsyncClient`` keeps the connection pool warm, which is most of the
    per-call latency at these sizes.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_s: float | None = None,
        connect_timeout_s: float | None = None,
        failure_threshold: int | None = None,
        cooldown_s: float | None = None,
    ) -> None:
        self._url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._timeout = float(timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S)
        self._connect_timeout = float(
            connect_timeout_s if connect_timeout_s is not None else DEFAULT_CONNECT_TIMEOUT_S
        )
        self._client = httpx.AsyncClient(
            base_url=self._url,
            timeout=httpx.Timeout(
                self._timeout, connect=self._connect_timeout, read=self._timeout
            ),
        )
        self._breaker = _Breaker(
            failure_threshold if failure_threshold is not None else FAILURE_THRESHOLD,
            cooldown_s if cooldown_s is not None else COOLDOWN_S,
        )
        self._lock = asyncio.Lock()
        self.calls = 0
        self.failures = 0
        self.last_error: str | None = None
        self.last_error_kind: str | None = None
        self.last_ok_monotonic: float | None = None

    @property
    def base_url(self) -> str:
        return self._url

    @base_url.setter
    def base_url(self, value: str) -> None:
        """Rebind the client to another ML service.

        Setting only the attribute would be a trap: ``httpx`` resolves relative
        URLs against the ``base_url`` captured at construction, so a silently
        ineffective assignment would send traffic to the old address while every
        log line claimed the new one.  Pushing it through keeps the two in step
        and makes failover to a replica possible at runtime.
        """
        self._url = str(value).rstrip("/")
        self._client.base_url = self._url

    # -- lifecycle -------------------------------------------------------- #

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- introspection ---------------------------------------------------- #

    def status(self) -> dict[str, Any]:
        """Local view of the boundary, for /metrics and the status endpoints."""
        breaker = self._breaker.state()
        healthy = breaker["open"] is False and self.last_error is None
        return {
            "url": self.base_url,
            "healthy": healthy,
            "last_error": self.last_error,
            "last_error_kind": self.last_error_kind,
            "calls": self.calls,
            "failures": self.failures,
            "breaker": breaker,
            "timeout_s": self._timeout,
            "connect_timeout_s": self._connect_timeout,
        }

    # -- transport -------------------------------------------------------- #

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if self._breaker.is_open:
            raise MlServiceError(
                f"ML service circuit open after {self._breaker.failures} consecutive failures",
                kind="unavailable",
            )
        self.calls += 1
        last: MlServiceError | None = None
        for attempt in (0, 1):
            try:
                response = await self._client.request(method, path, **kwargs)
            except httpx.TimeoutException as exc:
                last = MlServiceError(
                    f"ML service timeout after {self._timeout:g}s: {exc!s}", kind="timeout"
                )
            except httpx.HTTPError as exc:
                last = MlServiceError(
                    f"ML service unreachable at {self.base_url}: {exc!s}", kind="unavailable"
                )
            else:
                if response.status_code >= 400:
                    kind = "loading" if response.status_code == 503 else "status"
                    detail = _safe_detail(response)
                    last = MlServiceError(
                        f"ML service returned {response.status_code}: {detail}",
                        kind=kind,
                        status=response.status_code,
                    )
                else:
                    self._breaker.record_success()
                    self.last_error = None
                    self.last_error_kind = None
                    self.last_ok_monotonic = time.monotonic()
                    return _decode(response)
            if attempt == 0:
                await asyncio.sleep(RETRY_BACKOFF_S)
        assert last is not None
        self._breaker.record_failure()
        self.failures += 1
        self.last_error = str(last)
        self.last_error_kind = last.kind
        raise last

    async def health(self) -> dict[str, Any]:
        """GET /health on the ML service (proxied by the backend's /health)."""
        return await self._request("GET", "/health")

    async def models(self) -> dict[str, Any]:
        """GET /models -- feature names and readiness."""
        return await self._request("GET", "/models")

    async def predict_batch(
        self,
        points: list[dict[str, Any]],
        telemetry: Any = None,
    ) -> list[dict[str, Any]]:
        """POST /predict -- one batched inference call.

        ``telemetry`` may be a list of mappings or a DataFrame; a DataFrame is
        sent as records.  Raises :class:`MlServiceError` on every failure mode.
        """
        if not points:
            return []
        payload: dict[str, Any] = {"points": [_as_point(p) for p in points]}
        rows = _telemetry_rows(telemetry)
        if rows:
            payload["telemetry"] = rows
        body = await self._request("POST", "/predict", json=payload)
        if not isinstance(body, dict):
            raise MlServiceError("malformed ML response: not an object", kind="malformed")
        records = body.get("predictions")
        if not isinstance(records, list) or len(records) != len(points):
            raise MlServiceError(
                f"malformed ML response: {len(records) if isinstance(records, list) else '?'} "
                f"records for {len(points)} points",
                kind="malformed",
            )
        for record in records:
            if not isinstance(record, dict):
                raise MlServiceError("malformed ML response: record is not an object", kind="malformed")
            value = record.get("prediction")
            if not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
                raise MlServiceError(
                    f"malformed ML response: bad prediction {value!r}", kind="malformed"
                )
        return records

    # -- offline convenience ---------------------------------------------- #

    def predict_batch_sync(
        self, points: list[dict[str, Any]], telemetry: Any = None
    ) -> list[dict[str, Any]]:
        """Blocking wrapper for CLI scripts and tests.

        Refuses to run inside a live event loop -- an offline script has no loop,
        and quietly spawning one here would hide a design mistake.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "predict_batch_sync() called from a running event loop; await predict_batch()"
            )
        return asyncio.run(self._predict_and_close(points, telemetry))

    async def _predict_and_close(self, points: list[dict[str, Any]], telemetry: Any) -> list[dict[str, Any]]:
        try:
            return await self.predict_batch(points, telemetry)
        finally:
            await self.aclose()


# --------------------------------------------------------------------------- #
# Payload helpers
# --------------------------------------------------------------------------- #


def _as_point(point: dict[str, Any]) -> dict[str, Any]:
    """Reduce a point to the fields the ML service needs, JSON-safe."""
    out = {
        "tr_id": int(point["tr_id"]),
        "T": _iso(point["T"]),
        "target_stop_id": None if point.get("target_stop_id") is None else int(point["target_stop_id"]),
        "target_time_begin": None if point.get("target_time_begin") is None else _iso(point["target_time_begin"]),
        "cur_dev_s": float(point["cur_dev_s"]),
    }
    sample_id = point.get("sample_id")
    if sample_id is not None and str(sample_id) != "":
        out["sample_id"] = str(sample_id)
    return out


def _telemetry_rows(telemetry: Any) -> list[dict[str, Any]]:
    """Telemetry as JSON-safe records; accepts a DataFrame or an iterable."""
    if telemetry is None:
        return []
    if hasattr(telemetry, "to_dict"):
        telemetry = telemetry.to_dict("records")
    rows: list[dict[str, Any]] = []
    for row in telemetry:
        rows.append(
            {
                "tr_id": int(row["tr_id"]),
                "event_time": _iso(row["event_time"]),
                "location_valid": bool(row.get("location_valid", False)),
                "lon": _num(row.get("lon")),
                "lat": _num(row.get("lat")),
                "alt": _num(row.get("alt")),
                "speed": _num(row.get("speed")),
                "heading": _num(row.get("heading")),
            }
        )
    return rows


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _iso(value: Any) -> str:
    """Timestamp in one canonical shape, microseconds always present.

    Uniformity is the point: a column that mixes ``...:00`` with
    ``...:00.462764`` is exactly the input that made the receiver's timestamp
    parsing ambiguous.  Fixing the format here means the service never has to
    guess, and the fix survives a change of client library.
    """
    if value is None:
        raise ValueError("timestamp is required")
    if hasattr(value, "isoformat"):
        import pandas as pd

        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f")
    text = str(value).strip()
    if not text:
        raise ValueError("timestamp is empty")
    try:
        import pandas as pd

        return pd.Timestamp(text).strftime("%Y-%m-%d %H:%M:%S.%f")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unparseable timestamp {text!r}") from exc


def _decode(response: httpx.Response) -> Any:
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise MlServiceError(
            f"malformed ML response: not JSON ({exc!s})", kind="malformed"
        ) from exc


def _safe_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - an error page may not be JSON
        return (response.text or "")[:200]
    if isinstance(body, dict):
        return str(body.get("detail", body))[:200]
    return str(body)[:200]
