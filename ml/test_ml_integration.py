"""Integration tests for the Backend -> ML service boundary.

The unit tests in ``test_batching.py`` cover the pipeline in-process.  This file
covers the thing that only exists because the system was split in two: the
network boundary, and what the backend does when it misbehaves.

Covered:

  1. discovery  -- the service is reached by URL, not embedded; health is real;
  2. batch      -- one POST /predict returns one record per point;
  3. equivalence-- batching still changes nothing over HTTP (1 row == 3 rows),
                   which is the contract the micro-batcher trades on;
  4. causality  -- future telemetry appended to the request changes nothing;
  5. unreachable-- no service at all: a clear typed error, then the breaker
                   opens and fails fast;
  6. recovery   -- the same client succeeds again once the service is back;
  7. timeout    -- a service that never answers is abandoned, not awaited;
  8. malformed  -- non-JSON, wrong shape and null predictions are all rejected
                   rather than silently accepted;
  9. 4xx/5xx    -- the error kind is preserved so the backend can map it;
 10. degradation-- the backend's InferenceService raises a typed error and the
                   stream simulator keeps running instead of dying.

A real uvicorn subprocess backs tests 1-4, so the serialisation path is actually
exercised.  5-9 use tiny stub servers, because provoking a genuine timeout or a
genuine 500 against healthy code is not possible.

Every test shares **one** event loop.  That is not incidental: an
``httpx.AsyncClient`` binds its connection pool to the loop that first uses it,
and in production there is exactly one loop per process, so the tests mirror that
rather than papering over it with a loop per call.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from ml.ml_pipeline import ROOT as PROJECT_ROOT  # noqa: E402
from src.ml_client import MlClient, MlServiceError  # noqa: E402
from src.runtime import InferenceService  # noqa: E402

STARTED_SERVICE: subprocess.Popen | None = None
EXTERNAL_URL: str | None = os.getenv("ML_TEST_SERVICE_URL")


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def wait_for(url: str, *, timeout_s: float = 120.0) -> dict[str, Any]:
    """Poll ``url`` until it answers, or fail loudly."""
    deadline = time.monotonic() + timeout_s
    last: str = "no attempt"
    async with httpx.AsyncClient(timeout=5.0) as probe:
        while time.monotonic() < deadline:
            try:
                response = await probe.get(url)
                return response.json()
            except Exception as exc:  # noqa: BLE001 - not up yet
                last = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(0.4)
    raise RuntimeError(f"service at {url} never became ready: {last}")


def start_service() -> str:
    """Start a real ML service subprocess and return its base URL."""
    global STARTED_SERVICE
    if EXTERNAL_URL:
        return EXTERNAL_URL.rstrip("/")
    port = free_port()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "ml-service")])
    STARTED_SERVICE = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "service:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(ROOT / "ml-service"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return f"http://127.0.0.1:{port}"


def stop_service() -> None:
    global STARTED_SERVICE
    if STARTED_SERVICE is not None:
        STARTED_SERVICE.terminate()
        try:
            STARTED_SERVICE.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            STARTED_SERVICE.kill()
        STARTED_SERVICE = None


def sample_request(n: int = 3) -> dict[str, Any]:
    """A realistic /predict body built from the committed validate slice."""
    import pandas as pd

    ds = PROJECT_ROOT / "dataset" / "validate"
    points = pd.read_csv(ds / "points.csv").head(n)
    traffic = pd.read_csv(ds / "traffic.csv")
    return {
        "points": [
            {
                "sample_id": str(row["sample_id"]),
                "tr_id": int(row["tr_id"]),
                "T": str(row["T"]),
                "target_stop_id": int(row["target_stop_id"]),
                "target_time_begin": str(row["target_time_begin"]),
                "cur_dev_s": float(row["cur_dev_s"]),
            }
            for _, row in points.iterrows()
        ],
        "telemetry": [
            {
                "tr_id": int(row["tr_id"]),
                "event_time": str(row["event_time"]),
                "location_valid": str(row["location_valid"]).strip().lower() in ("true", "1", "yes"),
                "lon": None if pd.isna(row["lon"]) else float(row["lon"]),
                "lat": None if pd.isna(row["lat"]) else float(row["lat"]),
                "alt": None,
                "speed": None if pd.isna(row["speed"]) else float(row["speed"]),
                "heading": None,
            }
            for _, row in traffic.iterrows()
        ],
    }


ONE_POINT = [{"tr_id": 131672, "T": "2026-01-06 03:35:00", "cur_dev_s": 0.0}]


class _StubHandler(BaseHTTPRequestHandler):
    """Answers whatever the test asked the stub to do."""

    mode = "ok"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._send(200, {"status": "ok", "service": "stub", "features": 141})

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.mode == "garbage":
            body = b"<html>not json</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.mode == "wrong_shape":
            self._send(200, {"count": 99, "predictions": []})
            return
        if self.mode == "null_prediction":
            self._send(200, {"count": 1, "predictions": [{"prediction": None}]})
            return
        if self.mode == "boom":
            self._send(500, {"detail": "kaboom"})
            return
        if self.mode == "loading":
            self._send(503, {"detail": "model is still loading"})
            return
        if self.mode == "slow":
            time.sleep(5)
            self._send(200, {"count": 0, "predictions": []})
            return
        self._send(200, {"count": 1, "predictions": [{"prediction": 1.0}]})

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:  # keep the stub quiet
        return

    def handle_one_request(self) -> None:
        # The timeout test abandons the request mid-flight and then shuts the
        # server down; the resulting broken pipe is expected, not a failure.
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True


def start_stub(mode: str) -> tuple[str, ThreadingHTTPServer]:
    handler = type("Handler", (_StubHandler,), {"mode": mode})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


# --------------------------------------------------------------------------- #
# 1-4: against a real service
# --------------------------------------------------------------------------- #


async def test_health_over_http(base: str) -> None:
    client = MlClient(base, timeout_s=30)
    try:
        health = await client.health()
    finally:
        await client.aclose()
    assert health["status"] == "ok", health
    assert health["model_loaded"] is True, health
    assert health["features"] == 141, health
    assert health["schedule_vehicles"] == 13, health
    assert health["calibrated"] is True, health
    print(f"   health:     {health['service']} {health['version']} "
          f"features={health['features']} model={Path(health['model']).name}")


async def test_batch_predict(base: str) -> None:
    body = sample_request(3)
    client = MlClient(base, timeout_s=90)
    try:
        records = await client.predict_batch(body["points"], body["telemetry"])
    finally:
        await client.aclose()
    assert len(records) == 3, records
    for record, point in zip(records, body["points"]):
        assert record["tr_id"] == point["tr_id"], record
        assert record["sample_id"] == point["sample_id"], record
        assert isinstance(record["prediction"], float), record
        assert record["risk"] in ("on-time", "at-risk", "late"), record
        assert record["target_class"] in ("early", "ontime", "late"), record
        assert 0.0 <= (record["p_late"] or 0.0) <= 1.0, record
        assert isinstance(record["pattern_events"], list), record
    print(f"   batch:      {len(records)} records, "
          f"pred={[round(r['prediction'], 2) for r in records]}")


async def test_batch_equivalence_over_http(base: str) -> None:
    """One request per point must equal one request for all of them.

    This is the property the backend's micro-batcher trades on: it merges
    concurrent callers into a single call, and that is only sound if merging
    cannot move a number.  Asserted across the network boundary, not around it.
    """
    body = sample_request(3)
    client = MlClient(base, timeout_s=90)
    try:
        batched = await client.predict_batch(body["points"], body["telemetry"])
        singles: list[dict[str, Any]] = []
        for point in body["points"]:
            singles.extend(await client.predict_batch([point], body["telemetry"]))
    finally:
        await client.aclose()
    assert len(singles) == 3, singles
    worst = max(abs(a["prediction"] - b["prediction"]) for a, b in zip(batched, singles))
    assert worst == 0.0, f"batching changed a prediction by {worst}"
    for a, b in zip(batched, singles):
        assert a["risk"] == b["risk"] and a["p_late"] == b["p_late"], (a, b)
    print(f"   equivalence: 1-by-1 == batch of 3 (max abs diff {worst:.1e}) over HTTP")


async def test_causality_over_http(base: str) -> None:
    """Future telemetry in the request must not move any prediction."""
    body = sample_request(3)
    client = MlClient(base, timeout_s=90)
    try:
        clean = await client.predict_batch(body["points"], body["telemetry"])
        future = [
            {"tr_id": body["points"][0]["tr_id"], "event_time": "2030-01-01 00:00:00",
             "location_valid": True, "lon": 0.0, "lat": 0.0,
             "alt": None, "speed": 99.0, "heading": None}
        ] * 50
        dirty = await client.predict_batch(body["points"], body["telemetry"] + future)
    finally:
        await client.aclose()
    worst = max(abs(a["prediction"] - b["prediction"]) for a, b in zip(clean, dirty))
    assert worst == 0.0, f"future telemetry leaked: {worst}"
    print(f"   causality:  50 future rows changed nothing (max abs diff {worst:.1e})")


# --------------------------------------------------------------------------- #
# 5-9: failure modes
# --------------------------------------------------------------------------- #


async def test_unavailable_and_breaker() -> None:
    """A dead address must produce a typed error and then stop costing time.

    The specific transport kind depends on the platform -- a closed local port
    is refused instantly on Linux but times out on Windows -- so the assertion is
    on the contract: a typed :class:`MlServiceError` naming a transport failure,
    and a breaker that stops re-dialling.  The ``timeout`` kind itself is pinned
    separately by :func:`test_timeout`, where the cause is unambiguous.
    """
    dead = f"http://127.0.0.1:{free_port()}"
    client = MlClient(dead, timeout_s=1, connect_timeout_s=0.5, failure_threshold=3, cooldown_s=30)
    try:
        kinds = []
        for _ in range(3):
            try:
                await client.predict_batch(ONE_POINT)
            except MlServiceError as exc:
                kinds.append(exc.kind)
        assert len(kinds) == 3, kinds
        assert set(kinds) <= {"unavailable", "timeout"}, kinds
        state = client.status()
        assert state["breaker"]["open"] is True, state
        assert state["failures"] == 3, state
        # Once open the call must fail fast from a local verdict, not by
        # re-dialling a dead address on every single request.
        started = time.perf_counter()
        try:
            await client.predict_batch(ONE_POINT)
        except MlServiceError as exc:
            fast = time.perf_counter() - started
            assert "circuit open" in str(exc), exc
        else:
            raise AssertionError("breaker should have short-circuited this call")
        assert fast < 0.5, f"breaker still dialled for {fast:.2f}s"
    finally:
        await client.aclose()
    print(f"   unavailable: 3 failures ({','.join(kinds)}) -> breaker open, "
          f"next call failed fast ({fast * 1000:.0f} ms)")


async def test_recovery(base: str) -> None:
    """A client that failed must succeed again once the service returns."""
    url = f"http://127.0.0.1:{free_port()}"
    client = MlClient(url, timeout_s=5, connect_timeout_s=1, failure_threshold=2, cooldown_s=0.1)
    try:
        try:
            await client.health()
        except MlServiceError as exc:
            # Transport failure; see test_unavailable_and_breaker for why the
            # exact kind is platform-dependent.
            assert exc.kind in ("unavailable", "timeout"), exc
        else:
            raise AssertionError("expected the port to be closed")
        # Repoint the same client at a live service and confirm it recovers.
        client.base_url = base
        assert (await client.health())["status"] == "ok"
        assert client.status()["failures"] >= 1, client.status()
        body = sample_request(1)
        records = await client.predict_batch(body["points"], body["telemetry"])
        assert len(records) == 1 and isinstance(records[0]["prediction"], float), records
    finally:
        await client.aclose()
    print("   recovery:   same client succeeds after the service reappears")


async def test_timeout() -> None:
    url, server = start_stub("slow")
    try:
        client = MlClient(url, timeout_s=0.4, connect_timeout_s=1, failure_threshold=5)
        started = time.perf_counter()
        try:
            await client.predict_batch(ONE_POINT)
        except MlServiceError as exc:
            elapsed = time.perf_counter() - started
            assert exc.kind == "timeout", exc
        else:
            raise AssertionError("expected a timeout")
        assert elapsed < 4.0, f"waited {elapsed:.1f}s past the 0.4s budget"
    finally:
        await client.aclose()
        server.shutdown()
    print(f"   timeout:    abandoned after {elapsed:.2f}s instead of the stub's 5 s")


async def test_malformed_responses() -> None:
    cases = {
        "garbage": "non-JSON body",
        "wrong_shape": "record count does not match the request",
        "null_prediction": "non-numeric prediction",
    }
    for mode, description in cases.items():
        url, server = start_stub(mode)
        try:
            client = MlClient(url, timeout_s=5, failure_threshold=5)
            try:
                await client.predict_batch(ONE_POINT)
            except MlServiceError as exc:
                assert exc.kind == "malformed", (mode, exc.kind, exc)
            else:
                raise AssertionError(f"{mode}: expected a rejection")
        finally:
            await client.aclose()
            server.shutdown()
    print(f"   malformed:  rejected {len(cases)} bad shapes ({'; '.join(cases)})")


async def test_status_codes_preserved() -> None:
    for mode, kind, status in (("boom", "status", 500), ("loading", "loading", 503)):
        url, server = start_stub(mode)
        try:
            client = MlClient(url, timeout_s=5, failure_threshold=5)
            try:
                await client.predict_batch(ONE_POINT)
            except MlServiceError as exc:
                assert exc.kind == kind, (mode, exc.kind)
                assert exc.status == status, (mode, exc.status)
            else:
                raise AssertionError(f"{mode}: expected an error")
        finally:
            await client.aclose()
            server.shutdown()
    print("   statuses:   500 -> kind=status, 503 -> kind=loading (model loading)")


# --------------------------------------------------------------------------- #
# 10: the backend survives it
# --------------------------------------------------------------------------- #


async def test_backend_degradation() -> None:
    """The backend keeps serving when the ML service is gone."""
    import pandas as pd

    dead = f"http://127.0.0.1:{free_port()}"
    client = MlClient(dead, timeout_s=2, connect_timeout_s=1, failure_threshold=1, cooldown_s=30)
    service = InferenceService(PROJECT_ROOT, ml_client=client)
    try:
        # 1) A prediction attempt raises the typed error instead of crashing.
        try:
            await service.predict_records(
                service.points.head(2).to_dict("records"), service.traffic
            )
        except MlServiceError as exc:
            assert exc.kind in ("unavailable", "timeout"), exc
        else:
            raise AssertionError("expected a typed ML error")

        # 2) The replay stream keeps advancing instead of dying.
        sim = service.stream_simulator(speed=600.0)
        for _ in range(40):
            if sim._point_position >= len(sim._point_rows):
                break
            await sim.advance_to(pd.Timestamp(sim._point_rows[sim._point_position]["T"]))
        assert sim._ml_errors >= 1, "stream should have recorded the failure"
        assert sim._ml_last_error, "stream should have recorded why"
        assert sim.status()["simulated_time"] is not None, "stream stopped advancing"

        # 3) Everything that does not need inference still works.
        tr_id = int(service.points.iloc[0]["tr_id"])
        moment = pd.Timestamp(service.points.iloc[0]["T"])
        assert service.live_target(tr_id, moment)[2] in ("ok", "out_of_horizon", "no_schedule")
        assert service.schedule_for(tr_id) is not None
        assert service.cascade_snapshot({})["enabled"] in (True, False)
    finally:
        await client.aclose()
    print(f"   degradation: stream advanced, {sim._ml_errors} forecast(s) skipped, "
          "cascade/schedule/targets intact")


# --------------------------------------------------------------------------- #


async def run_all() -> None:
    base = start_service()
    print(f"[run ] ML service starting at {base}")
    try:
        health = await wait_for(f"{base}/health")
        assert health["status"] == "ok", health
        await test_health_over_http(base)
        await test_batch_predict(base)
        await test_batch_equivalence_over_http(base)
        await test_causality_over_http(base)
        # Recovery needs a live service on the far end, so it runs before the
        # shutdown: the point is that a client which has already failed against
        # a dead address works again once a real one is there.
        await test_recovery(base)
    finally:
        stop_service()

    await test_unavailable_and_breaker()
    await test_timeout()
    await test_malformed_responses()
    await test_status_codes_preserved()
    await test_backend_degradation()
    print("ALL ML INTEGRATION TESTS PASSED")


def main() -> None:
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
