from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.schedule_index import haversine_km, parse_point_wkt, schedule_index
from src.ml_client import MlClient, MlServiceError
from src.patterns import ROLE_QUALITY, ROLE_WEAK, split_events
from src.cascade import CascadeConfig, CascadeEngine

TELEMETRY_COLUMNS = [
    "tr_id",
    "event_time",
    "location_valid",
    "lon",
    "lat",
    "alt",
    "speed",
    "heading",
]
POINT_COLUMNS = [
    "sample_id",
    "tr_id",
    "T",
    "target_stop_id",
    "target_time_begin",
    "cur_dev_s",
]


RISK_COLORS = {"on-time": "#22c55e", "at-risk": "#eab308", "late": "#ef4444"}

# Nuisance dwell (seconds) attributed to a stop where the dwell detector fired.
# Fed into the cascade as a local source so the forward projection keeps
# regenerating delay the way a boarding surge actually does.
CASCADE_DWELL_OVERRUN_S = 30.0

#: Target window for live units, mirroring the horizon the model was trained on
#: (``lead_minus_660 = lead_s - 660`` in the feature list).  Used by the live
#: target search and kept here so the service and the cascade agree.
LIVE_TARGET_MIN_S = 600
LIVE_TARGET_MAX_S = 900

#: Offset (metres) past which a GPS fix is reported as ``on_route = False`` in
#: the map-matching result.  Urban bus telemetry drifts a few dozen metres off
#: the planned stop-to-stop corridor, so this is a "not on this route at all"
#: flag rather than a precision claim.
MAP_MATCH_TOLERANCE_M = 150.0

CAUSE_LABELS = {
    "stale": "телеметрия недостоверна (простой потока или потеря GPS)",
    "backlog": "накопленное отставание от графика сохраняется к целевой остановке",
    "dwell": "простой или посадка на подходе к целевой остановке",
    "speed_drop": "аномальное снижение скорости на подходе к участку",
}
CAUSE_FALLBACK = "отклонение операционного режима от планового графика"

#: Used when the only thing the detectors found is a data-quality problem: the
#: behaviour cause is then genuinely undetermined, and saying so is more useful to
#: a dispatcher than inventing a pattern.
CAUSE_UNKNOWN_UNTRUSTWORTHY = (
    "причина по характеру движения не определена: телеметрия недостоверна"
)

#: Reasons used when no pattern fired at all, kept with the quality role so the UI
#: renders them as a caveat about the data rather than as a detected pattern.
CAUSE_NO_TELEMETRY = "телеметрия отсутствует: прогноз опирается на плановое расписание и последнее известное отклонение"
CAUSE_NO_POSITION = "нет достоверных координат: траекторию и причину по движению определить нельзя"


def _window_gps_valid(rows: Iterable[Mapping[str, Any]], reference: Any) -> float | None:
    """Share of valid fixes over the last 300 s before ``reference``.

    Mirrors the model's ``gps_valid_300s`` feature so the freshness number shown
    to the dispatcher is the same one the detector threshold was applied to.
    Returns ``None`` when the window holds no rows at all -- "no data" is a
    different statement from "0% valid".
    """
    cutoff = None
    try:
        cutoff = pd.Timestamp(reference) - pd.Timedelta(seconds=300)
    except (TypeError, ValueError):
        return None
    total = 0
    valid = 0
    for row in rows or ():
        stamp = row.get("event_time")
        if stamp is None:
            continue
        try:
            if pd.Timestamp(stamp) < cutoff:
                continue
        except (TypeError, ValueError):
            continue
        total += 1
        if bool(row.get("location_valid", False)):
            valid += 1
    if total == 0:
        return None
    return valid / total


@dataclass(frozen=True, slots=True)
class CauseDiagnosis:
    """What the dispatcher is told, split by what the evidence actually supports.

    Attributes
    ----------
    cause:
        One-line statement of the primary reason.
    cause_role:
        ``strong_signal`` / ``weak_signal`` when a real pattern was found,
        ``data_quality`` when the reason is a telemetry caveat, ``none`` when
        nothing was detected.
    cause_type:
        Detector type behind :attr:`cause`, or ``None``.
    additional:
        Other cause events that fired but are not the primary reason.  These stay
        visible instead of being discarded by the ranking.
    quality:
        Telemetry trustworthiness summary -- status, the quality events and the
        measured freshness behind them.
    """

    cause: str
    cause_role: str
    cause_type: str | None
    additional: list[dict[str, Any]]
    quality: dict[str, Any]


def _event_view(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": event.get("type"),
        "role": event.get("role"),
        "confidence": _finite_float(event.get("confidence")) or 0.0,
        "reason": event.get("reason"),
    }


def _telemetry_quality(
    quality: list[Mapping[str, Any]],
    *,
    last_event_age_s: float | None = None,
    gps_valid: float | None = None,
) -> dict[str, Any]:
    """Summarise how far the telemetry behind this forecast can be trusted.

    Reported independently of the cause: a dispatcher needs to know both "the bus
    is running 4 minutes late" and "the last fix we have is 22 minutes old".
    """
    if quality:
        status = "degraded"
        note = "; ".join(str(e.get("reason", "")) for e in quality if e.get("reason"))
    else:
        status = "ok"
        note = "телеметрия свежая и достоверная"
    return {
        "status": status,
        "stale": bool(quality),
        "last_event_age_s": last_event_age_s,
        "gps_valid_300s": gps_valid,
        "events": [_event_view(e) for e in quality],
        "note": note,
    }


def diagnose_events(
    events: Any,
    *,
    last_event_age_s: float | None = None,
    gps_valid: float | None = None,
) -> CauseDiagnosis:
    """Split pattern events into a primary cause, extra signals and data quality.

    The two categories are never ranked against each other.  A ``stale`` feed
    sitting next to a ``backlog`` leaves the backlog as the primary cause and
    reports the staleness as a separate warning -- the previous implementation
    sorted quality events *first*, so a stale feed silently replaced the real
    reason on every incident where both fired.
    """
    causes, quality = split_events(events)
    quality_summary = _telemetry_quality(
        quality, last_event_age_s=last_event_age_s, gps_valid=gps_valid
    )
    if causes:
        primary = causes[0]
        return CauseDiagnosis(
            cause=CAUSE_LABELS.get(
                str(primary.get("type")), CAUSE_FALLBACK
            ),
            cause_role=str(primary.get("role", ROLE_WEAK)),
            cause_type=str(primary.get("type")),
            additional=[_event_view(e) for e in causes[1:]],
            quality=quality_summary,
        )
    if quality:
        # Only telemetry complaints: say the pattern is undetermined instead of
        # dressing a data problem up as a delay mechanism.
        return CauseDiagnosis(
            cause=CAUSE_UNKNOWN_UNTRUSTWORTHY,
            cause_role=ROLE_QUALITY,
            cause_type=None,
            additional=[],
            quality=quality_summary,
        )
    return CauseDiagnosis(
        cause=CAUSE_FALLBACK,
        cause_role="none",
        cause_type=None,
        additional=[],
        quality=quality_summary,
    )


def cause_from_events(events: Any) -> tuple[str, str]:
    """Backwards-compatible ``(cause, cause_role)`` view of :func:`diagnose_events`."""
    diagnosis = diagnose_events(events)
    return diagnosis.cause, diagnosis.cause_role


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _latest_position(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    for row in reversed(list(rows)):
        if not bool(row.get("location_valid", False)):
            continue
        lon = _finite_float(row.get("lon"))
        lat = _finite_float(row.get("lat"))
        if lon is None or lat is None:
            continue
        return {
            "lon": lon,
            "lat": lat,
            "event_time": _json_value(row.get("event_time")),
            "speed": _json_value(row.get("speed")),
            "heading": _json_value(row.get("heading")),
        }
    return None


def route_polyline(stops: Iterable[Mapping[str, Any]]) -> list[list[float]]:
    """Planned route geometry as ``[[lat, lon], ...]`` for map matching and drawing.

    Stops whose coordinates are missing or non-numeric are dropped so the result
    is always a usable chain of vertices; callers treat an empty result as
    "this vehicle has no drawable route".
    """
    polyline: list[list[float]] = []
    for stop in stops or ():
        if not isinstance(stop, Mapping):
            continue
        lat = _finite_float(stop.get("lat"))
        lon = _finite_float(stop.get("lon"))
        if lat is None or lon is None:
            continue
        polyline.append([lat, lon])
    return polyline


def snap_to_polyline(
    lon: Any, lat: Any, polyline: Iterable[Sequence[float]] | None
) -> dict[str, Any] | None:
    """Map-match one GPS fix onto a planned route polyline.

    Projects the fix onto the nearest segment of ``polyline`` (given as
    ``[[lat, lon], ...]``) and returns the matched point plus how far the raw
    fix sat from it.  This is deliberately 2-D point-to-polyline geometry: no
    road graph, no routing engine, no ``time_fact_begin``.

    The projection runs in raw lon/lat degrees -- the same planar space the
    polyline lives in -- while ``snap_distance_m`` is measured with the
    haversine so the caller gets real metres.  Segment selection therefore
    compares the fix against the route as drawn, which is what the map shows.

    Returns ``None`` -- never raises -- for degenerate input: unusable
    coordinates, fewer than two vertices, or a polyline of zero-length
    segments.  A fix that projects beyond the last vertex clamps to that vertex
    instead of extrapolating off the end of the route.
    """
    x = _finite_float(lon)
    y = _finite_float(lat)
    if x is None or y is None:
        return None

    vertices: list[tuple[float, float]] = []
    for entry in polyline or ():
        try:
            vertex_lon = entry[1]
            vertex_lat = entry[0]
        except (TypeError, IndexError, KeyError):
            continue
        vertex_lon = _finite_float(vertex_lon)
        vertex_lat = _finite_float(vertex_lat)
        if vertex_lon is None or vertex_lat is None:
            continue
        vertices.append((vertex_lon, vertex_lat))
    if len(vertices) < 2:
        return None

    best: tuple[float, int, float, float, float] | None = None
    for index in range(len(vertices) - 1):
        x1, y1 = vertices[index]
        x2, y2 = vertices[index + 1]
        dx = x2 - x1
        dy = y2 - y1
        span = dx * dx + dy * dy
        if span <= 0.0:
            t = 0.0
        else:
            t = ((x - x1) * dx + (y - y1) * dy) / span
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        px = x1 + t * dx
        py = y1 + t * dy
        offset = math.hypot(x - px, y - py)
        if best is None or offset < best[0]:
            best = (offset, index, t, px, py)

    _, segment_index, t, matched_lon, matched_lat = best
    distance_m = float(
        haversine_km(np.array([x]), np.array([y]), matched_lon, matched_lat)[0]
    ) * 1000.0
    return {
        "lat": round(matched_lat, 7),
        "lon": round(matched_lon, 7),
        "snap_distance_m": round(distance_m, 1),
        "segment_index": segment_index,
        "segment_t": round(t, 4),
    }


def _route_network(schedule: pd.DataFrame) -> dict[int, dict[str, Any]]:
    frame = schedule.copy()
    frame["time_begin"] = pd.to_datetime(frame["time_begin"], errors="coerce")
    frame["tr_id"] = pd.to_numeric(frame["tr_id"], errors="coerce")
    frame = frame.dropna(subset=["tr_id"]).copy()
    frame["tr_id"] = frame["tr_id"].astype(int)
    parsed = frame["geom"].map(parse_point_wkt)
    frame["lon"] = [value[0] for value in parsed]
    frame["lat"] = [value[1] for value in parsed]
    frame = frame.sort_values(["tr_id", "time_begin", "tt_action_item_id"], kind="stable")
    routes: dict[int, dict[str, Any]] = {}
    for vehicle_id, group in frame.groupby("tr_id", sort=False):
        stops = [
            {
                "stop_id": _json_value(row.get("tt_action_item_id")),
                "time": _json_value(row.get("time_begin")),
                "address": _json_value(row.get("building_address")),
                "lon": _json_value(row.get("lon")),
                "lat": _json_value(row.get("lat")),
            }
            for _, row in group.iterrows()
        ]
        routes[int(vehicle_id)] = {"tr_id": int(vehicle_id), "stops": stops}
    return routes


def _empty_traffic() -> pd.DataFrame:
    return pd.DataFrame(columns=TELEMETRY_COLUMNS)


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Parse a datetime column only when it is not already one.

    ``pd.to_datetime`` on a 100k-row string column costs ~150 ms and was being
    redone on every request for data that never changes.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def _ensure_numeric(series: pd.Series) -> pd.Series:
    """Coerce a column to numeric only when it is not already numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def _normalise_telemetry(
    traffic: pd.DataFrame | Iterable[dict[str, Any]] | None,
    *,
    already_sorted: bool = False,
) -> pd.DataFrame:
    """Coerce arbitrary telemetry into the canonical, causal-ready frame.

    The result is the single representation every downstream stage expects:
    typed columns, ``tr_id`` as int64, ``event_time`` as datetime64, sorted by
    ``event_time``.  Pass ``already_sorted=True`` when the caller appends to a
    frame that is already normalised and sorted (the replay simulator does), and
    the redundant sort is skipped.

    ``NaT`` timestamps are dropped here rather than downstream: a row with an
    unparseable ``event_time`` can never pass the ``event_time <= T`` cut, so
    keeping it only inflates the frame and the window scans.
    """
    if traffic is None:
        return _empty_traffic()
    frame = traffic.copy() if isinstance(traffic, pd.DataFrame) else pd.DataFrame(list(traffic))
    if frame.empty:
        return _empty_traffic()
    for column in TELEMETRY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["tr_id"] = _ensure_numeric(frame["tr_id"])
    frame = frame.dropna(subset=["tr_id"])
    frame["tr_id"] = frame["tr_id"].astype("int64")
    frame["event_time"] = _ensure_datetime(frame["event_time"])
    frame = frame.dropna(subset=["event_time"])
    frame["speed"] = _ensure_numeric(frame["speed"])
    frame["lon"] = _ensure_numeric(frame["lon"])
    frame["lat"] = _ensure_numeric(frame["lat"])
    if frame["location_valid"].dtype == object:
        frame["location_valid"] = frame["location_valid"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    else:
        frame["location_valid"] = frame["location_valid"].fillna(False).astype(bool)
    frame = frame[TELEMETRY_COLUMNS]
    if already_sorted:
        return frame.reset_index(drop=True)
    return frame.sort_values("event_time", kind="stable").reset_index(drop=True)


def _normalise_points(points: pd.DataFrame | Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = points.copy() if isinstance(points, pd.DataFrame) else pd.DataFrame(list(points))
    if frame.empty:
        raise ValueError("points must not be empty")
    for column in POINT_COLUMNS:
        if column not in frame.columns:
            if column == "sample_id":
                frame[column] = ""
            else:
                raise ValueError(f"missing point column: {column}")
    frame["tr_id"] = pd.to_numeric(frame["tr_id"], errors="raise").astype(int)
    frame["target_stop_id"] = pd.to_numeric(frame["target_stop_id"], errors="raise").astype("int64")
    frame["cur_dev_s"] = pd.to_numeric(frame["cur_dev_s"], errors="raise").astype(float)
    frame["T"] = pd.to_datetime(frame["T"], errors="raise")
    frame["target_time_begin"] = pd.to_datetime(frame["target_time_begin"], errors="raise")
    generated = frame["sample_id"].isna() | frame["sample_id"].astype(str).eq("")
    if generated.any():
        generated_ids = [
            f"{int(tr_id)}_{int(timestamp.timestamp())}"
            for tr_id, timestamp in zip(frame.loc[generated, "tr_id"], frame.loc[generated, "T"])
        ]
        frame.loc[generated, "sample_id"] = generated_ids
    frame["sample_id"] = frame["sample_id"].astype(str)
    return frame[POINT_COLUMNS].reset_index(drop=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (str, bytes, list, tuple, dict)) else False:
        return None
    return value


def _has_event(record: Mapping[str, Any], event_type: str) -> bool:
    """True when a prediction record carries a pattern event of the given type."""
    events = record.get("pattern_events")
    if not isinstance(events, list):
        return False
    return any(isinstance(item, dict) and item.get("type") == event_type for item in events)


def _record_for_response(row: pd.Series) -> dict[str, Any]:
    return {
        "sample_id": _json_value(row.get("sample_id")),
        "tr_id": int(row["tr_id"]),
        "T": _json_value(row.get("T")),
        "target_stop_id": _json_value(row.get("target_stop_id")),
        "target_time_begin": _json_value(row.get("target_time_begin")),
        "prediction": float(row["prediction"]),
        "target_class": str(row["target_class"]),
        "risk": str(row["risk"]),
        "recommendation": str(row["recommendation"]),
        "p_late": _json_value(row.get("p_late")),
        "pattern_events": row.get("pattern_events") if isinstance(row.get("pattern_events"), list) else [],
    }


def _append_sorted(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two event_time-sorted telemetry frames cheaply.

    Both inputs come from the replay simulator in chronological order, so the
    common case is a plain append; a real sort is only paid when a chunk starts
    earlier than the buffer tail (clock skew, out-of-order emulator frames).
    """
    if left is None or len(left) == 0:
        return right.reset_index(drop=True)
    if right is None or len(right) == 0:
        return left
    if right["event_time"].iloc[0] >= left["event_time"].iloc[-1]:
        merged = pd.concat([left, right], ignore_index=True)
    else:
        merged = pd.concat([left, right], ignore_index=True).sort_values(
            "event_time", kind="stable"
        )
    return merged.reset_index(drop=True)


def _safe_int(value: Any, default: int = -1) -> int:
    """Best-effort int coercion that never raises (live units may send junk)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _empty_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["tr_id", "tt_action_item_id", "time_begin", "stop_lon", "stop_lat", "stop_idx"]
    )


def _build_live_target_index(schedule: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Pre-index planned arrivals per vehicle as sorted numpy arrays.

    The live endpoint used to filter and re-sort the whole 5 558-row schedule
    with pandas for *every connected unit on every request* -- ~1.5 ms per unit,
    so ~45 ms for a 30-bus fleet, all of it repeated work.  With per-vehicle
    ``int64`` nanosecond arrays the same lookup is a ``searchsorted``.
    """
    index: dict[int, dict[str, Any]] = {}
    if schedule is None or len(schedule) == 0:
        return index
    frame = schedule[["tr_id", "tt_action_item_id", "time_begin"]].copy()
    frame = frame.dropna(subset=["time_begin", "tt_action_item_id"])
    for tr_id, group in frame.groupby("tr_id", sort=False):
        times = group["time_begin"].to_numpy(dtype="datetime64[ns]").astype("int64")
        stops = group["tt_action_item_id"].to_numpy(dtype="int64")
        order = np.argsort(times, kind="stable")
        index[int(tr_id)] = {
            "times_ns": np.ascontiguousarray(times[order]),
            "stops": np.ascontiguousarray(stops[order]),
        }
    return index


class InferenceService:
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        dataset_dir: str | Path | None = None,
        ml_client: MlClient | None = None,
    ) -> None:
        project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
        self.root = project_root.resolve()
        dataset_value = dataset_dir if dataset_dir is not None else os.getenv("DATASET_DIR", "dataset")
        self.dataset_dir = self._path(dataset_value)
        self.points = self._read_csv(self.dataset_dir / "validate" / "points.csv")
        self.traffic = self._read_csv(self.dataset_dir / "validate" / "traffic.csv")
        self.schedule = self._read_csv(self.dataset_dir / "validate" / "schedule_plan.csv")
        # The model lives in the ML service, not here.  This service holds only
        # an HTTP client, so importing this module pulls in no ML code at all.
        self.ml = ml_client if ml_client is not None else MlClient()
        # Normalise the static telemetry once. The normalised frame *replaces*
        # the raw read rather than sitting next to it, so this costs no extra
        # resident memory and removes ~150 ms of parsing from every request.
        self.traffic = _normalise_telemetry(self.traffic)
        # Static timetable view (WKT parsing, sorting, inter-stop geometry) is
        # likewise derived once and memoised for the whole process.
        self.schedule_index = schedule_index(self.schedule)
        self._live_target_index = _build_live_target_index(self.schedule_index.prepared)
        # Cascading-delay engine. The graph (one vertex per planned stop-visit
        # plus corridor transfer links) is built lazily on first use, because it
        # costs ~1-2 s on a full extract and /health must not pay for it.
        self.cascade_config = CascadeConfig.for_regime(
            os.getenv("CASCADE_REGIME", "normal")
        )
        self.cascade = CascadeEngine(self.cascade_config)
        self._cascade_ready = False
        self.cascade_horizon = max(1, int(os.getenv("CASCADE_HORIZON", "6")))

    def cascade_engine(self) -> CascadeEngine:
        """Return the cascade engine, building the route graph on first call.

        The graph is derived from the *plan* only (``time_begin``), never from
        ``time_fact_begin``, so it is safe for the live and validate contours.
        """
        if not self._cascade_ready:
            self.cascade.attach_schedule(self.schedule)
            self._cascade_ready = True
        return self.cascade

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, low_memory=False)

    def p_late_for(self, delay_s: float) -> float | None:
        """Deprecated shim: calibration now lives in the ML service.

        Kept so existing call sites do not break; returns ``None`` because the
        curve is no longer loaded in this process.  ``predict_frame`` gets
        ``p_late`` straight from the service response.
        """
        return None

    async def predict_frame(
        self,
        points: pd.DataFrame | Iterable[dict[str, Any]],
        traffic: pd.DataFrame | Iterable[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Predict for a set of points by calling the ML service.

        The backend no longer builds features or runs the model: the whole
        pipeline (feature engineering, CatBoost forward pass, risk / class /
        recommendation, calibrated probability, pattern detectors) lives behind
        the service's ``POST /predict``.  This method keeps the call-site shape
        the rest of the backend already uses -- a frame in, a frame out -- so
        nothing above it had to change.

        Batching note: passing many points with different ``T`` is *safe* and is
        the intended usage.  The authoritative per-point causal cut
        (``event_time <= T``) is applied inside the ML service's feature builder;
        the ``max_time`` filter here is only a coarse pre-trim that keeps the
        request payload small.  Equivalence between single-row and batched
        prediction is asserted in ``ml/test_batching.py`` (max abs difference
        0.0).

        Raises :class:`~src.ml_client.MlServiceError` if the service is
        unreachable, times out or answers with something unusable.
        """
        point_frame = _normalise_points(points)
        if traffic is None:
            # Fast path: the service's telemetry is already normalised and sorted.
            telemetry = self.traffic
        else:
            telemetry = _normalise_telemetry(traffic)
        max_time = point_frame["T"].max()
        ids = set(point_frame["tr_id"].tolist())
        causal_traffic = telemetry[
            telemetry["tr_id"].isin(ids) & (telemetry["event_time"] <= max_time)
        ]
        records = await self.ml.predict_batch(
            point_frame.to_dict("records"), causal_traffic
        )
        result = point_frame.copy()
        result["prediction"] = [float(record["prediction"]) for record in records]
        result["target_class"] = [record.get("target_class") for record in records]
        result["risk"] = [record.get("risk") for record in records]
        result["recommendation"] = [record.get("recommendation") for record in records]
        result["p_late"] = [record.get("p_late") for record in records]
        result["pattern_events"] = [record.get("pattern_events") or [] for record in records]
        return result

    def live_target(self, tr_id: int, event_time: Any) -> tuple[Any, Any, str]:
        """First planned stop whose arrival falls in ``(T+600, T+900]``.

        O(log n) per call: binary search on the pre-indexed arrival times of the
        vehicle.  Mirrors the previous pandas implementation exactly, including
        the ``no_schedule`` / ``no_time`` / ``out_of_horizon`` statuses that the
        live UI renders verbatim.
        """
        entry = self._live_target_index.get(_safe_int(tr_id))
        if entry is None:
            return None, None, "no_schedule"
        try:
            reference = pd.to_datetime(pd.Timestamp(event_time).tz_localize(None) if pd.Timestamp(event_time).tzinfo else pd.Timestamp(event_time))
        except (TypeError, ValueError):
            return None, None, "no_time"
        if reference is None or pd.isna(reference):
            return None, None, "no_time"
        times = entry["times_ns"]
        reference_ns = int(reference.value)
        start = int(np.searchsorted(times, reference_ns + LIVE_TARGET_MIN_S * 1_000_000_000, side="right"))
        stop = int(np.searchsorted(times, reference_ns + LIVE_TARGET_MAX_S * 1_000_000_000, side="right"))
        if stop <= start:
            return None, None, "out_of_horizon"
        target_id = int(entry["stops"][start])
        return target_id, pd.Timestamp(times[start]), "ok"

    def schedule_for(self, tr_id: int) -> pd.DataFrame:
        """Planned timetable of one vehicle, from the pre-computed index."""
        return self.schedule_index.by_vehicle.get(_safe_int(tr_id), _empty_schedule())

    def _eta_projection(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        """Forward ETA projection of one prediction along its own route.

        The predicted delay at ``target_stop_id`` is pushed through the cascade
        chain so a consumer can see how far down the route the lateness reaches
        and where the fleet claws it back.  A detected ``dwell`` pattern event at
        the origin is folded in as a nuisance source for the following stops.

        Returns an empty list when the vehicle is not in the plan or the
        prediction is not positive -- never raises, it sits in the request path.
        """
        delay = record.get("prediction")
        try:
            delay = float(delay)
        except (TypeError, ValueError):
            return []
        if not delay > 0.0:
            return []
        overrun = CASCADE_DWELL_OVERRUN_S if _has_event(record, "dwell") else None
        return self.cascade_engine().project(
            record.get("tr_id"),
            record.get("target_stop_id"),
            delay,
            horizon=self.cascade_horizon,
            dwell_overrun_s=overrun,
        )

    async def predict_records(
        self,
        points: Iterable[dict[str, Any]],
        shared_telemetry: Iterable[dict[str, Any]] | None = None,
        with_cascade: bool = True,
    ) -> list[dict[str, Any]]:
        """Predict for a batch of points and optionally project ETAs forward.

        Async because inference now crosses a network boundary.  The cascade
        projection stays here: it is business logic over the fleet, not
        inference, and it has no business in the ML service.

        Parameters
        ----------
        points : iterable of dict
            Forecast points; a point may carry its own ``telemetry`` list.
        shared_telemetry : DataFrame or iterable of dict, optional
            Telemetry shared by every point in the batch.  A DataFrame is passed
            through untouched -- :meth:`predict_frame` already accepts one, and
            the micro-batcher hands over the merged frame.  Note that a DataFrame
            must never be tested for truthiness: ``frame or []`` raises
            ``ValueError: The truth value of a DataFrame is ambiguous``, which
            used to make every telemetry-carrying live prediction fail.
        with_cascade : bool
            When true (default) each record gains an ``eta_projection`` list: the
            per-stop ETA the cascade model expects from ``target_stop_id``
            onwards.  Set false to skip it on hot paths that only need the point
            prediction.

        Returns
        -------
        list of dict
            Prediction records in the input order.
        """
        point_rows: list[dict[str, Any]] = []
        local_rows: list[dict[str, Any]] = []
        for point in points:
            row = dict(point)
            local = row.pop("telemetry", None)
            if local is not None:
                local_rows.extend(dict(event) for event in local)
            point_rows.append(row)
        if isinstance(shared_telemetry, pd.DataFrame):
            telemetry: Any = shared_telemetry
            if local_rows:
                telemetry = pd.concat(
                    [shared_telemetry, pd.DataFrame(local_rows)], ignore_index=True
                )
        else:
            merged = list(shared_telemetry or [])
            merged.extend(local_rows)
            telemetry = merged or None
        result = await self.predict_frame(point_rows, telemetry)
        records = [_record_for_response(row) for _, row in result.iterrows()]
        if with_cascade:
            # The cascade projection is the only CPU-bound step left in this
            # process, so it goes to a thread rather than stalling the event
            # loop (and therefore the NDTP receiver and the status endpoints).
            records = await asyncio.to_thread(self._attach_projections, records)
        return records

    def _attach_projections(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for record in records:
            record["eta_projection"] = self._eta_projection(record)
        return records

    def cascade_snapshot(self, vehicles: Mapping[int, dict[str, Any]]) -> dict[str, Any]:
        """Solve the network cascade for the current fleet state.

        Each vehicle's predicted delay at its target stop becomes an injection at
        the matching graph vertex; a detected ``dwell`` event becomes a nuisance
        source on that stop.  The returned payload is the JSON-ready summary plus
        a per-route projection of the first ``cascade_horizon`` stops.

        Parameters
        ----------
        vehicles : mapping
            ``tr_id -> record`` as held by the stream simulator.

        Returns
        -------
        dict
            ``{"enabled": bool, ...}``.  ``enabled`` is false when the plan could
            not be turned into a graph (no schedule rows, no geometry) -- the
            rest of the service keeps working, it simply has no cascade view.
        """
        engine = self.cascade_engine()
        if not engine.graph.stops:
            return {"enabled": False, "reason": "no planned stop-visits", "vehicles": []}
        injections: dict[tuple[int, int], float] = {}
        overruns: dict[tuple[int, int], float] = {}
        for tr_id, record in (vehicles or {}).items():
            try:
                delay = float(record.get("prediction"))
            except (TypeError, ValueError):
                continue
            if not delay > 0.0:
                continue
            key = (int(tr_id), int(record.get("target_stop_id")))
            if engine.graph.vertex_for(*key) < 0:
                continue
            injections[key] = delay
            if _has_event(record, "dwell"):
                overruns[key] = CASCADE_DWELL_OVERRUN_S
        solved = engine.solve(injections, overruns)
        summary = solved.summary()
        routes: list[dict[str, Any]] = []
        for tr_id in summary["injected_routes"]:
            routes.append(
                {
                    "tr_id": tr_id,
                    "stops": solved.route_projection(tr_id, horizon=self.cascade_horizon),
                    "absorption": solved.absorption(tr_id),
                }
            )
        summary["enabled"] = True
        summary["vehicles"] = routes
        return summary

    async def generate_submission(self, output_path: str | Path | None = None) -> dict[str, Any]:
        """Recompute ``submission.csv`` for the validate period via the ML service."""
        result = await self.predict_frame(self.points, self.traffic)
        expected = self.points["sample_id"].astype(str)
        actual = result["sample_id"].astype(str)
        if len(result) != len(self.points):
            raise ValueError("submission coverage mismatch")
        if actual.duplicated().any() or not actual.equals(expected.reset_index(drop=True)):
            raise ValueError("submission sample_id coverage mismatch")
        if not np.isfinite(result["prediction"].to_numpy(dtype=float)).all():
            raise ValueError("submission contains non-finite predictions")
        destination = Path(output_path) if output_path is not None else self.submission_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        submission = result[["sample_id", "prediction"]].copy()
        submission.to_csv(destination, index=False, sep=";", float_format="%.6f")
        return {
            "path": str(destination),
            "rows": len(submission),
            "first_rows": [
                {"sample_id": str(row["sample_id"]), "prediction": float(row["prediction"])}
                for _, row in submission.head(5).iterrows()
            ],
        }

    def submission_path(self) -> Path:
        """Where ``/generate_submission`` writes its artifact.

        Configurable through ``SUBMISSION_PATH`` so the container can point it at
        a mounted volume instead of its own writable layer.  Without that the
        file would land in the container's ephemeral filesystem and be lost on
        recreate -- which is also why the backend container no longer bind-mounts
        the repository: it must not need host files to run.
        """
        configured = os.getenv("SUBMISSION_PATH", "").strip()
        if configured:
            path = Path(configured)
            return path if path.is_absolute() else self.root / path
        return self.root / "submission.csv"

    def stream_simulator(self, speed: float = 60.0) -> StreamSimulator:
        return StreamSimulator(self, speed=speed)


class StreamSimulator:
    def __init__(self, service: InferenceService, *, speed: float = 60.0) -> None:
        self.service = service
        self.traffic = _normalise_telemetry(service.traffic)
        self.points = service.points.copy()
        self.points["T_dt"] = pd.to_datetime(self.points["T"], errors="coerce")
        self.points["_point_order"] = np.arange(len(self.points), dtype=np.int64)
        self.points = self.points.sort_values(["T_dt", "_point_order"], kind="stable").reset_index(drop=True)
        self._traffic_rows = self.traffic[TELEMETRY_COLUMNS].to_dict("records")
        self._point_rows = self.points.drop(columns=["T_dt", "_point_order"], errors="ignore").to_dict("records")
        self._vehicles: dict[int, dict[str, Any]] = {}
        self._vehicle_telemetry: dict[int, list[dict[str, Any]]] = {}
        # Degradation counters for an unreachable ML service, surfaced in status().
        self._ml_errors = 0
        self._ml_last_error: str | None = None
        self._visible_traffic: list[dict[str, Any]] = []
        # Normalised mirror of ``_visible_traffic``, extended incrementally so
        # the replay loop never re-parses the whole frame.
        self._normalised_traffic: pd.DataFrame | None = None
        self._normalised_upto: int = 0
        self._routes = _route_network(service.schedule)
        self.speed = self._safe_speed(speed)
        self._reset_state()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        # Latest network cascade view; recomputed lazily in status() so the
        # 10 Hz replay tick does not re-solve an unchanged fleet.
        self._cascade: dict[str, Any] | None = None
        self._cascade_dirty = True

    @staticmethod
    def _safe_speed(value: float) -> float:
        speed = float(value)
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("speed must be positive")
        return min(speed, 10000.0)

    def set_speed(self, speed: float) -> None:
        self.speed = self._safe_speed(speed)

    def _remember_telemetry(self, row: dict[str, Any]) -> None:
        self._visible_traffic.append(row)
        raw_vehicle_id = row.get("tr_id")
        if raw_vehicle_id is None:
            return
        try:
            vehicle_id = int(raw_vehicle_id)
        except (TypeError, ValueError):
            return
        self._vehicle_telemetry.setdefault(vehicle_id, []).append(row)

    def _reset_state(self) -> None:
        self._traffic_position = 0
        self._point_position = 0
        self._visible_traffic = []
        self._normalised_traffic = None
        self._normalised_upto = 0
        self._vehicle_telemetry = {}
        self._vehicles = {}
        if hasattr(self, "_cascade"):
            self._cascade = None
            self._cascade_dirty = True
        point_times = [value for value in self.points["T_dt"].tolist() if pd.notna(value)]
        traffic_times = [value for value in self.traffic["event_time"].tolist() if pd.notna(value)]
        candidates = point_times or traffic_times
        self.current_time = min(candidates) if candidates else pd.Timestamp.now()
        while self._traffic_position < len(self._traffic_rows):
            event_time = pd.Timestamp(self._traffic_rows[self._traffic_position]["event_time"])
            if event_time > self.current_time:
                break
            self._remember_telemetry(self._traffic_rows[self._traffic_position])
            self._traffic_position += 1
        self.finished = False

    def reset(self, speed: float | None = None) -> None:
        self._reset_state()
        if speed is not None:
            self.set_speed(speed)

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        previous_wall = time.monotonic()
        while not self._stop_event.is_set() and not self.finished:
            await asyncio.sleep(0.1)
            current_wall = time.monotonic()
            elapsed = current_wall - previous_wall
            previous_wall = current_wall
            await self.advance_to(
                self.current_time + pd.Timedelta(seconds=elapsed * self.speed)
            )

    def _causal_history(self, vehicle_id: int, forecast_time: pd.Timestamp) -> list[dict[str, Any]]:
        return [
            row
            for row in self._vehicle_telemetry.get(vehicle_id, [])
            if pd.Timestamp(row.get("event_time")) <= forecast_time
        ]

    def _vehicle_schedule(self, vehicle_id: int) -> pd.DataFrame:
        """Planned timetable of one vehicle.

        Delegates to the service's pre-computed schedule index, which already
        holds the parsed coordinates and is sorted by plan time.  The previous
        implementation copied and re-sorted the full 5 558-row schedule for
        *every* forecast point.
        """
        return self.service.schedule_for(vehicle_id)

    @staticmethod
    def _stop_card(row: pd.Series | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "stop_id": _json_value(row.get("tt_action_item_id")),
            "time": _json_value(row.get("time_begin")),
            "address": _json_value(row.get("building_address")),
            "lon": _json_value(row.get("lon")),
            "lat": _json_value(row.get("lat")),
        }

    def _snapshot_for_point(
        self, point: dict[str, Any], events: list | None = None
    ) -> dict[str, Any]:
        vehicle_id = int(point["tr_id"])
        forecast_time = pd.Timestamp(point["T"])
        target_id = int(point["target_stop_id"])
        target_time = pd.Timestamp(point["target_time_begin"])
        cur_dev_raw = point.get("cur_dev_s")
        current_deviation = _finite_float(cur_dev_raw) or 0.0
        cur_dev_hint = "none" if cur_dev_raw is None else "input"
        history = self._causal_history(vehicle_id, forecast_time)
        position = _latest_position(history)
        last_time = pd.Timestamp(history[-1]["event_time"]) if history else None
        last_age = float((forecast_time - last_time).total_seconds()) if last_time is not None else None
        window_start = forecast_time - pd.Timedelta(seconds=600)
        window_speeds = [
            speed
            for row in history
            if pd.Timestamp(row.get("event_time")) >= window_start
            for speed in [_finite_float(row.get("speed"))]
            if speed is not None
        ]
        average_speed = float(sum(window_speeds) / len(window_speeds)) if window_speeds else None
        schedule = self._vehicle_schedule(vehicle_id)
        target_row = schedule[pd.to_numeric(schedule["tt_action_item_id"], errors="coerce") == target_id]
        target = self._stop_card(target_row.iloc[0] if not target_row.empty else None)
        target_lon = _finite_float(target.get("lon")) if target else None
        target_lat = _finite_float(target.get("lat")) if target else None
        distance = None
        if position and target_lon is not None and target_lat is not None:
            distance = float(haversine_km([position["lon"]], [position["lat"]], target_lon, target_lat)[0])
        matched_position = self._match_position(vehicle_id, position)
        remaining = schedule[(schedule["time_begin"] > forecast_time) & (schedule["time_begin"] <= target_time)]
        gaps = pd.to_numeric(remaining["time_begin"].diff().dt.total_seconds(), errors="coerce").dropna()
        previous = schedule[schedule["time_begin"] <= forecast_time].tail(1)
        following = schedule[schedule["time_begin"] > forecast_time].head(1)
        target_previous = schedule[schedule["time_begin"] < target_time].tail(1)
        # Cause comes from the measured pattern detectors; the hand-rolled
        # speed/deviation heuristics stay only as a fallback for rows the
        # detectors could not score (no history at all).  Causes and data-quality
        # signals are ranked separately, so a stale feed never displaces a real
        # pattern as the reason.
        diagnosis = diagnose_events(
            events,
            last_event_age_s=last_age,
            gps_valid=_window_gps_valid(history, forecast_time),
        )
        if diagnosis.cause_role == "none":
            if not history:
                diagnosis = replace(
                    diagnosis, cause=CAUSE_NO_TELEMETRY, cause_role=ROLE_QUALITY
                )
            elif position is None:
                diagnosis = replace(
                    diagnosis, cause=CAUSE_NO_POSITION, cause_role=ROLE_QUALITY
                )
        return {
            "forecast_time": _json_value(forecast_time),
            "target": target,
            "target_time": _json_value(target_time),
            "current_deviation_s": current_deviation,
            "position": position,
            "last_event_age_s": last_age,
            "telemetry_points": float(len(history)),
            "average_speed_10m_kmh": average_speed,
            "distance_to_target_km": distance,
            "matched_position": matched_position,
            "planned_stops_remaining": len(remaining),
            "remaining_gap_mean_s": float(gaps.mean()) if not gaps.empty else None,
            "current_segment": {
                "previous": self._stop_card(previous.iloc[0] if not previous.empty else None),
                "next": self._stop_card(following.iloc[0] if not following.empty else None),
            },
            "target_segment": {
                "previous": self._stop_card(target_previous.iloc[0] if not target_previous.empty else None),
                "target": target,
            },
            "cause": diagnosis.cause,
            "cause_role": diagnosis.cause_role,
            "cause_type": diagnosis.cause_type,
            "additional_causes": diagnosis.additional,
            "telemetry_quality": diagnosis.quality,
            "pattern_events": events or [],
            "cur_dev_hint": cur_dev_hint,
        }

    def _select_incident(self) -> dict[str, Any] | None:
        candidates = [
            record
            for record in self._vehicles.values()
            if isinstance(record.get("snapshot"), dict) and float(record.get("prediction", 0.0)) >= 60.0
        ]
        if not candidates:
            return None
        record = max(candidates, key=lambda item: float(item.get("prediction", 0.0)))
        snapshot = record["snapshot"]
        vehicle_id = int(record["tr_id"])
        target_time = pd.Timestamp(record["target_time_begin"])
        prediction = float(record["prediction"])
        arrival = target_time + pd.Timedelta(seconds=prediction) if pd.notna(target_time) else None
        return {
            "tr_id": vehicle_id,
            "sample_id": record.get("sample_id"),
            "forecast_time": record.get("T"),
            "target_stop_id": record.get("target_stop_id"),
            "target_time": record.get("target_time_begin"),
            "predicted_delay_s": prediction,
            "predicted_arrival": _json_value(arrival),
            "risk": record.get("risk"),
            "recommendation": record.get("recommendation"),
            "cause": snapshot.get("cause"),
            "cause_role": snapshot.get("cause_role"),
            "cause_type": snapshot.get("cause_type"),
            "additional_causes": snapshot.get("additional_causes") or [],
            "telemetry_quality": snapshot.get("telemetry_quality")
            or {"status": "unknown", "stale": False, "events": [], "note": ""},
            "p_late": record.get("p_late"),
            "cur_dev_hint": snapshot.get("cur_dev_hint"),
            "position_now": _latest_position(self._vehicle_telemetry.get(vehicle_id, [])),
            "snapshot": snapshot,
        }

    async def advance_to(self, target_time: pd.Timestamp) -> None:
        """Advance the replay clock, ingesting telemetry and scoring points.

        Forecast points that fall due in the same tick are scored in **one
        batch** rather than one call per point.  That is the single largest
        latency win in the replay path: feature building carries a large fixed
        cost per call (schedule lookup, telemetry slicing, dataframe merge), so
        a 10-point tick costs about the same as a 1-point tick.  The batch is
        causally sound because the per-point ``event_time <= T`` cut happens
        inside the feature builder, not here.

        Async because the scoring call is an HTTP round trip to the ML service.
        An unreachable service costs this tick's forecast, not the stream.
        """
        target = pd.Timestamp(target_time)
        appended = False
        while self._traffic_position < len(self._traffic_rows):
            event_time = pd.Timestamp(self._traffic_rows[self._traffic_position]["event_time"])
            if event_time > target:
                break
            self._remember_telemetry(self._traffic_rows[self._traffic_position])
            self._traffic_position += 1
            appended = True
        if appended:
            # Append only the new rows to the already-normalised buffer: no
            # re-parse, no re-sort of the 100k-row frame.
            new_rows = self._visible_traffic[self._normalised_upto :]
            if new_rows:
                chunk = _normalise_telemetry(new_rows)
                self._normalised_traffic = (
                    chunk
                    if self._normalised_traffic is None
                    else _append_sorted(self._normalised_traffic, chunk)
                )
                self._normalised_upto = len(self._visible_traffic)

        due: list[dict[str, Any]] = []
        while self._point_position < len(self._point_rows):
            point = self._point_rows[self._point_position]
            point_time = pd.Timestamp(point["T"])
            if point_time > target:
                break
            due.append(point)
            self._point_position += 1

        if due:
            traffic = (
                self._normalised_traffic
                if self._normalised_traffic is not None
                else _normalise_telemetry(self._visible_traffic)
            )
            if traffic is None or len(traffic) == 0:
                batched = pd.DataFrame()
            else:
                try:
                    batched = await self.service.predict_frame(due, traffic)
                except MlServiceError as exc:
                    # The replay clock keeps moving and the NDTP buffer keeps
                    # filling; only the forecast for this tick is lost.  Skipping
                    # is strictly better than killing the stream task, and the
                    # points stay consumed so they are not re-requested forever.
                    self._ml_errors += 1
                    self._ml_last_error = str(exc)
                    batched = pd.DataFrame()
            for position, point in enumerate(due):
                if position >= len(batched):
                    break
                record = _record_for_response(batched.iloc[position])
                events = record.get("pattern_events") or []
                record["snapshot"] = self._snapshot_for_point(point, events)
                self._vehicles[int(record["tr_id"])] = record
            self._cascade_dirty = True

        if self._traffic_position >= len(self._traffic_rows) and self._point_position >= len(self._point_rows):
            self.current_time = target
            self.finished = True
        else:
            self.current_time = target

    def cascade_view(self) -> dict[str, Any]:
        """Return the network cascade view, re-solving only when the fleet moved.

        The replay loop ticks ten times a second while the fleet changes only when
        a new forecast point is reached, so the solve is gated behind a dirty flag
        to keep ``/stream/status`` cheap.
        """
        if self._cascade_dirty or self._cascade is None:
            try:
                self._cascade = self.service.cascade_snapshot(self._vehicles)
            except Exception:  # noqa: BLE001 - cascade must never break the stream
                self._cascade = {"enabled": False, "reason": "cascade solve failed"}
            self._cascade_dirty = False
        return self._cascade

    def _match_position(
        self, vehicle_id: int, position: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        """Snap a vehicle's latest fix onto its planned route polyline.

        Returns the matched point together with the raw fix and the offset, or
        ``None`` when the vehicle has no planned route to match against -- the
        honest outcome for a live NDTP unit without a ``tr_id`` mapping, where
        there is no schedule geometry to project onto.
        """
        if not position:
            return None
        route = self._routes.get(vehicle_id)
        if route is None:
            return None
        polyline = route_polyline(route["stops"])
        if len(polyline) < 2:
            return None
        matched = snap_to_polyline(position.get("lon"), position.get("lat"), polyline)
        if matched is None:
            return None
        return {
            "raw": {"lon": position.get("lon"), "lat": position.get("lat")},
            "lon": matched["lon"],
            "lat": matched["lat"],
            "snap_distance_m": matched["snap_distance_m"],
            "segment_index": matched["segment_index"],
            "segment_t": matched["segment_t"],
            "on_route": matched["snap_distance_m"] <= MAP_MATCH_TOLERANCE_M,
        }

    def status(self) -> dict[str, Any]:
        vehicles = [self._vehicles[key] for key in sorted(self._vehicles)]
        next_traffic = None
        if self._traffic_position < len(self._traffic_rows):
            next_traffic = _json_value(self._traffic_rows[self._traffic_position]["event_time"])
        next_point = None
        if self._point_position < len(self._point_rows):
            next_point = _json_value(self._point_rows[self._point_position]["T"])
        active_routes = []
        active_positions = []
        for vehicle_id in sorted(self._vehicles):
            route = self._routes.get(vehicle_id)
            risk = str(self._vehicles[vehicle_id].get("risk", "on-time"))
            if route is not None:
                active_routes.append(
                    {
                        "tr_id": vehicle_id,
                        "risk": risk,
                        "color": RISK_COLORS.get(risk, "#9ca3af"),
                        "stops": route["stops"],
                        "polyline": route_polyline(route["stops"]),
                    }
                )
            position = _latest_position(self._vehicle_telemetry.get(vehicle_id, []))
            if position is not None:
                # Map-matched position: the fix projected onto the planned route.
                # ``raw`` keeps the uncorrected fix so the dispatcher can see the
                # real GPS scatter, ``matched`` is what belongs on the route.
                entry = {"tr_id": vehicle_id, "risk": risk, **position}
                entry["matched"] = self._match_position(vehicle_id, position)
                active_positions.append(entry)
        return {
            "simulated_time": _json_value(self.current_time),
            "speed": self.speed,
            "finished": self.finished,
            "processed_traffic_rows": self._traffic_position,
            "total_traffic_rows": len(self._traffic_rows),
            "processed_points": self._point_position,
            "total_points": len(self._point_rows),
            "next_traffic_time": next_traffic,
            "next_point_time": next_point,
            "vehicles": vehicles,
            "predictions": vehicles,
            "map": {
                "routes": active_routes,
                "positions": active_positions,
                "incident": self._select_incident(),
                "cascade": self.cascade_view(),
            },
        }
