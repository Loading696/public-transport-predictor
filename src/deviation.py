"""Current schedule deviation for live NDTP units.

Why this module exists
----------------------
The model's strongest single feature is ``cur_dev_s`` -- the deviation of the
vehicle from its timetable at the forecast moment ``T``.  In the replay/validate
contour it arrives ready-made in ``points.csv``.  For a live NDTP unit there is no
such column, and the endpoint used to feed the model a hard-coded ``0.0``, which
silently told the model "this bus is exactly on time" for every live unit.  In the
training data ``cur_dev_s > 120`` holds for 26.5% of points, so the placeholder
was not a neutral prior -- it was a systematic bias towards "on schedule" and it
suppressed the ``backlog`` detector (``cur_dev_s >= 120``) for live units.

How the estimate is obtained
----------------------------
The live feed gives GPS fixes with ``event_time``; the plan gives a stop
sequence with ``time_begin`` and geometry.  Neither carries an actual arrival
time, and ``time_fact_begin`` must never be read.  So the actual passage time of
a stop is *measured* from telemetry: the fix closest to that stop's coordinates
is the moment the vehicle was physically there.

The estimate is therefore

    cur_dev_s = actual passage time of the last demonstrably passed stop
                - its planned ``time_begin``

with a documented fallback when no stop can be matched.

Anti-leak guarantees
--------------------
Two independent cuts, both enforced inside :func:`estimate_cur_deviation` rather
than trusted from the caller:

* telemetry is filtered to ``event_time <= reference_time``;
* the plan is filtered to ``time_begin <= reference_time`` -- a vehicle cannot
  have passed a stop that is scheduled after the forecast moment.

Nothing else is read, and ``time_fact_begin`` is never referenced.  Both cuts are
asserted by ``ml/test_deviation.py``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.schedule_index import EARTH_RADIUS_KM

#: A fix this close to a planned stop counts as "the vehicle was at this stop".
#: Urban stop-to-stop GPS scatter is tens of metres; 250 m is generous but still
#: far tighter than the spacing between consecutive planned stops.
PASS_RADIUS_KM = 0.25

#: How far back telemetry is searched for a stop match.  The most recent passed
#: stop of a running vehicle is never older than this; older fixes only add cost.
LOOKBACK_S = 5400.0

#: When nothing was demonstrably passed (the vehicle has not reached any planned
#: stop yet, or the GPS is too coarse), fall back to the nearest planned stop and
#: treat *now* as the estimated passage time.  Beyond this distance even that
#: fallback is meaningless, so the estimate is refused.
NEAREST_STOP_KM = 1.5

#: Largest deviation that is still attributed to a stop rather than treated as a
#: failed identification.  A vehicle's plan and its telemetry can be minutes out
#: of step, and a terminal's coordinates are shared by every run of the day; when
#: the only geometrically matching plan instance sits hours away in time, that is
#: not a two-hour delay, it is the wrong instance.  The model is trained on
#: ``cur_dev_s`` in roughly [-320, +440] s, so 30 minutes is already generous --
#: this bound only decides *whether to report a measurement*, never the value.
MAX_PLAUSIBLE_DEV_S = 1800.0

#: A "held" (reused) estimate is not trusted for longer than this.
HELD_MAX_AGE_S = 300.0

#: Confidence tiers, ordered from best to worst.  ``none`` is the only value the
#: pre-existing API already used, and it keeps its meaning: no usable hint.
HINT_MEASURED = "measured"
HINT_NEAREST = "nearest_stop"
HINT_HELD = "held"
HINT_NONE = "none"


def _haversine_grid_km(
    fix_lons: np.ndarray,
    fix_lats: np.ndarray,
    stop_lons: np.ndarray,
    stop_lats: np.ndarray,
) -> np.ndarray:
    """Great-circle distance in km between every fix and every stop.

    Shape ``(len(fix_lons), len(stop_lons))``.  Written out here rather than
    reusing :func:`src.predictor.haversine_km` because that helper broadcasts a
    *scalar* second point against the first and so cannot express a full
    fixes x stops matrix; this module needs the full matrix to pick the nearest
    stop per fix in one vectorised pass.
    """
    phi1 = np.deg2rad(fix_lats)[:, None]
    phi2 = np.deg2rad(stop_lats)[None, :]
    dphi = phi2 - phi1
    dlam = np.deg2rad(stop_lons)[None, :] - np.deg2rad(fix_lons)[:, None]
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _none(reason: str, *, tr_id: int | None = None) -> "DeviationEstimate":
    """Build the explicit low-confidence answer: zero, and marked as such."""
    return DeviationEstimate(
        cur_dev_s=0.0,
        hint=HINT_NONE,
        method="unavailable",
        stop_id=None,
        planned_time=None,
        actual_time=None,
        stop_distance_m=None,
        route_progress=None,
        tr_id=tr_id,
        note=reason,
    )


@dataclass(frozen=True, slots=True)
class DeviationEstimate:
    """Result of one deviation estimate, with everything needed to audit it.

    Attributes
    ----------
    cur_dev_s:
        Deviation in seconds, positive when the vehicle is late.  Feed this to
        the model in place of the old ``0.0`` placeholder.
    hint:
        ``measured`` (a stop was matched against telemetry), ``nearest_stop``
        (estimated from the closest planned stop), ``held`` (reused a previous
        good estimate) or ``none`` (no usable hint, value is a flagged zero).
    method:
        Machine-readable reason behind ``hint``, e.g. ``last_passed_stop``.
    stop_id, planned_time, actual_time:
        The stop the estimate is anchored to and the two times that produced it.
    stop_distance_m:
        Distance from the matched fix to the stop; proves the match was spatial.
    route_progress:
        Fraction of the planned stop sequence covered at ``planned_time``.
    tr_id:
        Vehicle the estimate belongs to, echoed for traceability.
    note:
        Russian, human-readable; rendered next to the forecast in the UI.
    """

    cur_dev_s: float
    hint: str
    method: str
    stop_id: int | None
    planned_time: pd.Timestamp | None
    actual_time: pd.Timestamp | None
    stop_distance_m: float | None
    route_progress: float | None
    tr_id: int | None
    note: str

    @property
    def confident(self) -> bool:
        """True when the value is a real measurement rather than a fallback."""
        return self.hint in (HINT_MEASURED, HINT_NEAREST, HINT_HELD)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready view for the API (``Timestamp`` values become ISO strings)."""
        return {
            "cur_dev_s": self.cur_dev_s,
            "hint": self.hint,
            "method": self.method,
            "confident": self.confident,
            "stop_id": self.stop_id,
            "planned_time": None if self.planned_time is None else self.planned_time.isoformat(),
            "actual_time": None if self.actual_time is None else self.actual_time.isoformat(),
            "stop_distance_m": self.stop_distance_m,
            "route_progress": self.route_progress,
            "tr_id": self.tr_id,
            "note": self.note,
        }


def _as_naive_timestamp(value: Any) -> pd.Timestamp | None:
    """Coerce to a tz-naive ``Timestamp``; ``None`` when unusable."""
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp


def _telemetry_frame(telemetry: Any) -> pd.DataFrame:
    """Coerce arbitrary telemetry rows into the canonical columns."""
    if telemetry is None:
        return pd.DataFrame(columns=["event_time", "lon", "lat", "location_valid"])
    if isinstance(telemetry, pd.DataFrame):
        frame = telemetry
    else:
        frame = pd.DataFrame(list(telemetry))
    if frame.empty:
        return pd.DataFrame(columns=["event_time", "lon", "lat", "location_valid"])
    for column in ("event_time", "lon", "lat", "location_valid"):
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[["event_time", "lon", "lat", "location_valid"]].copy()
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="coerce")
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    return frame.dropna(subset=["event_time"])


def estimate_cur_deviation(
    *,
    tr_id: int | None,
    reference_time: Any,
    telemetry: Any,
    schedule: pd.DataFrame | None,
) -> DeviationEstimate:
    """Estimate the live deviation of one vehicle from its planned timetable.

    Parameters
    ----------
    tr_id:
        Vehicle identifier, echoed back on the result.
    reference_time:
        The forecast moment ``T`` (the live fix's ``event_time``).  Both the
        telemetry cut and the plan cut are taken relative to it.
    telemetry:
        Rows with ``event_time``/``lon``/``lat``/``location_valid`` for this
        vehicle.  A DataFrame or an iterable of mappings.
    schedule:
        The vehicle's planned stops -- ``time_begin``, ``tt_action_item_id`` and
        optionally ``stop_lon``/``stop_lat`` (as produced by
        :class:`src.predictor.ScheduleIndex`).

    Returns
    -------
    DeviationEstimate
        Never raises: an unusable input yields ``cur_dev_s = 0.0`` with
        ``hint="none"`` and a note explaining why.
    """
    reference = _as_naive_timestamp(reference_time)
    if reference is None:
        return _none("нет времени события: отклонение не вычислено")

    # ---- plan side: only stops already planned at T are admissible ---------- #
    if schedule is None or len(schedule) == 0:
        return _none("нет расписания для ТС: отклонение не вычислено", tr_id=tr_id)
    plan = schedule.copy()
    if "time_begin" not in plan.columns:
        return _none("в расписании нет time_begin: отклонение не вычислено", tr_id=tr_id)
    plan["time_begin"] = pd.to_datetime(plan["time_begin"], errors="coerce")
    plan = plan.dropna(subset=["time_begin"])
    total_stops = len(plan)
    due = plan[plan["time_begin"] <= reference]
    if len(due) == 0:
        return _none(
            "по расписанию ТС ещё не вышло на маршрут: отклонение не вычислено",
            tr_id=tr_id,
        )
    due = due.sort_values("time_begin", kind="stable").reset_index(drop=True)

    # ---- telemetry side: causal cut, then keep only usable fixes ------------ #
    fixes = _telemetry_frame(telemetry)
    if len(fixes) == 0:
        return _none("нет телеметрии: отклонение не вычислено", tr_id=tr_id)
    fixes = fixes[fixes["event_time"] <= reference]
    if "location_valid" in fixes.columns:
        valid_flag = fixes["location_valid"]
        if valid_flag.dtype == object:
            valid_flag = valid_flag.astype(str).str.lower().isin({"true", "1", "yes"})
        else:
            valid_flag = valid_flag.fillna(False).astype(bool)
        fixes = fixes[valid_flag]
    fixes = fixes[np.isfinite(fixes["lon"]) & np.isfinite(fixes["lat"])]
    if len(fixes) == 0:
        return _none("нет достоверных координат: отклонение не вычислено", tr_id=tr_id)

    fixes = fixes.sort_values("event_time", kind="stable").reset_index(drop=True)
    floor = reference - pd.Timedelta(seconds=LOOKBACK_S)
    windowed = fixes[fixes["event_time"] >= floor]
    if len(windowed) == 0:
        windowed = fixes.tail(1)
    fix_times = windowed["event_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    fix_lons = windowed["lon"].to_numpy(dtype=float)
    fix_lats = windowed["lat"].to_numpy(dtype=float)

    stop_times = due["time_begin"].to_numpy(dtype="datetime64[ns]").astype("int64")
    stop_ids = (
        due["tt_action_item_id"].to_numpy(dtype="int64")
        if "tt_action_item_id" in due.columns
        else np.full(len(due), -1, dtype="int64")
    )
    if "stop_lon" in due.columns and "stop_lat" in due.columns:
        stop_lons = due["stop_lon"].to_numpy(dtype=float)
        stop_lats = due["stop_lat"].to_numpy(dtype=float)
    elif "lon" in due.columns and "lat" in due.columns:
        stop_lons = due["lon"].to_numpy(dtype=float)
        stop_lats = due["lat"].to_numpy(dtype=float)
    else:
        return _none("в расписании нет координат остановок: отклонение не вычислено", tr_id=tr_id)

    usable = np.isfinite(stop_lons) & np.isfinite(stop_lats)
    if not usable.any():
        return _none("нет координат остановок: отклонение не вычислено", tr_id=tr_id)
    stop_lons = np.where(usable, stop_lons, np.nan)
    stop_lats = np.where(usable, stop_lats, np.nan)

    # (fixes x stops) great-circle distance, km.
    grid = _haversine_grid_km(fix_lons, fix_lats, stop_lons, stop_lats)
    masked = np.where(usable[None, :], grid, np.inf)

    def _finish(
        stop_pos: int,
        *,
        hint: str,
        method: str,
        actual_ns: int,
        distance_km: float,
        anchor_dev: float,
    ) -> DeviationEstimate:
        # The reported value is the deviation *at the anchor stop*, deliberately
        # not carried forward to "now".  Propagation was implemented and
        # measured on the 151 validate points against the ground-truth column and
        # rejected: it centres the distribution better (bias +81 s -> -20 s, MAE
        # 137 -> 126) but collapses the signal the feature exists for -- the
        # `backlog` detector fires at cur_dev_s >= 120 and its recall on truly
        # late points fell from 0.68 to 0.12, because a bus that was late at its
        # last stop and has since driven normally averages back towards zero.
        # Deviation at the last confirmed stop is what the threshold means.
        deviation = anchor_dev
        planned = pd.Timestamp(stop_times[stop_pos])
        actual = pd.Timestamp(actual_ns)
        if hint == HINT_MEASURED:
            note = (
                f"отклонение {deviation:+.0f} с: остановка "
                f"{planned.strftime('%H:%M:%S')}, фактическое прохождение "
                f"{actual.strftime('%H:%M:%S')}, расстояние до остановки "
                f"{distance_km * 1000.0:.0f} м"
            )
        elif hint == HINT_NEAREST:
            note = (
                f"отклонение {deviation:+.0f} с (оценка): ТС ещё не проходила плановые "
                f"остановки, взята ближайшая {planned.strftime('%H:%M:%S')} "
                f"в {distance_km * 1000.0:.0f} м"
            )
        else:
            note = f"отклонение {deviation:+.0f} с ({method})"
        return DeviationEstimate(
            cur_dev_s=deviation,
            hint=hint,
            method=method,
            stop_id=int(stop_ids[stop_pos]) if stop_ids[stop_pos] >= 0 else None,
            planned_time=planned,
            actual_time=actual,
            stop_distance_m=round(distance_km * 1000.0, 1),
            route_progress=(
                round((stop_pos + 1) / total_stops, 4) if total_stops else None
            ),
            tr_id=tr_id,
            note=note,
        )

    # ---- primary: the most recent fix that stood at a planned stop --------- #
    # The anchor is a *fix*, not a stop.  A single tr_id's plan spans the whole
    # service day while reusing the same coordinates many times (in the validate
    # set: 408 planned stops over 33 distinct positions, one terminus planned 24
    # times).  Picking "the last stop that any fix happens to be near" therefore
    # pairs a location with a fix from a different run hours earlier and reports
    # a bogus half-hour deviation.  Anchoring on an observed moment in time and
    # resolving which planned instance it was avoids that entirely.
    at_stop = np.flatnonzero(masked.min(axis=1) <= PASS_RADIUS_KM)
    if at_stop.size:
        fix_pos = int(at_stop[-1])
        row = masked[fix_pos]
        near = np.flatnonzero(row <= PASS_RADIUS_KM)
        if near.size:
            # Several plan instances can share one coordinate (a terminus served
            # every N minutes).  The instance is identified by which one is
            # temporally closest to the fix -- that is an identification choice
            # between geometrically identical candidates, not a bias on the
            # deviation: the passage time stays exactly the fix's own time.
            planned_ns = stop_times[near]
            stop_pos = int(near[int(np.argmin(np.abs(planned_ns - fix_times[fix_pos])))])
            anchor = (int(fix_times[fix_pos]) - int(stop_times[stop_pos])) / 1e9
            if abs(anchor) <= MAX_PLAUSIBLE_DEV_S:
                return _finish(
                    stop_pos,
                    hint=HINT_MEASURED,
                    method="last_stop_presence",
                    actual_ns=int(fix_times[fix_pos]),
                    distance_km=float(row[stop_pos]),
                    anchor_dev=anchor,
                )

    # ---- fallback: the planned instance the vehicle is sitting on ----------- #
    # Nothing credible to measure, so bound the answer from the plan instead: the
    # bus's latest position is nearest to some stop P, and the plan says it should
    # have been at P at ``tau``.  Still being at P at T means at least ``T - tau``
    # late -- a lower bound on current lateness, from real position and real plan.
    last_fix = int(len(fix_lons) - 1)
    row = masked[last_fix]
    near = np.flatnonzero(row <= NEAREST_STOP_KM)
    if near.size:
        stop_pos = int(near[int(np.argmin(np.abs(stop_times[near] - reference.value)))])
        bound_s = (reference.value - int(stop_times[stop_pos])) / 1e9
        if 0.0 <= bound_s <= MAX_PLAUSIBLE_DEV_S:
            return _finish(
                stop_pos,
                hint=HINT_NEAREST,
                method="nearest_planned_stop",
                actual_ns=reference.value,
                distance_km=float(row[stop_pos]),
                anchor_dev=bound_s,
            )
        return _none(
            f"ТС вне планового коридора (ближайшая остановка планируется "
            f"на {pd.Timestamp(stop_times[stop_pos]).strftime('%H:%M:%S')}): "
            f"отклонение не вычислено",
            tr_id=tr_id,
        )
    return _none(
        "ТС ещё не проходила плановые остановки: отклонение не вычислено",
        tr_id=tr_id,
    )


class LiveDeviationTracker:
    """Per-unit memory of the last good estimate, with a TTL.

    :func:`estimate_cur_deviation` is pure: when it cannot measure anything it
    says so (``hint="none"``) instead of inventing a number.  This class adds
    the last fallback tier on top of it -- reuse the most recent *reliable*
    estimate for the same unit while it is still fresh -- so a brief GPS outage
    degrades the hint instead of silently snapping the model input back to zero.

    Memory is bounded by the caller: one entry per unit, overwritten in place.
    """

    def __init__(self, *, max_age_s: float = HELD_MAX_AGE_S) -> None:
        self.max_age_s = float(max_age_s)
        self._last: dict[Any, tuple[float, DeviationEstimate]] = {}
        self._mono = time.monotonic

    def _remember(self, key: Any, estimate: DeviationEstimate) -> None:
        if estimate.hint in (HINT_MEASURED, HINT_NEAREST):
            self._last[key] = (self._mono(), estimate)

    def _held(self, key: Any, reason: str) -> DeviationEstimate:
        entry = self._last.get(key)
        if entry is None:
            return _none(reason)
        stamp, estimate = entry
        if self._mono() - stamp > self.max_age_s:
            return _none(f"{reason} (прежняя оценка устарела)")
        return DeviationEstimate(
            cur_dev_s=estimate.cur_dev_s,
            hint=HINT_HELD,
            method=f"reused:{estimate.method}",
            stop_id=estimate.stop_id,
            planned_time=estimate.planned_time,
            actual_time=estimate.actual_time,
            stop_distance_m=estimate.stop_distance_m,
            route_progress=estimate.route_progress,
            tr_id=estimate.tr_id,
            note=f"{reason}; показана последняя достоверная оценка ({estimate.cur_dev_s:+.0f} с)",
        )

    def estimate(
        self,
        *,
        key: Any,
        tr_id: int | None,
        reference_time: Any,
        telemetry: Any,
        schedule: pd.DataFrame | None,
    ) -> DeviationEstimate:
        """Measure the deviation, falling back to the last reliable estimate."""
        result = estimate_cur_deviation(
            tr_id=tr_id,
            reference_time=reference_time,
            telemetry=telemetry,
            schedule=schedule,
        )
        if result.hint in (HINT_MEASURED, HINT_NEAREST):
            self._remember(key, result)
            return result
        return self._held(key, result.note)

    def forget(self, key: Any) -> None:
        """Drop the remembered estimate for a unit (used when it disconnects)."""
        self._last.pop(key, None)

    def known_units(self) -> Iterable[Any]:
        return tuple(self._last)
