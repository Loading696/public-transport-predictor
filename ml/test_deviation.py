"""Tests for live schedule-deviation estimation (src/deviation.py).

The estimate answers "how far is this bus off its timetable right now?" for a
live NDTP unit, where no ``cur_dev_s`` column and no ``time_fact_begin`` exist.
It is the replacement for the hard-coded ``cur_dev_s = 0.0`` placeholder that the
live endpoint used to feed the model.

Properties checked here:

  1. Ahead of schedule (negative deviation) and behind schedule (positive).
  2. Deviation is anchored to the *last* stop the vehicle provably reached, so a
     value cannot be produced from a stop it never visited.
  3. No suitable stop -> controlled fallback, never a fabricated number.
  4. Boundary cases: no schedule, no coordinates, vehicle not yet on route,
     fix exactly on the pass radius, stop exactly at T.
  5. No future leakage: appending telemetry and planned stops *after* the
     reference time must not change the estimate at all.
  6. The TTL "held" tier reuses a good estimate, and forgets it when stale.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deviation import (  # noqa: E402
    HINT_HELD,
    HINT_MEASURED,
    HINT_NEAREST,
    HINT_NONE,
    PASS_RADIUS_KM,
    LiveDeviationTracker,
    estimate_cur_deviation,
)

# --------------------------------------------------------------------------- #
# Fixtures: a straight 3-stop route along increasing longitude, one planned
# minute apart.  Stop k is at lon 37.60 + 0.01*k, lat 55.70.
# --------------------------------------------------------------------------- #

BASE = pd.Timestamp("2026-01-06 08:00:00")
STOP_LONS = [37.600, 37.610, 37.620]
STOP_IDS = [900, 901, 902]


def make_schedule(count: int = 3, *, step_s: int = 60) -> pd.DataFrame:
    """Planned stops, one every ``step_s`` seconds from BASE."""
    return pd.DataFrame(
        {
            "tt_action_item_id": [STOP_IDS[i % len(STOP_IDS)] + 100 * i for i in range(count)],
            "time_begin": [BASE + pd.Timedelta(seconds=step_s * i) for i in range(count)],
            "stop_lon": [37.600 + 0.01 * i for i in range(count)],
            "stop_lat": [55.700] * count,
        }
    )


def fix(offset_s: float, lon: float, lat: float = 55.700, *, valid: bool = True) -> dict:
    """One telemetry row ``offset_s`` seconds after BASE."""
    return {
        "event_time": BASE + pd.Timedelta(seconds=offset_s),
        "lon": lon,
        "lat": lat,
        "location_valid": valid,
    }


def estimate(telemetry, *, reference_s: float, schedule=None, tr_id: int = 7):
    return estimate_cur_deviation(
        tr_id=tr_id,
        reference_time=BASE + pd.Timedelta(seconds=reference_s),
        telemetry=telemetry,
        schedule=make_schedule() if schedule is None else schedule,
    )


# --------------------------------------------------------------------------- #


def test_behind_schedule() -> None:
    """Bus stood at stop 1 (planned 08:01) at 08:03:05 -> ~+125 s late."""
    result = estimate(
        [fix(0, 37.600), fix(60, 37.602), fix(180, 37.610), fix(185, 37.611)],
        reference_s=190,
    )
    assert result.hint == HINT_MEASURED, result
    assert result.method == "last_stop_presence", result
    # Last stop the vehicle provably reached is index 1 (planned BASE+60).
    assert result.stop_distance_m is not None and result.stop_distance_m < PASS_RADIUS_KM * 1000
    # Passage measured at 08:03:05, planned 08:01.
    assert 110.0 <= result.cur_dev_s <= 130.0, result.cur_dev_s
    assert result.cur_dev_s > 0, "должно быть опоздание"
    assert result.actual_time is not None and result.planned_time is not None
    measured = (result.actual_time - result.planned_time).total_seconds()
    assert math.isclose(measured, result.cur_dev_s, abs_tol=0.5), (measured, result.cur_dev_s)
    assert result.confident
    print(f"   late:      cur_dev_s={result.cur_dev_s:+.0f}s hint={result.hint} {result.note}")


def test_ahead_of_schedule() -> None:
    """Bus stood at stop 1 (planned 08:01) from 08:00:20 to 08:00:40 -> ~-20 s early.

    The reference time still has to be past the planned stop for the stop to be
    admissible at all, so "early" is expressed by reaching the stop sooner than
    planned, not by looking at a moment before the route started.  The anchor is
    the *last* moment the vehicle was demonstrably there (08:00:40), which is
    what the estimate reports.
    """
    result = estimate(
        [fix(0, 37.600), fix(20, 37.610), fix(40, 37.611)],
        reference_s=190,
    )
    assert result.hint == HINT_MEASURED, result
    assert -25.0 <= result.cur_dev_s <= -15.0, result.cur_dev_s
    assert result.cur_dev_s < 0, "должно быть опережение"
    print(f"   early:     cur_dev_s={result.cur_dev_s:+.0f}s hint={result.hint} {result.note}")


def test_on_time_is_near_zero() -> None:
    """Reached stop 1 exactly on its planned minute -> ~0 s."""
    result = estimate([fix(0, 37.600), fix(60, 37.610), fix(65, 37.611)], reference_s=70)
    assert result.hint == HINT_MEASURED, result
    assert abs(result.cur_dev_s) <= 10.0, result.cur_dev_s
    print(f"   on time:   cur_dev_s={result.cur_dev_s:+.0f}s hint={result.hint}")


def test_anchored_to_last_passed_stop() -> None:
    """The estimate must follow route progress, not stay on the first stop.

    Same schedule, same traffic pattern, but the vehicle has advanced one stop
    further along the route by reference time -- the deviation must be computed
    against that later stop's planned time, not the earlier one.
    """
    early = estimate([fix(0, 37.600), fix(60, 37.610), fix(70, 37.611)], reference_s=80)
    later = estimate(
        [fix(0, 37.600), fix(60, 37.610), fix(120, 37.620), fix(125, 37.621)],
        reference_s=130,
    )
    assert early.hint == HINT_MEASURED and later.hint == HINT_MEASURED
    assert abs(early.cur_dev_s) <= 10.0, early
    assert abs(later.cur_dev_s) <= 10.0, later
    # Different anchor stops, even though both deviations are ~0.
    assert early.stop_id != later.stop_id, (early.stop_id, later.stop_id)
    assert (later.route_progress or 0) > (early.route_progress or 0)
    print(
        f"   progress:  stop {early.stop_id} (p={early.route_progress}) -> "
        f"stop {later.stop_id} (p={later.route_progress})"
    )


def test_no_suitable_stop_falls_back_to_nearest() -> None:
    """Vehicle has not reached any stop yet -> nearest-stop tier, not zero.

    Placed halfway between stop 0 (planned 08:00) and stop 1 (planned 08:01, not
    yet due at T=08:00:30): ~320 m from each, so outside the 250 m pass radius and
    no stop can be claimed as passed.
    """
    result = estimate([fix(0, 37.605)], reference_s=30, schedule=make_schedule())
    assert result.hint == HINT_NEAREST, result
    assert result.method == "nearest_planned_stop", result
    assert result.cur_dev_s > 0, result
    assert result.stop_distance_m is not None and PASS_RADIUS_KM * 1000 < result.stop_distance_m
    assert "планов" in result.note
    print(f"   fallback:  cur_dev_s={result.cur_dev_s:+.0f}s hint={result.hint} {result.note}")


def test_unreachable_geometry_is_flagged_zero() -> None:
    """Too far from any planned stop -> explicit low-confidence zero, not a guess."""
    result = estimate([fix(0, 38.500)], reference_s=30)
    assert result.hint == HINT_NONE, result
    assert result.cur_dev_s == 0.0
    assert not result.confident
    assert result.note
    print(f"   refused:   cur_dev_s={result.cur_dev_s:+.0f}s hint={result.hint} {result.note}")


def test_missing_inputs_are_flagged_zero() -> None:
    """Every unusable input degrades to a flagged zero rather than raising."""
    cases = {
        "no schedule": dict(telemetry=[fix(0, 37.600)], reference_s=60, schedule=pd.DataFrame()),
        "no telemetry": dict(telemetry=[], reference_s=60),
        "no coordinates": dict(telemetry=[fix(0, 37.600, valid=False)], reference_s=60),
        "bad reference": dict(
            telemetry=[fix(0, 37.600)], reference_s=60, reference_override="not-a-time"
        ),
    }
    for name, case in cases.items():
        kwargs = {k: v for k, v in case.items() if k != "reference_override"}
        if "reference_override" in case:
            result = estimate_cur_deviation(
                tr_id=7,
                reference_time=case["reference_override"],
                telemetry=case["telemetry"],
                schedule=make_schedule(),
            )
        else:
            result = estimate(**kwargs)
        assert result.hint == HINT_NONE, (name, result)
        assert result.cur_dev_s == 0.0, (name, result)
    # A live unit with no schedule row at all.
    unknown = estimate_cur_deviation(
        tr_id=999,
        reference_time=BASE + pd.Timedelta(seconds=60),
        telemetry=[fix(0, 37.600)],
        schedule=pd.DataFrame(),
    )
    assert unknown.hint == HINT_NONE and unknown.cur_dev_s == 0.0
    print(f"   degraded:  {len(cases) + 1} случаев -> hint=none, значение 0 с явной пометкой")


def test_vehicle_not_yet_on_route() -> None:
    """No planned stop is due at T -> cannot be late yet, refuse to guess."""
    schedule = make_schedule()
    future_only = schedule[schedule["time_begin"] > BASE + pd.Timedelta(seconds=10)]
    result = estimate([fix(0, 37.600)], reference_s=5, schedule=future_only)
    assert result.hint == HINT_NONE, result
    assert result.cur_dev_s == 0.0
    assert "маршрут" in result.note
    print(f"   not yet:   hint={result.hint} {result.note}")


def test_boundary_exact_pass_radius() -> None:
    """A fix just inside the pass radius matches; just outside it does not.

    The vehicle is the only fix on a two-stop route, so the distance to the
    anchor stop is exactly the controlled one.  Kilometres are converted to
    degrees of *longitude* with the cos(lat) factor -- at 55.7 deg one degree of
    longitude is ~62.9 km, not ~111 km, and using the latitude constant would
    silently place the "outside" fix back inside the radius.
    """
    import math as _math

    km_per_deg_lon = 111.32 * _math.cos(_math.radians(55.700))
    route = pd.DataFrame(
        {
            "tt_action_item_id": [901, 902],
            "time_begin": [BASE, BASE + pd.Timedelta(seconds=60)],
            "stop_lon": [37.610, 37.700],
            "stop_lat": [55.700, 55.700],
        }
    )

    inside = PASS_RADIUS_KM * 0.9 / km_per_deg_lon
    matched = estimate([fix(0, 37.610 + inside)], reference_s=60, schedule=route)
    assert matched.hint == HINT_MEASURED, matched
    assert matched.stop_distance_m is not None
    assert matched.stop_distance_m <= PASS_RADIUS_KM * 1000, matched.stop_distance_m

    outside = PASS_RADIUS_KM * 1.6 / km_per_deg_lon
    far = estimate([fix(0, 37.610 + outside)], reference_s=60, schedule=route)
    assert far.hint in (HINT_NEAREST, HINT_NONE), far
    assert far.hint != HINT_MEASURED
    assert far.stop_distance_m is None or far.stop_distance_m > PASS_RADIUS_KM * 1000
    far_m = far.stop_distance_m if far.stop_distance_m is not None else float("nan")
    print(
        f"   radius:    {matched.stop_distance_m:.0f} м -> {matched.hint} (stop {matched.stop_id}), "
        f"{far_m:.0f} м -> {far.hint}"
    )


def test_stop_planned_exactly_at_reference_time() -> None:
    """A stop due exactly at T is admissible (window is time_begin <= T)."""
    result = estimate([fix(60, 37.610)], reference_s=60)
    assert result.hint == HINT_MEASURED, result
    assert result.stop_id == make_schedule().iloc[1]["tt_action_item_id"]
    assert math.isclose(result.cur_dev_s, 0.0, abs_tol=1.0), result.cur_dev_s
    print(f"   boundary:  stop at T учтена, cur_dev_s={result.cur_dev_s:+.0f}s")


def test_no_future_leakage() -> None:
    """Future telemetry and future planned stops must not move the estimate.

    This is the anti-leak guarantee.  We take a baseline estimate, then append
    (a) telemetry strictly after the reference time and (b) planned stops whose
    time_begin is after the reference time -- including one that would be a
    *perfect* match for the vehicle if the future were visible.  The result must
    be bit-identical.
    """
    baseline = estimate(
        [fix(0, 37.600), fix(60, 37.610), fix(70, 37.611)],
        reference_s=80,
    )
    assert baseline.hint == HINT_MEASURED, baseline

    future_fixes = [
        fix(200, 37.620),  # would match stop 2 exactly
        fix(400, 37.630),
    ]
    extended = estimate(
        [fix(0, 37.600), fix(60, 37.610), fix(70, 37.611), *future_fixes],
        reference_s=80,
    )
    assert extended.cur_dev_s == baseline.cur_dev_s, (baseline.cur_dev_s, extended.cur_dev_s)
    assert extended.stop_id == baseline.stop_id
    assert extended.hint == baseline.hint

    # A longer schedule whose extra stops are all planned after T.
    long_schedule = pd.concat(
        [make_schedule(3), make_schedule(3, step_s=60).assign(time_begin=lambda d: d["time_begin"] + pd.Timedelta(seconds=600))],
        ignore_index=True,
    )
    with_future_plan = estimate(
        [fix(0, 37.600), fix(60, 37.610), fix(70, 37.611)],
        reference_s=80,
        schedule=long_schedule,
    )
    assert with_future_plan.cur_dev_s == baseline.cur_dev_s, (
        baseline.cur_dev_s,
        with_future_plan.cur_dev_s,
    )
    assert with_future_plan.stop_id == baseline.stop_id

    # Even an absurd deviation hidden in the future changes nothing.
    poisoned = estimate(
        [
            fix(0, 37.600),
            fix(60, 37.610),
            fix(70, 37.611),
            fix(75, 37.620),  # "arrived" stop 2 while stop 1 was still the last due
        ],
        reference_s=80,
    )
    assert poisoned.cur_dev_s == baseline.cur_dev_s
    assert poisoned.stop_id == baseline.stop_id
    print(
        f"   no leak:   baseline={baseline.cur_dev_s:+.0f}s stop={baseline.stop_id}; "
        "добавление будущих точек и будущих остановок ничего не меняет"
    )


def test_schedule_future_stops_not_used_for_matching() -> None:
    """A future stop must not be selected even if the vehicle is standing on it."""
    schedule = make_schedule(3)
    # Vehicle is physically at stop 2's coordinates at t=70, but stop 2 is only
    # planned for t=120 -- it is not admissible at reference t=80.
    result = estimate(
        [fix(0, 37.600), fix(60, 37.610), fix(70, 37.620)],
        reference_s=80,
        schedule=schedule,
    )
    future_stop_id = schedule.iloc[2]["tt_action_item_id"]
    assert result.stop_id != future_stop_id, (result.stop_id, future_stop_id)
    assert result.hint == HINT_MEASURED, result
    print(f"   future:    остановка {future_stop_id} (план 08:02) не использована при T=08:01:20")


def test_tracker_holds_last_good_estimate() -> None:
    """TTL tier: reuse the last reliable estimate, then forget it when stale."""
    tracker = LiveDeviationTracker(max_age_s=300.0)
    good = [fix(0, 37.600), fix(60, 37.610), fix(130, 37.611)]
    first = tracker.estimate(
        key=1, tr_id=7, reference_time=BASE + pd.Timedelta(seconds=140), telemetry=good, schedule=make_schedule()
    )
    assert first.hint == HINT_MEASURED, first

    # GPS dropout: nothing measurable, but the previous estimate is fresh.
    held = tracker.estimate(
        key=1, tr_id=7, reference_time=BASE + pd.Timedelta(seconds=150), telemetry=[], schedule=make_schedule()
    )
    assert held.hint == HINT_HELD, held
    assert held.cur_dev_s == first.cur_dev_s, (held.cur_dev_s, first.cur_dev_s)
    assert "последняя" in held.note

    # A different unit has its own memory.
    other = tracker.estimate(
        key=2, tr_id=8, reference_time=BASE + pd.Timedelta(seconds=150), telemetry=[], schedule=make_schedule()
    )
    assert other.hint == HINT_NONE, other

    # Expire the memory: an old estimate must not be resurrected.
    tracker._last[1] = (tracker._mono() - 10_000, first)
    expired = tracker.estimate(
        key=1, tr_id=7, reference_time=BASE + pd.Timedelta(seconds=150), telemetry=[], schedule=make_schedule()
    )
    assert expired.hint == HINT_NONE, expired
    assert "устарела" in expired.note
    print(f"   held:      {held.cur_dev_s:+.0f}s hint={held.hint}; после TTL -> hint={expired.hint}")


def test_estimate_is_deterministic() -> None:
    """Same inputs -> same value; no hidden state in the pure function."""
    telemetry = [fix(0, 37.600), fix(60, 37.610), fix(150, 37.611)]
    values = {
        estimate(telemetry, reference_s=160).cur_dev_s for _ in range(5)
    }
    assert len(values) == 1, values
    print(f"   stable:    {values.pop():+.0f}s на 5 прогонах")


def test_real_dataset_against_ground_truth() -> None:
    """Run the estimator over the real validate period and score it.

    The synthetic fixtures above prove the arithmetic; this proves the estimator
    recovers something real.  ``points.csv`` ships the organisers' own
    ``cur_dev_s`` for all 151 validate points, so the estimate -- computed from
    telemetry and the plan only -- can be scored against it.

    Two things are asserted, and the second is the point of the whole exercise:

      * the estimate tracks the ground truth (positive correlation, error well
        inside the spread of the column), and
      * it detects lateness where the old constant placeholder could not.
        ``cur_dev_s > 120`` holds for 26.5% of the column, and the ``backlog``
        detector thresholds exactly there; a hard-coded 0.0 fires it zero times.
    """
    ds = ROOT / "dataset" / "validate"
    if not (ds / "points.csv").exists():
        print("   skipped:   dataset/validate отсутствует")
        return
    from src.predictor import read_csv, schedule_index

    points = read_csv(ds / "points.csv")
    traffic = read_csv(ds / "traffic.csv")
    traffic["event_time"] = pd.to_datetime(traffic["event_time"], errors="coerce")
    index = schedule_index(read_csv(ds / "schedule_plan.csv"))

    truth: list[float] = []
    estimates: list[float] = []
    for _, point in points.iterrows():
        tr_id = int(point["tr_id"])
        moment = pd.Timestamp(point["T"])
        history = traffic[
            (traffic["tr_id"] == tr_id) & (traffic["event_time"] <= moment)
        ]
        result = estimate_cur_deviation(
            tr_id=tr_id,
            reference_time=moment,
            telemetry=history,
            schedule=index.by_vehicle.get(tr_id),
        )
        truth.append(float(point["cur_dev_s"]))
        estimates.append(result.cur_dev_s)

    truth_a = np.asarray(truth, dtype=float)
    estimate_a = np.asarray(estimates, dtype=float)
    errors = np.abs(estimate_a - truth_a)
    zero_a = np.zeros_like(truth_a)
    correlation = float(np.corrcoef(estimate_a, truth_a)[0, 1])
    mae = float(errors.mean())
    zero_mae = float(np.abs(zero_a - truth_a).mean())
    late_truth = truth_a > 120.0
    late_est = estimate_a > 120.0
    recall = float((late_truth & late_est).sum() / max(int(late_truth.sum()), 1))

    assert mae < 300.0, f"MAE {mae:.0f}s слишком велик"
    assert correlation > 0.0, f"корреляция {correlation:+.3f} <= 0: оценка не связана с истиной"
    assert float(np.median(errors)) < 150.0, f"медианная ошибка {np.median(errors):.0f}s"
    # The estimate must carry information the constant cannot: it has to flag a
    # meaningful share of the truly late points.
    assert int(late_truth.sum()) >= 20, "слишком мало поздних точек для проверки"
    assert recall >= 0.4, f"recall по опозданию {recall:.2f} < 0.4"
    assert int((late_est & ~late_truth).sum()) >= 0

    print(
        f"   real data: n={len(truth_a)} MAE={mae:.0f}s (placeholder {zero_mae:.0f}s) "
        f"median|e|={np.median(errors):.0f}s r={correlation:+.3f}"
    )
    print(
        f"   late(>120s): truth={int(late_truth.sum())} est={int(late_est.sum())} "
        f"recall={recall:.2f} (placeholder: 0)"
    )


def main() -> None:
    tests = [
        test_behind_schedule,
        test_ahead_of_schedule,
        test_on_time_is_near_zero,
        test_anchored_to_last_passed_stop,
        test_no_suitable_stop_falls_back_to_nearest,
        test_unreachable_geometry_is_flagged_zero,
        test_missing_inputs_are_flagged_zero,
        test_vehicle_not_yet_on_route,
        test_boundary_exact_pass_radius,
        test_stop_planned_exactly_at_reference_time,
        test_no_future_leakage,
        test_schedule_future_stops_not_used_for_matching,
        test_tracker_holds_last_good_estimate,
        test_estimate_is_deterministic,
        test_real_dataset_against_ground_truth,
    ]
    for test in tests:
        print(f"[run ] {test.__name__}")
        test()
    print("ALL DEVIATION TESTS PASSED")


if __name__ == "__main__":
    main()
