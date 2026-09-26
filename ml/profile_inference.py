"""Profiling harness for the inference path.

Answers three questions that decide where optimisation effort should go:
  1. how much of a single prediction is feature building vs model.predict;
  2. how does cost scale with batch size (i.e. is per-row prediction wasteful);
  3. which functions dominate the hot path.

Run: py ml\profile_inference.py [--rows N]
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.runtime import InferenceService, StreamSimulator  # noqa: E402
from src import predictor as pred_mod  # noqa: E402


def timed(label, fn, repeat=1):
    start = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    elapsed = (time.perf_counter() - start) / repeat
    print(f"  {label:<46} {elapsed * 1000:9.1f} ms")
    return out, elapsed


def main() -> None:
    rows = 20
    if "--rows" in sys.argv:
        rows = int(sys.argv[sys.argv.index("--rows") + 1])

    print("Loading InferenceService (CSV + model)...")
    t0 = time.perf_counter()
    service = InferenceService(ROOT)
    print(f"  startup                                               {(time.perf_counter() - t0) * 1000:9.1f} ms")
    print(
        f"  validate points={len(service.points)} traffic_rows={len(service.traffic)} "
        f"schedule_rows={len(service.schedule)} features={len(service.features)}"
    )

    points_all = service.points
    traffic = service.traffic

    print("\n1) Cost of predict_frame vs batch size (single-row vs batch)")
    for n in (1, 2, 5, 10, 25, 50):
        pts = points_all.head(n).to_dict("records")
        _, elapsed = timed(f"predict_frame(n={n})", lambda p=pts: service.predict_frame(p, traffic), 3)
        if n == 1:
            single = elapsed
        if n == 50:
            print(f"    per-row cost: n=1 {single * 1000:.1f} ms  vs  n=50 {elapsed / 50 * 1000:.1f} ms")

    print("\n2) Split inside one batch: features vs model.predict")
    pts_frame = points_all.head(50)
    _, f_time = timed(
        "build_features_from_frames(50 rows)",
        lambda: pred_mod.build_features_from_frames(pts_frame, traffic, service.schedule),
        3,
    )
    frame = pred_mod.build_features_from_frames(pts_frame, traffic, service.schedule)
    _, m_time = timed("model.predict(50 rows)", lambda: service.model.predict(frame[service.features]), 20)
    print(f"    features share: {f_time / (f_time + m_time) * 100:.1f}%")

    print("\n3) Hot spots (cProfile over 5 x n=1 predict_frame)")
    one = points_all.head(1).to_dict("records")
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(5):
        service.predict_frame(one, traffic)
    profiler.disable()
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(14)
    for line in buffer.getvalue().splitlines()[:24]:
        print("   ", line)

    print("\n4) StreamSimulator.advance_to cost (replay hot path)")
    sim = StreamSimulator(service, speed=600.0)
    sim.advance_to(sim.current_time + pd.Timedelta(seconds=1))
    sim.status()
    _, adv = timed(
        "advance_to(+20 min, ~N points)",
        lambda: sim.advance_to(sim.current_time + pd.Timedelta(seconds=1200)),
        2,
    )
    _, stat = timed("status()", sim.status, 3)
    _, casc = timed("cascade_view() (dirty)", lambda: (setattr(sim, "_cascade_dirty", True), sim.cascade_view()), 2)
    print(f"    vehicles={len(sim._vehicles)} points_done={sim._point_position}")

    print("\n5) _live_target: pandas scan of the schedule per unit (per request)")
    schedule = service.schedule
    tr_id = int(schedule["tr_id"].iloc[0])
    frame = schedule.copy()
    frame["time_begin"] = pd.to_datetime(frame["time_begin"], errors="coerce")

    def live_target_once():
        sub = frame[pd.to_numeric(frame["tr_id"], errors="coerce") == tr_id].copy()
        sub = sub.dropna(subset=["time_begin"]).sort_values("time_begin", kind="stable")
        return sub

    _, scan = timed("filter+sort schedule for one tr_id", live_target_once, 20)
    print(f"    ~{scan * 1000:.2f} ms per unit per request -> 30 units = {scan * 30 * 1000:.0f} ms")

    print("\n6) parse_point_wkt memoisation potential")
    geoms = list(service.schedule["geom"].head(3000))
    _, wkt = timed("parse_point_wkt x3000 (uncached)", lambda: [pred_mod.parse_point_wkt(g) for g in geoms], 3)
    print(f"    ~{wkt * 1000:.1f} ms per call site; _route_network and add_schedule_features both parse the full schedule")

    print("\n7) Cascade engine")
    engine = service.cascade_engine()
    _, build = timed("build_cascade_graph (once per process)", lambda: service.cascade_snapshot({}), 1)
    print(f"    graph stats: {engine.graph.stats}")

    print("\n8) predict_records with projection (the /predict path)")
    batch = points_all.head(20).to_dict("records")
    _, rec = timed("predict_records(20) + eta_projection", lambda: service.predict_records(batch), 3)
    print(f"    per record: {rec / 20 * 1000:.1f} ms")


if __name__ == "__main__":
    main()
