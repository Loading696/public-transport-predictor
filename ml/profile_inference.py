"""Profiler for the two-service split.

Reports where time actually goes now that inference lives in its own container:

  A. ML side  -- feature building vs the CatBoost forward pass (in-process,
     because profiling a network hop tells you nothing about the code);
  B. Backend  -- replay hot path, target lookup, WKT memoisation, cascade, and
     the request path as the backend sees it (with the ML call stubbed to the
     local pipeline, so the numbers are about backend work).

Run: py ml\\profile_inference.py [--rows N]
"""

from __future__ import annotations

import asyncio
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ml.ml_pipeline import load_engine, require_ready  # noqa: E402
from src import predictor as pred_mod  # noqa: E402
from src.ml_client import MlClient  # noqa: E402
from src.runtime import InferenceService, StreamSimulator  # noqa: E402


def timed(label: str, fn, repeat: int = 3) -> tuple[object, float]:
    started = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = fn()
    return result, (time.perf_counter() - started) / repeat


async def atimed(label: str, fn, repeat: int = 3) -> tuple[object, float]:
    """Timing helper for coroutine work, so no nested ``asyncio.run`` is needed."""
    started = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = await fn()
    return result, (time.perf_counter() - started) / repeat


class LocalMlClient(MlClient):
    """Runs the pipeline in-process so the profiler measures backend work only."""

    def __init__(self, engine) -> None:
        self._engine = engine

    async def predict_batch(self, points, telemetry=None):  # type: ignore[override]
        return self._engine.predict(points, telemetry)

    async def aclose(self) -> None:  # type: ignore[override]
        return None


async def main() -> None:
    rows = 20
    if "--rows" in sys.argv:
        rows = int(sys.argv[sys.argv.index("--rows") + 1])

    print("Loading ML pipeline (model + schedule index)...")
    t0 = time.perf_counter()
    engine = load_engine(ROOT)
    require_ready(engine)
    print(f"  pipeline load                                          {(time.perf_counter() - t0) * 1000:9.1f} ms")
    health = engine.health()
    print(
        f"  features={health['features']} schedule_stops={health['schedule_stops']} "
        f"vehicles={health['schedule_vehicles']} calibrated={health['calibrated']}"
    )

    schedule = engine._schedule
    index = engine._schedule_index
    points_all = pred_mod.read_csv(ROOT / "dataset" / "validate" / "points.csv")
    import service as ml_service

    traffic = ml_service.normalise_telemetry(
        pred_mod.read_csv(ROOT / "dataset" / "validate" / "traffic.csv")
    )

    print("\n== A. ML service ==")
    print("A1) cost of one batched inference vs batch size")
    single = 0.0
    for n in (1, 2, 5, 10, 25, 50):
        pts = points_all.head(n).to_dict("records")
        _, elapsed = timed(
            f"engine.predict(n={n})", lambda p=pts: engine.predict(p, traffic), 3
        )
        if n == 1:
            single = elapsed
        print(
            f"    n={n:<3} {elapsed * 1000:8.1f} ms total  {elapsed / n * 1000:7.2f} ms/row"
        )
        if n == 50:
            print(
                f"    per-row cost: n=1 {single * 1000:.1f} ms  vs  n=50 "
                f"{elapsed / 50 * 1000:.1f} ms"
            )

    print("\nA2) split inside one batch: features vs model.predict")
    pts_frame = points_all.head(50)
    _, f_time = timed(
        "build_features_from_frames(50 rows)",
        lambda: pred_mod.build_features_from_frames(pts_frame, traffic, schedule, index),
        3,
    )
    frame = pred_mod.build_features_from_frames(pts_frame, traffic, schedule, index)
    _, m_time = timed(
        "model.predict(50 rows)", lambda: engine._model.predict(frame[engine._features]), 20
    )
    print(f"    features share: {f_time / (f_time + m_time) * 100:.1f}%")

    print("\nA3) hot spots (cProfile over 5 x n=1 inference)")
    one = points_all.head(1).to_dict("records")
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(5):
        engine.predict(one, traffic)
    profiler.disable()
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(14)
    for line in buffer.getvalue().splitlines()[:24]:
        print("   ", line)

    print("\n== B. Backend ==")
    service = InferenceService(ROOT, ml_client=LocalMlClient(engine))
    print(
        f"  validate points={len(service.points)} traffic_rows={len(service.traffic)} "
        f"schedule_rows={len(service.schedule)}"
    )

    print("\nB1) StreamSimulator.advance_to cost (replay hot path)")
    sim = StreamSimulator(service, speed=600.0)
    await sim.advance_to(sim.current_time + pd.Timedelta(seconds=1))
    sim.status()
    _, adv = await atimed(
        "advance_to(+20 min)",
        lambda: sim.advance_to(sim.current_time + pd.Timedelta(seconds=1200)),
        2,
    )
    _, stat = timed("status()", sim.status, 3)
    _, casc = timed(
        "cascade_view() (dirty)",
        lambda: (setattr(sim, "_cascade_dirty", True), sim.cascade_view()),
        2,
    )
    print(f"    advance_to={adv * 1000:.1f} ms  status={stat * 1000:.1f} ms  cascade={casc * 1000:.1f} ms")
    print(f"    vehicles={len(sim._vehicles)} points_done={sim._point_position}")

    print("\nB2) target lookup per unit (what /stream/status pays)")
    idx = service._live_target_index
    tr_id = int(next(iter(idx)))
    times = idx[tr_id]["times_ns"]

    def live_target_once():
        from src.runtime import LIVE_TARGET_MIN_S, LIVE_TARGET_MAX_S

        import numpy as np

        reference = int(pd.Timestamp("2026-01-06 06:00:00").value)
        start = int(np.searchsorted(times, reference + LIVE_TARGET_MIN_S * 10**9, side="right"))
        stop = int(np.searchsorted(times, reference + LIVE_TARGET_MAX_S * 10**9, side="right"))
        return start, stop

    _, lookup = timed("live_target (binary search)", live_target_once, 50)
    print(f"    {lookup * 1e6:.1f} us per unit -> 30 units = {lookup * 30 * 1000:.2f} ms")

    print("\nB3) parse_point_wkt memoisation")
    geoms = list(service.schedule["geom"].head(3000))
    _, wkt_cached = timed(
        "parse_point_wkt x3000 (memoised)", lambda: [pred_mod.parse_point_wkt(g) for g in geoms], 5
    )
    print(f"    ~{wkt_cached * 1000:.2f} ms for 3000 lookups (LRU hit path)")

    print("\nB4) cascade engine")
    cascade = service.cascade_engine()
    _, build = timed("cascade_snapshot({}) (first solve)", lambda: service.cascade_snapshot({}), 1)
    print(f"    graph stats: {cascade.graph.stats}")
    print(f"    first solve: {build * 1000:.1f} ms (built once per process)")

    print("\nB5) predict_records with ETA projection (the /predict path)")
    batch = points_all.head(20).to_dict("records")
    _, rec = await atimed(
        "predict_records(20) + eta_projection",
        lambda: service.predict_records(batch, traffic),
        3,
    )
    print(f"    {rec * 1000:.1f} ms for 20 records ({rec / 20 * 1000:.2f} ms/record)")

    print("\nB6) ML boundary overhead (HTTP, loopback)")
    try:
        import os

        url = os.getenv("ML_SERVICE_URL")
        if url and os.getenv("ML_PROFILE_HTTP", "1") == "1":
            import httpx

            async with httpx.AsyncClient(base_url=url, timeout=30) as probe:
                _, warm = await atimed("GET /health (warm pool)", lambda: probe.get("/health"), 5)
                print(f"    {warm * 1000:.1f} ms per call, url={url}")
        else:
            print("    skipped (set ML_SERVICE_URL to measure a live service)")
    except Exception as exc:  # noqa: BLE001 - the profiler must always finish
        print(f"    skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
