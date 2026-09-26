"""Micro-batching behaviour and end-to-end latency budget.

Verifies the two properties the batcher must hold:
  * EQUIVALENCE -- a batched call returns exactly what individual calls return;
  * COALESCING  -- concurrent single-row calls collapse into one model call.

Then measures the latency of the real request path.

Run: py ml\test_microbatch.py
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from src.runtime import InferenceService  # noqa: E402
from src.batching import MicroBatcher  # noqa: E402


async def main() -> None:
    service = InferenceService(ROOT)
    points = service.points.head(24).to_dict("records")

    print("1) baseline: individual calls, no batcher")
    started = time.perf_counter()
    baseline = [service.predict_records([point]) for point in points[:8]]
    serial_ms = (time.perf_counter() - started) * 1000
    print(f"   8 sequential single-row calls: {serial_ms:.1f} ms ({serial_ms / 8:.1f} ms/call)")

    print("\n2) batcher: same 8 calls, sequential submission")
    batcher = MicroBatcher(service, max_batch_rows=256, window_ms=0.0)
    started = time.perf_counter()
    solo = [await batcher.predict_records([point]) for point in points[:8]]
    solo_ms = (time.perf_counter() - started) * 1000
    print(f"   8 sequential calls via batcher: {solo_ms:.1f} ms")

    print("\n3) batcher: 8 CONCURRENT single-row calls (the dashboard burst)")
    batcher = MicroBatcher(service, max_batch_rows=256, window_ms=6.0)
    await batcher.start()
    started = time.perf_counter()
    gathered = await asyncio.gather(*(batcher.predict_records([p]) for p in points[:8]))
    burst_ms = (time.perf_counter() - started) * 1000
    stats = batcher.stats.as_dict()
    print(f"   wall clock: {burst_ms:.1f} ms")
    print(f"   batches={stats['batches']} jobs={stats['jobs']} rows={stats['rows']} "
          f"jobs/batch={stats['jobs_per_batch']} rows/batch={stats['rows_per_batch']}")
    assert stats["batches"] >= 1, f"the batcher never recorded a batch: {stats}"
    assert stats["batches"] <= 2, f"expected the 8 calls to coalesce, got {stats}"
    assert stats["jobs"] == 8, stats
    assert stats["rows"] == 8, stats
    assert stats["jobs_per_batch"] >= 4, f"coalescing did not happen: {stats}"

    print("\n4) equivalence: batched results == individual results")
    flat = [record for group in gathered for record in group]
    ref = [record for group in solo for record in group]
    assert len(flat) == len(ref) == 8, (len(flat), len(ref))
    worst = max(abs(float(a["prediction"]) - float(b["prediction"])) for a, b in zip(flat, ref))
    assert worst == 0.0, f"micro-batching changed predictions by {worst}"
    for a, b in zip(flat, ref):
        assert a["risk"] == b["risk"] and a["target_class"] == b["target_class"]
        assert a["sample_id"] == b["sample_id"], "records were handed to the wrong caller"
    print(f"   max abs prediction diff = {worst}; sample_ids routed correctly")

    print("\n5) per-request latency distribution (concurrent single-row, 40 calls)")
    latencies: list[float] = []

    async def one(point):
        t0 = time.perf_counter()
        await batcher.predict_records([point])
        latencies.append((time.perf_counter() - t0) * 1000)

    many = (points * 3)[:40]
    t0 = time.perf_counter()
    await asyncio.gather(*(one(p) for p in many))
    total_ms = (time.perf_counter() - t0) * 1000
    latencies.sort()
    print(
        f"   40 calls in {total_ms:.0f} ms | p50={statistics.median(latencies):.0f} ms "
        f"p95={latencies[int(len(latencies) * 0.95) - 1]:.0f} ms max={latencies[-1]:.0f} ms"
    )
    print(f"   throughput: {len(many) / (total_ms / 1000):.1f} req/s")
    print(f"   final batching stats: {batcher.stats.as_dict()}")

    print("\n6) batcher shutdown is clean")
    await batcher.stop()
    assert batcher.stats.closed is True
    print("   stopped, pending futures rejected")

    print(f"\nSUMMARY  serial 8 calls {serial_ms:.0f} ms -> concurrent 8 calls {burst_ms:.0f} ms")
    print("ALL MICROBATCH TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
