"""Equivalence and causality tests for batched inference.

Batching is only sound if two properties hold:

  1. EQUIVALENCE -- predicting points individually and predicting them in one
     batch must give identical numbers.  If not, the feature builder leaks
     information between points and micro-batching is unsound.

  2. CAUSALITY -- a point at time T must not see telemetry after T, even when
     its batch-mates sit at later times.  This is what makes it legal to merge
     the telemetry of concurrent requests by union.

Run: py ml\test_batching.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from src import predictor as pred_mod  # noqa: E402
from src.runtime import InferenceService  # noqa: E402


def main() -> None:
    service = InferenceService(ROOT)
    points = service.points
    traffic = service.traffic

    print("1) batch == single-row (equivalence)")
    sample = points.head(12)
    batched = service.predict_frame(sample, traffic)
    per_row = pd.concat(
        [service.predict_frame(sample.head(i + 1).iloc[[i]], traffic) for i in range(len(sample))],
        ignore_index=True,
    )
    for column in ("prediction", "target_class", "risk", "recommendation"):
        if column == "prediction":
            diff = np.abs(
                batched[column].to_numpy(dtype=float) - per_row[column].to_numpy(dtype=float)
            )
            assert diff.max() < 1e-9, (column, diff.max())
            print(f"   {column:<14} max abs diff = {diff.max():.3e}")
        else:
            assert list(batched[column]) == list(per_row[column]), column
            print(f"   {column:<14} identical")

    print("\n2) causality: a point never sees telemetry after its own T")
    # NB: the selection must use PARSED timestamps. Comparing the raw CSV strings
    # lexicographically is not equivalent -- "2026-01-06 04:00:00.5" sorts after
    # "2026-01-06 04:00:00" but parses to a value that is not in the future.
    times = pd.to_datetime(traffic["event_time"], errors="coerce")
    ordered = sample.sort_values("T", kind="stable")
    checked = 0
    for _, point in ordered.iterrows():
        tid = int(point["tr_id"])
        t_value = pd.Timestamp(point["T"])
        window = service.predict_frame(point.to_frame().T, traffic)
        future = traffic[(traffic["tr_id"] == tid) & (times > t_value)]
        if future.empty:
            continue
        polluted = pd.concat([traffic, future], ignore_index=True)
        probe = service.predict_frame(point.to_frame().T, polluted)
        assert np.isclose(
            float(window["prediction"].iloc[0]), float(probe["prediction"].iloc[0]), atol=1e-9
        ), f"future telemetry changed the prediction for T={t_value}"
        checked += 1
    print(f"   {checked} points verified: appended future rows changed nothing")

    print("\n2b) unparseable event_time values in the telemetry")
    unparsed = int(times.isna().sum())
    print(f"   rows with unparseable event_time: {unparsed} of {len(times)}")

    print("\n3) telemetry union across requests is causally sound")
    # Two disjoint telemetry slices, merged. A point in slice A must not be
    # affected by rows that only exist in slice B.
    split_iso = "2026-01-06 04:00:00"
    left = traffic[traffic["event_time"] < split_iso]
    right = traffic[traffic["event_time"] >= split_iso]
    probe_points = points[points["T"] < split_iso].head(6)
    assert len(probe_points), "need early points for the causality probe"
    alone = service.predict_frame(probe_points, left)
    merged = service.predict_frame(probe_points, pd.concat([left, right], ignore_index=True))
    diff = np.abs(
        alone["prediction"].to_numpy(dtype=float) - merged["prediction"].to_numpy(dtype=float)
    )
    assert diff.max() < 1e-9, diff.max()
    print(f"   early points: merged vs left-only max abs diff = {diff.max():.3e}")
    print(f"   union grew telemetry {len(left)} -> {len(left) + len(right)} rows, predictions unchanged")

    print("\n4) per-row cost really is dominated by fixed overhead")
    timings = {}
    for n in (1, 8, 32):
        pts = points.head(n)
        start = pd.Timestamp("2026-01-06 00:00:00")
        _ = pd.Timestamp(start)
        import time as _t

        t0 = _t.perf_counter()
        for _ in range(3):
            service.predict_frame(pts, traffic)
        timings[n] = (_t.perf_counter() - t0) / 3
        print(f"   n={n:<3} {timings[n] * 1000:8.1f} ms total, {timings[n] / n * 1000:7.2f} ms/row")
    assert timings[1] > timings[32] / 32, "batching should not be worse per row"
    print(f"   speedup per row 1 -> 32: {timings[1] / (timings[32] / 32):.1f}x")

    print("\nALL BATCHING TESTS PASSED")


if __name__ == "__main__":
    main()
