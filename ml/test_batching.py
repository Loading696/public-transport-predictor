"""Equivalence and causality tests for the ML pipeline's batching.

Batching is only sound if two properties hold:

  1. EQUIVALENCE -- predicting points individually and predicting them in one
     batch must give identical numbers.  If not, the feature builder leaks
     information between points and the backend's micro-batching is unsound.

  2. CAUSALITY -- a point at time T must not see telemetry after T, even when
     its batch-mates sit at later times.  This is what makes it legal to merge
     the telemetry of concurrent requests by union.

Scope
-----
This file tests the *pipeline* -- feature building, the CatBoost forward pass, the
detectors -- in-process, through the same ``MlEngine`` the ML service serves.
Those two properties live in the feature builder, not in the transport.  The
boundary itself (HTTP, timeouts, breaker, degradation) is covered by
``ml/test_ml_integration.py``, which re-asserts equivalence *across the wire* so
the contract is verified end to end rather than only in the middle.

Run: py ml\\test_batching.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from ml.ml_pipeline import ML_SERVICE_DIR, load_engine, require_ready  # noqa: E402
from src import predictor as pred_mod  # noqa: E402

DS = ROOT / "dataset" / "validate"


def main() -> None:
    import service as ml_service  # provided by ml-service/ on sys.path

    engine = load_engine(ROOT)
    require_ready(engine)
    points = pred_mod.read_csv(DS / "points.csv")
    traffic = ml_service.normalise_telemetry(pred_mod.read_csv(DS / "traffic.csv"))

    def predict(frame: pd.DataFrame, tel=traffic) -> pd.DataFrame:
        """Run the pipeline and shape the result like a prediction frame."""
        records = engine.predict(frame.to_dict("records"), tel)
        out = frame.reset_index(drop=True).copy()
        out["prediction"] = [float(record["prediction"]) for record in records]
        out["risk"] = [record["risk"] for record in records]
        out["target_class"] = [record["target_class"] for record in records]
        return out

    print("1) batch == single-row (equivalence)")
    sample = points.head(12)
    batched = predict(sample)
    per_row = pd.concat(
        [predict(sample.head(i + 1).iloc[[i]]) for i in range(len(sample))],
        ignore_index=True,
    )
    for column in ("prediction", "target_class", "risk"):
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
        window = predict(point.to_frame().T)
        future = traffic[(traffic["tr_id"] == tid) & (times > t_value)]
        if future.empty:
            continue
        probe = predict(point.to_frame().T, pd.concat([traffic, future], ignore_index=True))
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
    alone = predict(probe_points, left)
    merged = predict(probe_points, pd.concat([left, right], ignore_index=True))
    diff = np.abs(
        alone["prediction"].to_numpy(dtype=float) - merged["prediction"].to_numpy(dtype=float)
    )
    assert diff.max() < 1e-9, diff.max()
    print(f"   early points: merged vs left-only max abs diff = {diff.max():.3e}")
    print(
        f"   union grew telemetry {len(left)} -> {len(left) + len(right)} rows, "
        "predictions unchanged"
    )

    print("\n4) per-row cost really is dominated by fixed overhead")
    timings: dict[int, float] = {}
    for n in (1, 8, 32):
        pts = points.head(n)
        started = time.perf_counter()
        for _ in range(3):
            predict(pts)
        timings[n] = (time.perf_counter() - started) / 3
        print(
            f"   n={n:<3} {timings[n] * 1000:8.1f} ms total, "
            f"{timings[n] / n * 1000:7.2f} ms/row"
        )
    assert timings[1] > timings[32] / 32, "batching should not be worse per row"
    print(f"   speedup per row 1 -> 32: {timings[1] / (timings[32] / 32):.1f}x")

    print("\n5) the service accepts telemetry as a frame and as records")
    # The live path and the micro-batcher both hand over DataFrames, while the
    # HTTP client sends records.  Both must reach the same numbers, or the two
    # call sites would disagree about what "the same request" means.
    frame = traffic.head(400).copy()
    point_rows = [
        {
            "tr_id": int(row["tr_id"]),
            "T": str(row["T"]),
            "target_stop_id": int(row["target_stop_id"]),
            "target_time_begin": str(row["target_time_begin"]),
            "cur_dev_s": float(row["cur_dev_s"]),
        }
        for _, row in points.head(3).iterrows()
    ]
    from_frame = engine.predict(point_rows, frame)
    from_records = engine.predict(point_rows, frame.to_dict("records"))
    assert len(from_frame) == 3, from_frame
    diff = max(
        abs(float(a["prediction"]) - float(b["prediction"]))
        for a, b in zip(from_frame, from_records)
    )
    assert diff < 1e-9, diff
    # An empty request is an error, not a silent empty answer.
    try:
        engine.predict([], frame)
    except ValueError:
        pass
    else:
        raise AssertionError("empty batch should be rejected")
    print(
        f"   frame == records (max abs diff {diff:.1e}), empty batch rejected"
    )

    print(f"\n   pipeline loaded from {ML_SERVICE_DIR.name}/service.py in-process")
    print("ALL BATCHING TESTS PASSED")


if __name__ == "__main__":
    main()
