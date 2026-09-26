"""Measure pattern detectors against real late labels (Person A).

Builds causal train/test frames, runs detect_all per row and reports
per-type precision/recall vs target_delay_s > 120s (test = honest).
Writes ml/pattern_metrics.json.

Slow one-off (~8 min frame build); run in background:
    py ml/eval_patterns.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.patterns import detect_all  # noqa: E402
from src.predictor import build_features, read_csv  # noqa: E402

LATE_THRESHOLD_S = 120.0
OUT_PATH = ROOT / "ml" / "pattern_metrics.json"
TYPES = ("dwell", "speed_drop", "backlog", "stale")


def log(message: str) -> None:
    print(message, flush=True)


def evaluate(frame, labels) -> dict[str, dict[str, float]]:
    is_late = (labels["target_delay_s"].to_numpy(dtype=float) > LATE_THRESHOLD_S)
    stats = {t: {"tp": 0, "fp": 0, "fn": 0} for t in TYPES}
    for pos, (_, row) in enumerate(frame.iterrows()):
        try:
            fired = {e["type"] for e in detect_all(row.to_dict())}
        except Exception:  # noqa: BLE001
            fired = set()
        late = bool(is_late[pos])
        for dtype in TYPES:
            if dtype in fired and late:
                stats[dtype]["tp"] += 1
            elif dtype in fired:
                stats[dtype]["fp"] += 1
            elif late:
                stats[dtype]["fn"] += 1
    report = {}
    for dtype in TYPES:
        tp, fp, fn = stats[dtype]["tp"], stats[dtype]["fp"], stats[dtype]["fn"]
        report[dtype] = {
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "support": int(tp + fp),
            "n_late": int(tp + fn),
        }
    return report


def main() -> None:
    t0 = time.time()
    dataset = ROOT / "dataset"
    log("[eval 1/3] building frames (slow)...")
    train_labels = read_csv(dataset / "labels" / "labels_train.csv")
    train_frame = build_features(
        train_labels, dataset / "train" / "traffic.csv", dataset / "train" / "schedule.csv"
    )
    test_labels = read_csv(dataset / "labels" / "labels_test.csv")
    test_frame = build_features(
        test_labels, dataset / "test" / "traffic.csv", dataset / "test" / "schedule.csv"
    )
    log("[eval 2/3] scoring detectors...")
    payload = {
        "late_threshold_s": LATE_THRESHOLD_S,
        "train": evaluate(train_frame, train_labels),
        "test": evaluate(test_frame, test_labels),
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[eval 3/3] wrote {OUT_PATH} in {time.time() - t0:.1f}s")
    for dtype in TYPES:
        m = payload["test"][dtype]
        log(f"  test/{dtype}: P={m['precision']} R={m['recall']} support={m['support']}")


if __name__ == "__main__":
    main()
