"""Calibrate P(delay > 120s) on top of the CatBoost delay regressor (Person A).

Pipeline:
  1. build causal frames (train/test) via src/predictor.build_features (~8 min);
  2. train regressor, predict test (honest holdout);
  3. fit IsotonicRegression: prediction -> P(late);
  4. save ml/prob_cal.json {"kind", "X", "p", "brier", "n", "late_rate"}.

The JSON is the contract Person B's UI will read; X/p define a stepwise
calibration curve (prediction thresholds -> probability).

Usage:
    py ml/calibrate.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predictor import build_features, predict, read_csv, train_model  # noqa: E402

LATE_THRESHOLD_S = 120.0
OUT_PATH = ROOT / "ml" / "prob_cal.json"


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    t0 = time.time()
    dataset = ROOT / "dataset"
    log("[cal 1/4] building train/test frames (slow, one-off)...")
    train_labels = read_csv(dataset / "labels" / "labels_train.csv")
    train_frame = build_features(
        train_labels, dataset / "train" / "traffic.csv", dataset / "train" / "schedule.csv"
    )
    test_labels = read_csv(dataset / "labels" / "labels_test.csv")
    test_frame = build_features(
        test_labels, dataset / "test" / "traffic.csv", dataset / "test" / "schedule.csv"
    )
    log("[cal 2/4] training regressor...")
    model, features = train_model(train_frame)
    log("[cal 3/4] fitting isotonic P(late | prediction)...")
    test_pred = np.asarray(predict(model, test_frame, features), dtype=float)
    is_late = (test_labels["target_delay_s"].to_numpy(dtype=float) > LATE_THRESHOLD_S).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip")
    proba = np.asarray(iso.fit_transform(test_pred, is_late), dtype=float)
    brier = float(brier_score_loss(is_late, proba))
    thresholds = np.asarray(iso.X_thresholds_, dtype=float)
    curve = np.asarray(iso.predict(thresholds), dtype=float)
    payload = {
        "kind": "isotonic",
        "late_threshold_s": LATE_THRESHOLD_S,
        "X": [float(v) for v in thresholds],
        "p": [float(v) for v in curve],
        "brier": brier,
        "n": int(len(is_late)),
        "late_rate": float(is_late.mean()),
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[cal 4/4] wrote {OUT_PATH} (n={len(is_late)} brier={brier:.4f}) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
