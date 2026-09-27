"""Baseline v2: rich causal features from src/predictor + CatBoost.

Reuses the proven feature builder from src/predictor.py (causal telemetry
windows, schedule features, anti-leak event_time <= T) and adds:
  - honest eval on labels_test (MAE vs naive cur_dev_s),
  - final refit on train+test for the submission,
  - submission.csv in project root (sep=';', sample_id;prediction).

Usage:
    py ml/baseline_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predictor import build_features, predict, read_csv, train_model  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    t0 = time.time()
    dataset = ROOT / "dataset"

    log("[v2 1/5] building train features...")
    train_labels = read_csv(dataset / "labels" / "labels_train.csv")
    train_frame = build_features(
        train_labels, dataset / "train" / "traffic.csv", dataset / "train" / "schedule.csv"
    )
    log(f"[v2] train_frame: {train_frame.shape}")

    log("[v2 2/5] building test features...")
    test_labels = read_csv(dataset / "labels" / "labels_test.csv")
    test_frame = build_features(
        test_labels, dataset / "test" / "traffic.csv", dataset / "test" / "schedule.csv"
    )
    log(f"[v2] test_frame: {test_frame.shape}")

    log("[v2 3/5] training CatBoost on train...")
    model, features = train_model(train_frame)
    log(f"[v2] n_features={len(features)}")

    pred_test = predict(model, test_frame, features)
    mae_test = float(mean_absolute_error(test_labels["target_delay_s"], pred_test))
    mae_naive = float(mean_absolute_error(test_labels["target_delay_s"], test_labels["cur_dev_s"]))
    log(f"[v2] test MAE model={mae_test:.3f} naive(cur_dev_s)={mae_naive:.3f}")

    log("[v2 4/5] refit on train+test for final submission...")
    full_frame = pd.concat([train_frame, test_frame], ignore_index=True)
    model_full, features_full = train_model(full_frame)
    assert features_full == features, "feature list changed after refit"

    log("[v2 5/5] building validate features + predicting...")
    validate_points = read_csv(dataset / "validate" / "points.csv")
    validate_frame = build_features(
        validate_points,
        dataset / "validate" / "traffic.csv",
        dataset / "validate" / "schedule_plan.csv",
    )
    log(f"[v2] validate_frame: {validate_frame.shape}")
    preds = predict(model_full, validate_frame, features_full)

    # The submission format lives in ml/submission.py, shared with the HTTP
    # endpoint and the offline tools, so a retrained model cannot produce a file
    # in a different shape than the deployed one.
    from ml.submission import SubmissionFormatError, write_submission

    out_path = ROOT / "submission.csv"
    try:
        report = write_submission(validate_points, list(preds), out_path)
    except SubmissionFormatError as exc:
        log(f"[v2 FAILED] {exc}")
        raise
    if not report.ok:
        log(f"[v2 FAILED] invalid submission: {'; '.join(report.errors)}")
        raise ValueError("submission failed validation")
    log(
        f"[v2 done] wrote {out_path} ({report.rows} rows, "
        f"min={report.min_prediction:.2f} max={report.max_prediction:.2f}) "
        f"in {time.time() - t0:.1f}s"
    )

    model_path = ROOT / "ml" / "model.cbm"
    features_path = ROOT / "ml" / "features.json"
    model_full.save_model(str(model_path))
    features_path.write_text(json.dumps(features_full, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[v2 done] saved model -> {model_path}, features ({len(features_full)}) -> {features_path}")


if __name__ == "__main__":
    main()
