"""Sanity-check анти-утечки для validate-контура.

Проверки:
  1. submission.csv: формат `;`, колонки sample_id;prediction, полное покрытие.
     Если файла нет, он генерируется через InferenceService, чтобы скрипт
     запускался из свежего клона (submission.csv не коммитится).
  2. Граница среза: times[end-1] <= T < times[end] (searchsorted right).
  3. Запрещенные колонки: time_fact_begin / target_* отсутствуют в фичах
     мини-сборки build_features на 5 точках + статический аудит исходника.

Usage:
    py ml/sanity_check.py [--full]   # по умолчанию 50 точек, --full = все 151
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "dataset"
N_SAMPLE = 50
SUBMISSION = ROOT / "submission.csv"


def ensure_submission() -> None:
    """Generate submission.csv when absent so the check is runnable standalone.

    Generation runs the ML pipeline in-process (see ``ml/ml_pipeline.py``) rather
    than calling the deployed service, so the anti-leak check works from a fresh
    clone with no services up.  The HTTP boundary has its own test,
    ``ml/test_ml_integration.py``.
    """
    if SUBMISSION.exists():
        return
    from ml.ml_pipeline import write_submission

    print(f"[info] {SUBMISSION.name} не найден, генерирую через ML-пайплайн...")
    write_submission(ROOT)
    if not SUBMISSION.exists():
        raise AssertionError(
            f"{SUBMISSION.name} не создан: проверьте наличие ml/model.cbm и dataset/validate/"
        )


def check_submission() -> None:
    ensure_submission()
    path = SUBMISSION
    assert path.exists(), "submission.csv не найден"
    sub = pl.read_csv(path, separator=";")
    assert sub.columns == ["sample_id", "prediction"], f"колонки: {sub.columns}"
    pts = pl.read_csv(DATASET / "validate" / "points.csv", columns=["sample_id"])
    assert sub.height == pts.height, f"покрытие: {sub.height} vs {pts.height}"
    assert sub["sample_id"].n_unique() == sub.height, "дубли sample_id"
    assert sub["prediction"].null_count() == 0, "NaN в prediction"
    print(f"[PASS] submission: {sub.height} строк, sep=';', покрытие полное")


def check_cutoff(n: int) -> None:
    pts = (
        pl.read_csv(DATASET / "validate" / "points.csv", columns=["sample_id", "tr_id", "T"])
        .with_columns(pl.col("T").str.slice(0, 19).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False))
        .with_row_index("_i")
    )
    sample = pts.sample(min(n, pts.height), seed=42).sort("_i") if pts.height > n else pts
    trf = (
        pl.read_csv(DATASET / "validate" / "traffic.csv", columns=["tr_id", "event_time"])
        .with_columns(pl.col("event_time").str.slice(0, 19).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False))
        .drop_nulls()
    )
    bad = 0
    for row in sample.iter_rows(named=True):
        ev = trf.filter(pl.col("tr_id") == row["tr_id"]).sort("event_time")["event_time"]
        used = ev.filter(ev <= row["T"])
        future = ev.filter(ev > row["T"])
        # граница searchsorted(right): последний used <= T, первый future > T
        if used.len() and not (used.max() <= row["T"]):
            bad += 1
            print(f"[FAIL] {row['sample_id']}: used.max > T")
        if future.len() and used.len() and not (future.min() > used.max()):
            bad += 1
            print(f"[FAIL] {row['sample_id']}: порядок нарушен")
    assert bad == 0, f"нарушений границы: {bad}"
    print(f"[PASS] cutoff event_time <= T на {sample.height} точках (нарушений: 0)")


def check_forbidden_columns() -> None:
    from src.predictor import build_features, read_csv, select_feature_columns

    mini = read_csv(DATASET / "validate" / "points.csv").head(5)
    frame = build_features(mini, DATASET / "validate" / "traffic.csv", DATASET / "validate" / "schedule_plan.csv")
    forbidden = {"time_fact_begin", "target_delay_s", "target_class"}
    leak = forbidden & set(frame.columns)
    assert not leak, f"запрещенные колонки в фичах: {leak}"
    feats = set(select_feature_columns(frame))
    assert not (forbidden & feats), "запрещенные колонки в списке фичей модели"
    src = (ROOT / "src" / "predictor.py").read_text(encoding="utf-8")
    uses = [ln for ln in src.splitlines() if "time_fact_begin" in ln and "never reads" not in ln]
    assert not uses, f"time_fact_begin используется: {uses}"
    print("[PASS] time_fact_begin/target отсутствуют в признаках (мини-сборка + аудит исходника)")


def main() -> None:
    n = N_SAMPLE if "--full" not in sys.argv else 10**9
    check_submission()
    check_cutoff(n)
    check_forbidden_columns()
    print("ALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
