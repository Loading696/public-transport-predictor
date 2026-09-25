"""Baseline ML pipeline for Moscow transport delay prediction.

Vectorized Polars implementation with strict anti-leakage:
telemetry is joined per (tr_id, T) with condition event_time <= T
via a backward asof-join (no .iterrows(), no future peek).

Features:
  - cur_dev_s        (hint from points/labels, known at T)
  - speed_current    (last known speed with event_time <= T)
  - time_to_target   (target_time_begin - T, seconds)
  - hour             (hour of target_time_begin)
  - day_of_week      (ISO weekday of target_time_begin, Mon=1..Sun=7)

Train: CatBoostRegressor on train labels + train traffic.
Predict: validate points + validate traffic -> submission.csv (; separated).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"

FEATURES = ["cur_dev_s", "speed_current", "time_to_target", "hour", "day_of_week"]
TARGET = "target_delay_s"

DT_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_dt(expr: pl.Expr) -> pl.Expr:
    """Parse 'YYYY-MM-DD HH:MM:SS[.nanos]' -> Datetime (second precision)."""
    return expr.str.slice(0, 19).str.strptime(pl.Datetime, DT_FORMAT, strict=False)


def load_points(path: Path, *, is_labels: bool) -> pl.DataFrame:
    print(f"[load] points: {path}", flush=True)
    df = pl.read_csv(path)
    print(f"[load] rows={df.height} cols={df.columns}", flush=True)
    df = df.with_columns(
        _parse_dt(pl.col("T")).alias("T_dt"),
        _parse_dt(pl.col("target_time_begin")).alias("target_dt"),
        pl.col("tr_id").cast(pl.Int64, strict=False),
        pl.col("cur_dev_s").cast(pl.Float64, strict=False),
    )
    if is_labels:
        df = df.with_columns(pl.col(TARGET).cast(pl.Float64, strict=False))
    return df


def load_traffic(path: Path) -> pl.DataFrame:
    print(f"[load] traffic: {path}", flush=True)
    df = pl.read_csv(
        path,
        columns=["tr_id", "event_time", "speed"],
    )
    print(f"[load] traffic rows={df.height}", flush=True)
    df = df.with_columns(
        pl.col("tr_id").cast(pl.Int64, strict=False),
        _parse_dt(pl.col("event_time")).alias("event_time_dt"),
        pl.col("speed").cast(pl.Float64, strict=False).alias("speed_current"),
    ).select(["tr_id", "event_time_dt", "speed_current"])
    return df


def build_features(points: pl.DataFrame, traffic: pl.DataFrame) -> pl.DataFrame:
    """Backward asof-join last telemetry (event_time <= T) per tr_id + time features."""
    print("[features] building (asof join event_time <= T, by tr_id)...", flush=True)
    pts = points.with_row_index("_row_nr").sort("T_dt")
    trf = traffic.sort("event_time_dt")
    joined = pts.join_asof(
        trf,
        left_on="T_dt",
        right_on="event_time_dt",
        by="tr_id",
        strategy="backward",
    )
    out = (
        joined.with_columns(
            ((pl.col("target_dt") - pl.col("T_dt")).dt.total_seconds().cast(pl.Float64)).alias("time_to_target"),
            pl.col("target_dt").dt.hour().cast(pl.Float64).alias("hour"),
            pl.col("target_dt").dt.weekday().cast(pl.Float64).alias("day_of_week"),
        )
        .sort("_row_nr")
        .drop("_row_nr")
    )
    n_missing_speed = out["speed_current"].null_count()
    print(
        f"[features] done: rows={out.height}, missing speed_current={n_missing_speed}",
        flush=True,
    )
    return out


def train_and_predict() -> None:
    print("[step 1/4] loading train data", flush=True)
    train_points = load_points(DATASET / "labels" / "labels_train.csv", is_labels=True)
    train_traffic = load_traffic(DATASET / "train" / "traffic.csv")

    print("[step 2/4] building train features", flush=True)
    train_df = build_features(train_points, train_traffic)
    train_df = train_df.drop_nulls(subset=FEATURES[:1] + [TARGET])
    print(f"[train] usable rows={train_df.height}", flush=True)

    X = train_df.select(FEATURES).to_numpy()
    y = train_df.select(TARGET).to_numpy().ravel()
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, random_state=42)
    print(f"[train] split: fit={len(X_tr)} eval={len(X_va)}", flush=True)

    print("[step 3/4] training CatBoostRegressor (MAE, 1000 iters, early_stop=50)", flush=True)
    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=1000,
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    va_pred = model.predict(X_va)
    print(f"[train] best_iteration={model.best_iteration_} val_MAE={mean_absolute_error(y_va, va_pred):.3f}", flush=True)

    print("[step 4/4] loading validate + predicting", flush=True)
    valid_points = load_points(DATASET / "validate" / "points.csv", is_labels=False)
    valid_traffic = load_traffic(DATASET / "validate" / "traffic.csv")
    valid_df = build_features(valid_points, valid_traffic)

    X_valid = valid_df.select(FEATURES).to_numpy()
    preds = model.predict(X_valid)
    print(f"[predict] n={len(preds)}", flush=True)

    submission = pl.DataFrame(
        {"sample_id": valid_df["sample_id"], "prediction": preds}
    )
    assert submission.height == valid_points.height, "coverage mismatch"
    assert submission["sample_id"].n_unique() == submission.height, "duplicate sample_id"

    out_path = ROOT / "submission.csv"
    submission.write_csv(out_path, separator=";")
    print(f"[done] wrote {out_path} ({submission.height} rows, sep=';')", flush=True)


def main() -> None:
    train_and_predict()


if __name__ == "__main__":
    main()
