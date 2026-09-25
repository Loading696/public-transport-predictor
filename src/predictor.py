"""Causal feature engineering and delay prediction for the NDTP hackathon data.

The feature builder deliberately never reads ``time_fact_begin``.  Labels are
used only as the supervised target; all telemetry-derived features are cut at
the forecast timestamp ``T``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error


EARTH_RADIUS_KM = 6371.0088
WINDOW_SECONDS = (60, 180, 300, 600, 900, 1800)
SPEED_LIMITS = (2.0, 5.0, 20.0, 40.0)


def read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read a UTF-8 CSV while keeping mixed packet identifiers stable."""
    return pd.read_csv(path, low_memory=False, **kwargs)


def parse_point_wkt(value: object) -> tuple[float, float]:
    """Return ``(lon, lat)`` from the POINT WKT used by the schedule."""
    if not isinstance(value, str):
        return math.nan, math.nan
    match = re.search(
        r"POINT\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", value
    )
    if not match:
        return math.nan, math.nan
    return float(match.group(1)), float(match.group(2))


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: float, lat2: float) -> np.ndarray:
    """Vectorized great-circle distance in kilometres."""
    lon1 = np.asarray(lon1, dtype=float)
    lat1 = np.asarray(lat1, dtype=float)
    phi1 = np.deg2rad(lat1)
    lat2_arr = np.asarray(lat2, dtype=float)
    phi2 = np.deg2rad(lat2_arr)
    dphi = np.deg2rad(lat2_arr - lat1)
    dlambda = np.deg2rad(np.asarray(lon2, dtype=float) - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _safe_stats(values: np.ndarray, prefix: str, result: dict[str, float]) -> None:
    """Add robust speed statistics for one telemetry window."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        result[f"{prefix}_mean"] = math.nan
        result[f"{prefix}_median"] = math.nan
        result[f"{prefix}_std"] = math.nan
        result[f"{prefix}_p10"] = math.nan
        result[f"{prefix}_p90"] = math.nan
        result[f"{prefix}_last"] = math.nan
        result[f"{prefix}_moving_frac"] = math.nan
        for limit in SPEED_LIMITS:
            result[f"{prefix}_le_{int(limit)}"] = math.nan
        return

    result[f"{prefix}_mean"] = float(np.mean(values))
    result[f"{prefix}_median"] = float(np.median(values))
    result[f"{prefix}_std"] = float(np.std(values))
    result[f"{prefix}_p10"] = float(np.quantile(values, 0.10))
    result[f"{prefix}_p90"] = float(np.quantile(values, 0.90))
    result[f"{prefix}_last"] = float(values[-1])
    result[f"{prefix}_moving_frac"] = float(np.mean(values > 2.0))
    for limit in SPEED_LIMITS:
        result[f"{prefix}_le_{int(limit)}"] = float(np.mean(values <= limit))


def _telemetry_features_for_point(
    *,
    end: int,
    times_ns: np.ndarray,
    speeds: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    valid_locations: np.ndarray,
    cumulative_distance: np.ndarray,
    target_lon: float,
    target_lat: float,
) -> dict[str, float]:
    """Compute features from the causal telemetry prefix ending at ``end``."""
    out: dict[str, float] = {}
    if end <= 0:
        return out

    last_time_ns = int(times_ns[end - 1])
    out["telemetry_points"] = float(end)
    out["last_event_age_s"] = float((int(times_ns[-1]) - last_time_ns) / 1e9) if False else math.nan
    # The caller supplies the query timestamp separately through the array
    # offset below; keeping the initial value finite makes missing-history cases
    # explicit rather than accidentally using a future row.
    out.pop("last_event_age_s", None)

    for window in WINDOW_SECONDS:
        start = int(np.searchsorted(times_ns, last_time_ns - window * 1_000_000_000, side="left"))
        idx = slice(start, end)
        sp = speeds[idx]
        lo = lons[idx]
        la = lats[idx]
        valid = valid_locations[idx] & np.isfinite(lo) & np.isfinite(la)
        _safe_stats(sp, f"speed_{window}s", out)

        if np.any(valid):
            valid_lon = lo[valid]
            valid_lat = la[valid]
            # Displacement and path length over the window.  Invalid GPS gaps
            # are intentionally not interpolated.
            out[f"gps_valid_{window}s"] = float(np.mean(valid))
            out[f"displacement_{window}s"] = float(
                haversine_km(valid_lon[-1:], valid_lat[-1:], valid_lon[0], valid_lat[0])[0]
            )
            if end - start > 1:
                seg = haversine_km(valid_lon[:-1], valid_lat[:-1], valid_lon[1:], valid_lat[1:])
                out[f"path_distance_{window}s"] = float(np.nansum(seg))
            else:
                out[f"path_distance_{window}s"] = 0.0
        else:
            out[f"gps_valid_{window}s"] = 0.0
            out[f"displacement_{window}s"] = math.nan
            out[f"path_distance_{window}s"] = math.nan

        # Cumulative path distance is precomputed from the valid GPS stream.
        if end > start:
            out[f"path_distance_cumulative_{window}s"] = float(
                cumulative_distance[end - 1] - cumulative_distance[start]
            )
        else:
            out[f"path_distance_cumulative_{window}s"] = 0.0

    # Latest valid location and its distance to the target.
    valid_idx = np.flatnonzero(valid_locations[:end] & np.isfinite(lons[:end]) & np.isfinite(lats[:end]))
    if valid_idx.size:
        j = int(valid_idx[-1])
        out["last_valid_lon"] = float(lons[j])
        out["last_valid_lat"] = float(lats[j])
        out["target_distance_km"] = float(haversine_km(lons[j], lats[j], target_lon, target_lat))
        out["target_distance_per_speed"] = (
            out["target_distance_km"] / max(float(speeds[j]), 2.0)
            if np.isfinite(speeds[j])
            else math.nan
        )
        out["last_valid_age_s"] = float((last_time_ns - int(times_ns[j])) / 1e9)
    else:
        out["last_valid_lon"] = math.nan
        out["last_valid_lat"] = math.nan
        out["target_distance_km"] = math.nan
        out["target_distance_per_speed"] = math.nan
        out["last_valid_age_s"] = math.nan
    return out


def _vehicle_telemetry_features(
    traffic: pd.DataFrame,
    points: pd.DataFrame,
    schedule_by_vehicle: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Build one row of causal telemetry features for every forecast point."""
    traffic = traffic.copy()
    traffic["event_time"] = pd.to_datetime(traffic["event_time"], errors="coerce")
    traffic["speed"] = pd.to_numeric(traffic["speed"], errors="coerce")
    traffic["lon"] = pd.to_numeric(traffic["lon"], errors="coerce")
    traffic["lat"] = pd.to_numeric(traffic["lat"], errors="coerce")
    traffic["location_valid"] = (
        traffic["location_valid"].astype(str).str.lower().isin({"true", "1", "yes"})
        if traffic["location_valid"].dtype == object
        else traffic["location_valid"].fillna(False).astype(bool)
    )

    records: list[dict[str, object]] = []
    grouped = {int(k): g for k, g in traffic.groupby("tr_id", sort=False)}
    for point_no, (_, point) in enumerate(points.iterrows(), start=1):
        tid = int(point["tr_id"])
        t = pd.Timestamp(point["T"])
        schedule = schedule_by_vehicle.get(tid)
        target_lon = float(point.get("target_lon", math.nan))
        target_lat = float(point.get("target_lat", math.nan))
        if schedule is not None and np.isfinite(target_lon) and np.isfinite(target_lat):
            # Target coordinates are normally already present in the point
            # frame; this branch only supplies a fallback for malformed rows.
            target_lon = float(point.get("target_lon", target_lon))
            target_lat = float(point.get("target_lat", target_lat))

        g = grouped.get(tid)
        if g is None or g.empty:
            records.append({"row_no": point_no, "tr_id": tid, "telemetry_missing": 1.0})
            continue

        g = g.sort_values("event_time", kind="stable")
        times_ns = g["event_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        speeds = g["speed"].to_numpy(dtype=float)
        lons = g["lon"].to_numpy(dtype=float)
        lats = g["lat"].to_numpy(dtype=float)
        valid_locations = g["location_valid"].to_numpy(dtype=bool)

        # GPS path length, with invalid coordinates treated as gaps.
        cumulative_distance = np.zeros(len(g), dtype=float)
        for i in range(1, len(g)):
            if valid_locations[i] and valid_locations[i - 1] and np.isfinite(lons[i]) and np.isfinite(lons[i - 1]) and np.isfinite(lats[i]) and np.isfinite(lats[i - 1]):
                cumulative_distance[i] = cumulative_distance[i - 1] + float(haversine_km(lons[i - 1], lats[i - 1], lons[i], lats[i]))
            else:
                cumulative_distance[i] = cumulative_distance[i - 1]

        t_ns = pd.Timestamp(t).value
        end = int(np.searchsorted(times_ns, t_ns, side="right"))
        row: dict[str, object] = {"row_no": point_no, "tr_id": tid, "telemetry_missing": 0.0}
        if end == 0:
            row["last_event_age_s"] = math.nan
        else:
            row["last_event_age_s"] = float((t_ns - int(times_ns[end - 1])) / 1e9)
        row.update(
            _telemetry_features_for_point(
                end=end,
                times_ns=times_ns,
                speeds=speeds,
                lons=lons,
                lats=lats,
                valid_locations=valid_locations,
                cumulative_distance=cumulative_distance,
                target_lon=target_lon,
                target_lat=target_lat,
            )
        )
        records.append(row)
        if point_no % 500 == 0:
            print(f"telemetry features: {point_no}/{len(points)}", flush=True)
    return pd.DataFrame.from_records(records)


def add_schedule_features(
    points: pd.DataFrame,
    schedule_path: Path | pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Attach static planned-route features without reading actual times."""
    schedule = schedule_path.copy() if isinstance(schedule_path, pd.DataFrame) else read_csv(schedule_path)
    schedule["time_begin"] = pd.to_datetime(schedule["time_begin"], errors="coerce")
    parsed = schedule["geom"].map(parse_point_wkt)
    schedule["stop_lon"] = [x[0] for x in parsed]
    schedule["stop_lat"] = [x[1] for x in parsed]
    schedule = schedule.sort_values(["tr_id", "time_begin", "tt_action_item_id"], kind="stable").reset_index(drop=True)
    schedule["stop_idx"] = schedule.groupby("tr_id").cumcount()
    schedule["route_stop_count"] = schedule.groupby("tr_id")["tt_action_item_id"].transform("size")
    schedule["planned_gap_prev_s"] = schedule.groupby("tr_id")["time_begin"].diff().dt.total_seconds()
    schedule["planned_gap_next_s"] = schedule.groupby("tr_id")["time_begin"].diff(-1).dt.total_seconds()
    schedule["leg_distance_km"] = math.nan
    for tid, group in schedule.groupby("tr_id", sort=False):
        idx = group.index.to_numpy()
        if len(idx) > 1:
            d = haversine_km(
                group["stop_lon"].to_numpy()[:-1],
                group["stop_lat"].to_numpy()[:-1],
                # vectorized haversine below needs arrays; use a small loop-free
                # expression for the next stop coordinates.
                np.asarray(group["stop_lon"].to_numpy()[1:], dtype=float),
                np.asarray(group["stop_lat"].to_numpy()[1:], dtype=float),
            )
            schedule.loc[idx[:-1], "leg_distance_km"] = d

    target_cols = [
        "tt_action_item_id", "tr_id", "time_begin", "stop_lon", "stop_lat", "stop_idx",
        "route_stop_count", "planned_gap_prev_s", "planned_gap_next_s",
    ]
    target_lookup = schedule[target_cols].rename(
        columns={
            "tt_action_item_id": "target_stop_id",
            "time_begin": "schedule_target_time",
            "stop_lon": "target_lon",
            "stop_lat": "target_lat",
            "stop_idx": "target_stop_idx",
            "route_stop_count": "target_route_stop_count",
            "planned_gap_prev_s": "target_prev_gap_s",
            "planned_gap_next_s": "target_next_gap_s",
        }
    )
    out = points.merge(
        target_lookup,
        on=["target_stop_id", "tr_id"],
        how="left",
        validate="many_to_one",
    )
    out["T_dt"] = pd.to_datetime(out["T"], errors="coerce")
    out["target_time_dt"] = pd.to_datetime(out["target_time_begin"], errors="coerce")
    out["lead_s"] = (out["target_time_dt"] - out["T_dt"]).dt.total_seconds()
    out["target_hour"] = out["target_time_dt"].dt.hour + out["target_time_dt"].dt.minute / 60.0
    out["T_hour"] = out["T_dt"].dt.hour + out["T_dt"].dt.minute / 60.0
    out["T_minute"] = out["T_dt"].dt.minute
    out["T_dayofweek"] = out["T_dt"].dt.dayofweek
    out["is_weekend"] = (out["T_dayofweek"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2.0 * np.pi * out["T_hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["T_hour"] / 24.0)
    out["cur_abs"] = out["cur_dev_s"].abs()
    out["cur_sq"] = out["cur_dev_s"] ** 2
    out["cur_sign"] = np.sign(out["cur_dev_s"])
    out["cur_positive"] = (out["cur_dev_s"] > 0).astype(int)
    out["lead_minus_660"] = out["lead_s"] - 660.0

    # Planned stops between the forecast instant and the target, and simple
    # summaries of their scheduled headways.
    schedule_by_vehicle = {int(k): g for k, g in schedule.groupby("tr_id", sort=False)}
    between = []
    gap_mean = []
    gap_std = []
    gap_min = []
    gap_max = []
    for _, row in out.iterrows():
        g = schedule_by_vehicle.get(int(row["tr_id"]))
        if g is None:
            between.append(math.nan); gap_mean.append(math.nan); gap_std.append(math.nan); gap_min.append(math.nan); gap_max.append(math.nan)
            continue
        t = row["T_dt"]
        target_t = row["target_time_dt"]
        q = g[(g["time_begin"] > t) & (g["time_begin"] <= target_t)]
        between.append(float(len(q)))
        gaps = q["planned_gap_prev_s"].to_numpy(dtype=float)
        gaps = gaps[np.isfinite(gaps)]
        gap_mean.append(float(np.mean(gaps)) if gaps.size else math.nan)
        gap_std.append(float(np.std(gaps)) if gaps.size else math.nan)
        gap_min.append(float(np.min(gaps)) if gaps.size else math.nan)
        gap_max.append(float(np.max(gaps)) if gaps.size else math.nan)
    out["planned_stops_between"] = between
    out["between_gap_mean_s"] = gap_mean
    out["between_gap_std_s"] = gap_std
    out["between_gap_min_s"] = gap_min
    out["between_gap_max_s"] = gap_max
    out["target_stop_fraction"] = out["target_stop_idx"] / out["target_route_stop_count"].replace(0, np.nan)
    out["target_prev_gap_s"] = out["target_prev_gap_s"].fillna(-1.0)
    out["target_next_gap_s"] = out["target_next_gap_s"].fillna(-1.0)
    return out, schedule_by_vehicle


def build_features(
    points: pd.DataFrame,
    traffic_path: Path,
    schedule_path: Path,
) -> pd.DataFrame:
    """Build static and causal telemetry features for a point set."""
    return build_features_from_frames(
        points,
        read_csv(traffic_path),
        read_csv(schedule_path),
    )


def build_features_from_frames(
    points: pd.DataFrame,
    traffic: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    points = points.copy()
    points["tr_id"] = points["tr_id"].astype(int)
    points["row_no"] = np.arange(1, len(points) + 1, dtype=np.int64)
    static, schedule_by_vehicle = add_schedule_features(points, schedule)
    telemetry = _vehicle_telemetry_features(traffic, static, schedule_by_vehicle)
    return static.merge(telemetry, on=["row_no", "tr_id"], how="left", validate="one_to_one")


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Select numeric features; identifiers are added separately as categories."""
    excluded = {
        "sample_id", "tr_id", "T", "target_stop_id", "target_time_begin",
        "target_delay_s", "target_class", "T_dt", "target_time_dt",
        "schedule_target_time", "row_no", "target_lon", "target_lat",
    }
    cols = [c for c in frame.columns if c not in excluded]
    return [c for c in cols if pd.api.types.is_numeric_dtype(frame[c])]


def train_model(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    *,
    use_vehicle_category: bool = True,
    iterations: int = 700,
) -> tuple[CatBoostRegressor, list[str]]:
    """Fit a robust CatBoost regressor and return it with its feature list."""
    features = select_feature_columns(train_frame)
    cat_features = ["tr_id"] if use_vehicle_category else []
    model = CatBoostRegressor(
        iterations=iterations,
        depth=7,
        learning_rate=0.035,
        loss_function="MAE",
        random_seed=2026,
        l2_leaf_reg=8.0,
        random_strength=0.7,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(
        train_frame[features + cat_features],
        train_frame["target_delay_s"],
        cat_features=cat_features,
    )
    return model, features + cat_features


def predict(model: CatBoostRegressor, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Produce predictions while tolerating missing optional columns."""
    x = frame.copy()
    for col in features:
        if col not in x:
            x[col] = np.nan
    return model.predict(x[features])


def evaluate(model: CatBoostRegressor, frame: pd.DataFrame, features: list[str]) -> float:
    """Return MAE in seconds for a labelled frame."""
    return float(mean_absolute_error(frame["target_delay_s"], predict(model, frame, features)))


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "dataset"
    train_labels = read_csv(root / "labels" / "labels_train.csv")
    test_labels = read_csv(root / "labels" / "labels_test.csv")
    validate_points = read_csv(root / "validate" / "points.csv")
    print("building train features", flush=True)
    train_frame = build_features(train_labels, root / "train" / "traffic.csv", root / "train" / "schedule.csv")
    print("building test features", flush=True)
    test_frame = build_features(test_labels, root / "test" / "traffic.csv", root / "test" / "schedule.csv")
    print("building validate features", flush=True)
    validate_frame = build_features(validate_points, root / "validate" / "traffic.csv", root / "validate" / "schedule_plan.csv")
    model, features = train_model(train_frame, test_frame)
    pred_test = predict(model, test_frame, features)
    print(f"test MAE: {mean_absolute_error(test_labels['target_delay_s'], pred_test):.3f}")
    print(f"baseline MAE: {mean_absolute_error(test_labels['target_delay_s'], test_labels['cur_dev_s']):.3f}")
    pred = predict(model, validate_frame, features)
    submission = pd.DataFrame({"sample_id": validate_points["sample_id"], "prediction": pred})
    submission.to_csv(root.parent / "submission.csv", index=False, sep=";", float_format="%.6f")
    print(f"wrote {root.parent / 'submission.csv'} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
