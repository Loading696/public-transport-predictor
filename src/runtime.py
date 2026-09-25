from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import time
from bisect import bisect_right
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.predictor import build_features_from_frames, haversine_km, parse_point_wkt
from src.predictor import predict as model_predict
from src.patterns import detect_all

TELEMETRY_COLUMNS = [
    "tr_id",
    "event_time",
    "location_valid",
    "lon",
    "lat",
    "alt",
    "speed",
    "heading",
]
POINT_COLUMNS = [
    "sample_id",
    "tr_id",
    "T",
    "target_stop_id",
    "target_time_begin",
    "cur_dev_s",
]


def target_class_for(delay_s: float) -> str:
    if delay_s < -60.0:
        return "early"
    if delay_s > 120.0:
        return "late"
    return "ontime"


def risk_for(delay_s: float) -> str:
    if delay_s > 120.0:
        return "late"
    if delay_s >= 60.0:
        return "at-risk"
    return "on-time"


def recommendation_for(delay_s: float) -> str:
    if delay_s > 300.0:
        return "рассмотреть выпуск резервного ТС"
    if delay_s > 120.0:
        return "скорректировать интервал на маршруте"
    if delay_s > 0.0:
        return "контролировать движение ТС"
    return "наблюдение без вмешательства"


RISK_COLORS = {"on-time": "#22c55e", "at-risk": "#eab308", "late": "#ef4444"}

CAUSE_LABELS = {
    "stale": "телеметрия недостоверна (простой потока или потеря GPS)",
    "backlog": "накопленное отставание от графика сохраняется к целевой остановке",
    "dwell": "простой или посадка на подходе к целевой остановке",
    "speed_drop": "аномальное снижение скорости на подходе к участку",
}
CAUSE_FALLBACK = "отклонение операционного режима от планового графика"


def cause_from_events(events: Any) -> tuple[str, str]:
    """Pick the incident cause from pattern events: data quality first, then strong signal."""
    if not isinstance(events, list) or not events:
        return CAUSE_FALLBACK, "none"
    ranked = sorted(
        (e for e in events if isinstance(e, dict)),
        key=lambda e: ({"data_quality": 0, "strong_signal": 1}.get(e.get("role"), 2), -float(e.get("confidence", 0.0))),
    )
    for event in ranked:
        label = CAUSE_LABELS.get(str(event.get("type")))
        if label is not None:
            return label, str(event.get("role", "weak_signal"))
    return CAUSE_FALLBACK, "none"


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _latest_position(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    for row in reversed(list(rows)):
        if not bool(row.get("location_valid", False)):
            continue
        lon = _finite_float(row.get("lon"))
        lat = _finite_float(row.get("lat"))
        if lon is None or lat is None:
            continue
        return {
            "lon": lon,
            "lat": lat,
            "event_time": _json_value(row.get("event_time")),
            "speed": _json_value(row.get("speed")),
            "heading": _json_value(row.get("heading")),
        }
    return None


def _route_network(schedule: pd.DataFrame) -> dict[int, dict[str, Any]]:
    frame = schedule.copy()
    frame["time_begin"] = pd.to_datetime(frame["time_begin"], errors="coerce")
    frame["tr_id"] = pd.to_numeric(frame["tr_id"], errors="coerce")
    frame = frame.dropna(subset=["tr_id"]).copy()
    frame["tr_id"] = frame["tr_id"].astype(int)
    parsed = frame["geom"].map(parse_point_wkt)
    frame["lon"] = [value[0] for value in parsed]
    frame["lat"] = [value[1] for value in parsed]
    frame = frame.sort_values(["tr_id", "time_begin", "tt_action_item_id"], kind="stable")
    routes: dict[int, dict[str, Any]] = {}
    for vehicle_id, group in frame.groupby("tr_id", sort=False):
        stops = [
            {
                "stop_id": _json_value(row.get("tt_action_item_id")),
                "time": _json_value(row.get("time_begin")),
                "address": _json_value(row.get("building_address")),
                "lon": _json_value(row.get("lon")),
                "lat": _json_value(row.get("lat")),
            }
            for _, row in group.iterrows()
        ]
        routes[int(vehicle_id)] = {"tr_id": int(vehicle_id), "stops": stops}
    return routes


def _empty_traffic() -> pd.DataFrame:
    return pd.DataFrame(columns=TELEMETRY_COLUMNS)


def _normalise_telemetry(traffic: pd.DataFrame | Iterable[dict[str, Any]] | None) -> pd.DataFrame:
    if traffic is None:
        return _empty_traffic()
    frame = traffic.copy() if isinstance(traffic, pd.DataFrame) else pd.DataFrame(list(traffic))
    if frame.empty:
        return _empty_traffic()
    for column in TELEMETRY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["tr_id"] = pd.to_numeric(frame["tr_id"], errors="coerce")
    frame = frame.dropna(subset=["tr_id"])
    frame["tr_id"] = frame["tr_id"].astype(int)
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="coerce")
    frame = frame.dropna(subset=["event_time"])
    frame["speed"] = pd.to_numeric(frame["speed"], errors="coerce")
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    if frame["location_valid"].dtype == object:
        frame["location_valid"] = frame["location_valid"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    else:
        frame["location_valid"] = frame["location_valid"].fillna(False).astype(bool)
    return frame[TELEMETRY_COLUMNS].sort_values("event_time", kind="stable").reset_index(drop=True)


def _normalise_points(points: pd.DataFrame | Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = points.copy() if isinstance(points, pd.DataFrame) else pd.DataFrame(list(points))
    if frame.empty:
        raise ValueError("points must not be empty")
    for column in POINT_COLUMNS:
        if column not in frame.columns:
            if column == "sample_id":
                frame[column] = ""
            else:
                raise ValueError(f"missing point column: {column}")
    frame["tr_id"] = pd.to_numeric(frame["tr_id"], errors="raise").astype(int)
    frame["target_stop_id"] = pd.to_numeric(frame["target_stop_id"], errors="raise").astype("int64")
    frame["cur_dev_s"] = pd.to_numeric(frame["cur_dev_s"], errors="raise").astype(float)
    frame["T"] = pd.to_datetime(frame["T"], errors="raise")
    frame["target_time_begin"] = pd.to_datetime(frame["target_time_begin"], errors="raise")
    generated = frame["sample_id"].isna() | frame["sample_id"].astype(str).eq("")
    if generated.any():
        generated_ids = [
            f"{int(tr_id)}_{int(timestamp.timestamp())}"
            for tr_id, timestamp in zip(frame.loc[generated, "tr_id"], frame.loc[generated, "T"])
        ]
        frame.loc[generated, "sample_id"] = generated_ids
    frame["sample_id"] = frame["sample_id"].astype(str)
    return frame[POINT_COLUMNS].reset_index(drop=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (str, bytes, list, tuple, dict)) else False:
        return None
    return value


def _record_for_response(row: pd.Series) -> dict[str, Any]:
    return {
        "sample_id": _json_value(row.get("sample_id")),
        "tr_id": int(row["tr_id"]),
        "T": _json_value(row.get("T")),
        "target_stop_id": _json_value(row.get("target_stop_id")),
        "target_time_begin": _json_value(row.get("target_time_begin")),
        "prediction": float(row["prediction"]),
        "target_class": str(row["target_class"]),
        "risk": str(row["risk"]),
        "recommendation": str(row["recommendation"]),
        "p_late": _json_value(row.get("p_late")),
        "pattern_events": row.get("pattern_events") if isinstance(row.get("pattern_events"), list) else [],
    }


class InferenceService:
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        dataset_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        features_path: str | Path | None = None,
    ) -> None:
        project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
        self.root = project_root.resolve()
        dataset_value = dataset_dir if dataset_dir is not None else os.getenv("DATASET_DIR", "dataset")
        model_value = model_path if model_path is not None else os.getenv("MODEL_PATH", "ml/model.cbm")
        features_value = features_path if features_path is not None else os.getenv("FEATURES_PATH", "ml/features.json")
        self.dataset_dir = self._path(dataset_value)
        self.model_path = self._path(model_value)
        self.features_path = self._path(features_value)
        self.points = self._read_csv(self.dataset_dir / "validate" / "points.csv")
        self.traffic = self._read_csv(self.dataset_dir / "validate" / "traffic.csv")
        self.schedule = self._read_csv(self.dataset_dir / "validate" / "schedule_plan.csv")
        self.model = CatBoostRegressor()
        self.model.load_model(str(self.model_path))
        self.features = json.loads(self.features_path.read_text(encoding="utf-8"))
        if not isinstance(self.features, list) or not self.features:
            raise ValueError(f"invalid feature list: {self.features_path}")
        self.prob_cal = self._load_prob_cal(self.root / "ml" / "prob_cal.json")

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, low_memory=False)

    @staticmethod
    def _load_prob_cal(path: Path) -> tuple[list[float], list[float]] | None:
        """Stepwise isotonic curve P(late | prediction); None when unavailable."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            xs = [float(v) for v in payload.get("X", [])]
            ps = [float(v) for v in payload.get("p", [])]
        except (OSError, ValueError, TypeError, AttributeError):
            return None
        if len(xs) != len(ps) or not xs:
            return None
        return xs, ps

    def p_late_for(self, delay_s: float) -> float | None:
        """Calibrated P(delay > 120s) for a point prediction; None without curve."""
        if self.prob_cal is None:
            return None
        try:
            value = float(delay_s)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        xs, ps = self.prob_cal
        idx = max(0, min(bisect_right(xs, value) - 1, len(ps) - 1))
        return float(ps[idx])

    def predict_frame(
        self,
        points: pd.DataFrame | Iterable[dict[str, Any]],
        traffic: pd.DataFrame | Iterable[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        point_frame = _normalise_points(points)
        source_traffic = self.traffic if traffic is None else traffic
        telemetry = _normalise_telemetry(source_traffic)
        max_time = point_frame["T"].max()
        ids = set(point_frame["tr_id"].tolist())
        causal_traffic = telemetry[
            telemetry["tr_id"].isin(ids) & (telemetry["event_time"] <= max_time)
        ].copy()
        features_frame = build_features_from_frames(
            point_frame,
            causal_traffic,
            self.schedule,
        )
        values = np.asarray(model_predict(self.model, features_frame, self.features), dtype=float)
        if len(values) != len(point_frame) or not np.isfinite(values).all():
            raise ValueError("model returned invalid predictions")
        result = point_frame.copy()
        result["prediction"] = values
        result["target_class"] = [target_class_for(float(value)) for value in values]
        result["risk"] = [risk_for(float(value)) for value in values]
        result["recommendation"] = [recommendation_for(float(value)) for value in values]
        result["p_late"] = [self.p_late_for(float(value)) for value in values]
        pattern_events: list[list[dict[str, Any]]] = []
        for pos in range(len(result)):
            if pos < len(features_frame):
                try:
                    events = detect_all(features_frame.iloc[pos].to_dict())
                except Exception:  # noqa: BLE001 - detectors must stay total
                    events = []
            else:
                events = []
            pattern_events.append(events)
        result["pattern_events"] = pattern_events
        return result

    def predict_records(
        self,
        points: Iterable[dict[str, Any]],
        shared_telemetry: Iterable[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        point_rows: list[dict[str, Any]] = []
        telemetry_rows: list[dict[str, Any]] = list(shared_telemetry or [])
        for point in points:
            row = dict(point)
            local = row.pop("telemetry", None) or []
            telemetry_rows.extend(dict(event) for event in local)
            point_rows.append(row)
        result = self.predict_frame(point_rows, telemetry_rows or None)
        return [_record_for_response(row) for _, row in result.iterrows()]

    def generate_submission(self, output_path: str | Path | None = None) -> dict[str, Any]:
        result = self.predict_frame(self.points, self.traffic)
        expected = self.points["sample_id"].astype(str)
        actual = result["sample_id"].astype(str)
        if len(result) != len(self.points):
            raise ValueError("submission coverage mismatch")
        if actual.duplicated().any() or not actual.equals(expected.reset_index(drop=True)):
            raise ValueError("submission sample_id coverage mismatch")
        if not np.isfinite(result["prediction"].to_numpy(dtype=float)).all():
            raise ValueError("submission contains non-finite predictions")
        destination = Path(output_path) if output_path is not None else self.root / "submission.csv"
        submission = result[["sample_id", "prediction"]].copy()
        submission.to_csv(destination, index=False, sep=";", float_format="%.6f")
        return {
            "path": str(destination),
            "rows": len(submission),
            "first_rows": [
                {"sample_id": str(row["sample_id"]), "prediction": float(row["prediction"])}
                for _, row in submission.head(5).iterrows()
            ],
        }

    def stream_simulator(self, speed: float = 60.0) -> StreamSimulator:
        return StreamSimulator(self, speed=speed)


class StreamSimulator:
    def __init__(self, service: InferenceService, *, speed: float = 60.0) -> None:
        self.service = service
        self.traffic = _normalise_telemetry(service.traffic)
        self.points = service.points.copy()
        self.points["T_dt"] = pd.to_datetime(self.points["T"], errors="coerce")
        self.points["_point_order"] = np.arange(len(self.points), dtype=np.int64)
        self.points = self.points.sort_values(["T_dt", "_point_order"], kind="stable").reset_index(drop=True)
        self._traffic_rows = self.traffic[TELEMETRY_COLUMNS].to_dict("records")
        self._point_rows = self.points.drop(columns=["T_dt", "_point_order"], errors="ignore").to_dict("records")
        self._vehicles: dict[int, dict[str, Any]] = {}
        self._vehicle_telemetry: dict[int, list[dict[str, Any]]] = {}
        self._visible_traffic: list[dict[str, Any]] = []
        self._routes = _route_network(service.schedule)
        self.speed = self._safe_speed(speed)
        self._reset_state()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @staticmethod
    def _safe_speed(value: float) -> float:
        speed = float(value)
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("speed must be positive")
        return min(speed, 10000.0)

    def set_speed(self, speed: float) -> None:
        self.speed = self._safe_speed(speed)

    def _remember_telemetry(self, row: dict[str, Any]) -> None:
        self._visible_traffic.append(row)
        raw_vehicle_id = row.get("tr_id")
        if raw_vehicle_id is None:
            return
        try:
            vehicle_id = int(raw_vehicle_id)
        except (TypeError, ValueError):
            return
        self._vehicle_telemetry.setdefault(vehicle_id, []).append(row)

    def _reset_state(self) -> None:
        self._traffic_position = 0
        self._point_position = 0
        self._visible_traffic = []
        self._vehicle_telemetry = {}
        self._vehicles = {}
        point_times = [value for value in self.points["T_dt"].tolist() if pd.notna(value)]
        traffic_times = [value for value in self.traffic["event_time"].tolist() if pd.notna(value)]
        candidates = point_times or traffic_times
        self.current_time = min(candidates) if candidates else pd.Timestamp.now()
        while self._traffic_position < len(self._traffic_rows):
            event_time = pd.Timestamp(self._traffic_rows[self._traffic_position]["event_time"])
            if event_time > self.current_time:
                break
            self._remember_telemetry(self._traffic_rows[self._traffic_position])
            self._traffic_position += 1
        self.finished = False

    def reset(self, speed: float | None = None) -> None:
        self._reset_state()
        if speed is not None:
            self.set_speed(speed)

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        previous_wall = time.monotonic()
        while not self._stop_event.is_set() and not self.finished:
            await asyncio.sleep(0.1)
            current_wall = time.monotonic()
            elapsed = current_wall - previous_wall
            previous_wall = current_wall
            self.advance_to(self.current_time + pd.Timedelta(seconds=elapsed * self.speed))

    def _causal_history(self, vehicle_id: int, forecast_time: pd.Timestamp) -> list[dict[str, Any]]:
        return [
            row
            for row in self._vehicle_telemetry.get(vehicle_id, [])
            if pd.Timestamp(row.get("event_time")) <= forecast_time
        ]

    def _vehicle_schedule(self, vehicle_id: int) -> pd.DataFrame:
        schedule = self.service.schedule.copy()
        schedule["tr_id"] = pd.to_numeric(schedule["tr_id"], errors="coerce")
        schedule = schedule[schedule["tr_id"] == vehicle_id].copy()
        schedule["time_begin"] = pd.to_datetime(schedule["time_begin"], errors="coerce")
        parsed = schedule["geom"].map(parse_point_wkt)
        schedule["lon"] = [value[0] for value in parsed]
        schedule["lat"] = [value[1] for value in parsed]
        return schedule.sort_values(["time_begin", "tt_action_item_id"], kind="stable")

    @staticmethod
    def _stop_card(row: pd.Series | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "stop_id": _json_value(row.get("tt_action_item_id")),
            "time": _json_value(row.get("time_begin")),
            "address": _json_value(row.get("building_address")),
            "lon": _json_value(row.get("lon")),
            "lat": _json_value(row.get("lat")),
        }

    def _snapshot_for_point(
        self, point: dict[str, Any], events: list | None = None
    ) -> dict[str, Any]:
        vehicle_id = int(point["tr_id"])
        forecast_time = pd.Timestamp(point["T"])
        target_id = int(point["target_stop_id"])
        target_time = pd.Timestamp(point["target_time_begin"])
        cur_dev_raw = point.get("cur_dev_s")
        current_deviation = _finite_float(cur_dev_raw) or 0.0
        cur_dev_hint = "none" if cur_dev_raw is None else "input"
        history = self._causal_history(vehicle_id, forecast_time)
        position = _latest_position(history)
        last_time = pd.Timestamp(history[-1]["event_time"]) if history else None
        last_age = float((forecast_time - last_time).total_seconds()) if last_time is not None else None
        window_start = forecast_time - pd.Timedelta(seconds=600)
        window_speeds = [
            speed
            for row in history
            if pd.Timestamp(row.get("event_time")) >= window_start
            for speed in [_finite_float(row.get("speed"))]
            if speed is not None
        ]
        average_speed = float(sum(window_speeds) / len(window_speeds)) if window_speeds else None
        schedule = self._vehicle_schedule(vehicle_id)
        target_row = schedule[pd.to_numeric(schedule["tt_action_item_id"], errors="coerce") == target_id]
        target = self._stop_card(target_row.iloc[0] if not target_row.empty else None)
        target_lon = _finite_float(target.get("lon")) if target else None
        target_lat = _finite_float(target.get("lat")) if target else None
        distance = None
        if position and target_lon is not None and target_lat is not None:
            distance = float(haversine_km([position["lon"]], [position["lat"]], target_lon, target_lat)[0])
        remaining = schedule[(schedule["time_begin"] > forecast_time) & (schedule["time_begin"] <= target_time)]
        gaps = pd.to_numeric(remaining["time_begin"].diff().dt.total_seconds(), errors="coerce").dropna()
        previous = schedule[schedule["time_begin"] <= forecast_time].tail(1)
        following = schedule[schedule["time_begin"] > forecast_time].head(1)
        target_previous = schedule[schedule["time_begin"] < target_time].tail(1)
        # Cause comes from the measured pattern detectors; the hand-rolled
        # speed/deviation heuristics stay only as a fallback for rows the
        # detectors could not score (no history at all).
        cause, cause_role = cause_from_events(events)
        if cause_role == "none":
            if not history:
                cause = "телеметрия отсутствует: прогноз опирается на плановое расписание и последнее известное отклонение"
            elif position is None:
                cause = "нет достоверных координат: траекторию и причину по движению определить нельзя"
        return {
            "forecast_time": _json_value(forecast_time),
            "target": target,
            "target_time": _json_value(target_time),
            "current_deviation_s": current_deviation,
            "position": position,
            "last_event_age_s": last_age,
            "telemetry_points": float(len(history)),
            "average_speed_10m_kmh": average_speed,
            "distance_to_target_km": distance,
            "planned_stops_remaining": len(remaining),
            "remaining_gap_mean_s": float(gaps.mean()) if not gaps.empty else None,
            "current_segment": {
                "previous": self._stop_card(previous.iloc[0] if not previous.empty else None),
                "next": self._stop_card(following.iloc[0] if not following.empty else None),
            },
            "target_segment": {
                "previous": self._stop_card(target_previous.iloc[0] if not target_previous.empty else None),
                "target": target,
            },
            "cause": cause,
            "cause_role": cause_role,
            "pattern_events": events or [],
            "cur_dev_hint": cur_dev_hint,
        }

    def _select_incident(self) -> dict[str, Any] | None:
        candidates = [
            record
            for record in self._vehicles.values()
            if isinstance(record.get("snapshot"), dict) and float(record.get("prediction", 0.0)) >= 60.0
        ]
        if not candidates:
            return None
        record = max(candidates, key=lambda item: float(item.get("prediction", 0.0)))
        snapshot = record["snapshot"]
        vehicle_id = int(record["tr_id"])
        target_time = pd.Timestamp(record["target_time_begin"])
        prediction = float(record["prediction"])
        arrival = target_time + pd.Timedelta(seconds=prediction) if pd.notna(target_time) else None
        return {
            "tr_id": vehicle_id,
            "sample_id": record.get("sample_id"),
            "forecast_time": record.get("T"),
            "target_stop_id": record.get("target_stop_id"),
            "target_time": record.get("target_time_begin"),
            "predicted_delay_s": prediction,
            "predicted_arrival": _json_value(arrival),
            "risk": record.get("risk"),
            "recommendation": record.get("recommendation"),
            "cause": snapshot.get("cause"),
            "cause_role": snapshot.get("cause_role"),
            "p_late": record.get("p_late"),
            "cur_dev_hint": snapshot.get("cur_dev_hint"),
            "position_now": _latest_position(self._vehicle_telemetry.get(vehicle_id, [])),
            "snapshot": snapshot,
        }

    def advance_to(self, target_time: pd.Timestamp) -> None:
        target = pd.Timestamp(target_time)
        while self._traffic_position < len(self._traffic_rows):
            event_time = pd.Timestamp(self._traffic_rows[self._traffic_position]["event_time"])
            if event_time > target:
                break
            self._remember_telemetry(self._traffic_rows[self._traffic_position])
            self._traffic_position += 1
        while self._point_position < len(self._point_rows):
            point = self._point_rows[self._point_position]
            point_time = pd.Timestamp(point["T"])
            if point_time > target:
                break
            result = self.service.predict_frame([point], pd.DataFrame(self._visible_traffic))
            if not result.empty:
                record = _record_for_response(result.iloc[0])
                events = record.get("pattern_events") or []
                record["snapshot"] = self._snapshot_for_point(point, events)
                self._vehicles[int(record["tr_id"])] = record
            self._point_position += 1
        if self._traffic_position >= len(self._traffic_rows) and self._point_position >= len(self._point_rows):
            self.current_time = target
            self.finished = True
        else:
            self.current_time = target

    def status(self) -> dict[str, Any]:
        vehicles = [self._vehicles[key] for key in sorted(self._vehicles)]
        next_traffic = None
        if self._traffic_position < len(self._traffic_rows):
            next_traffic = _json_value(self._traffic_rows[self._traffic_position]["event_time"])
        next_point = None
        if self._point_position < len(self._point_rows):
            next_point = _json_value(self._point_rows[self._point_position]["T"])
        active_routes = []
        active_positions = []
        for vehicle_id in sorted(self._vehicles):
            route = self._routes.get(vehicle_id)
            risk = str(self._vehicles[vehicle_id].get("risk", "on-time"))
            if route is not None:
                polyline = [
                    [stop["lat"], stop["lon"]]
                    for stop in route["stops"]
                    if isinstance(stop.get("lat"), (int, float)) and isinstance(stop.get("lon"), (int, float))
                ]
                active_routes.append(
                    {
                        "tr_id": vehicle_id,
                        "risk": risk,
                        "color": RISK_COLORS.get(risk, "#9ca3af"),
                        "stops": route["stops"],
                        "polyline": polyline,
                    }
                )
            position = _latest_position(self._vehicle_telemetry.get(vehicle_id, []))
            if position is not None:
                active_positions.append({"tr_id": vehicle_id, "risk": risk, **position})
        return {
            "simulated_time": _json_value(self.current_time),
            "speed": self.speed,
            "finished": self.finished,
            "processed_traffic_rows": self._traffic_position,
            "total_traffic_rows": len(self._traffic_rows),
            "processed_points": self._point_position,
            "total_points": len(self._point_rows),
            "next_traffic_time": next_traffic,
            "next_point_time": next_point,
            "vehicles": vehicles,
            "predictions": vehicles,
            "map": {
                "routes": active_routes,
                "positions": active_positions,
                "incident": self._select_incident(),
            },
        }
