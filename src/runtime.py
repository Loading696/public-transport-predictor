from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.predictor import build_features_from_frames
from src.predictor import predict as model_predict

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

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, low_memory=False)

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

    def _reset_state(self) -> None:
        self._traffic_position = 0
        self._point_position = 0
        self._visible_traffic = []
        self._vehicles = {}
        point_times = [value for value in self.points["T_dt"].tolist() if pd.notna(value)]
        traffic_times = [value for value in self.traffic["event_time"].tolist() if pd.notna(value)]
        candidates = point_times or traffic_times
        self.current_time = min(candidates) if candidates else pd.Timestamp.now()
        while self._traffic_position < len(self._traffic_rows):
            event_time = pd.Timestamp(self._traffic_rows[self._traffic_position]["event_time"])
            if event_time > self.current_time:
                break
            self._visible_traffic.append(self._traffic_rows[self._traffic_position])
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

    def advance_to(self, target_time: pd.Timestamp) -> None:
        target = pd.Timestamp(target_time)
        while self._traffic_position < len(self._traffic_rows):
            event_time = pd.Timestamp(self._traffic_rows[self._traffic_position]["event_time"])
            if event_time > target:
                break
            self._visible_traffic.append(self._traffic_rows[self._traffic_position])
            self._traffic_position += 1
        while self._point_position < len(self._point_rows):
            point = self._point_rows[self._point_position]
            point_time = pd.Timestamp(point["T"])
            if point_time > target:
                break
            result = self.service.predict_frame([point], pd.DataFrame(self._visible_traffic))
            if not result.empty:
                record = _record_for_response(result.iloc[0])
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
        }
