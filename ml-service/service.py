"""ML service: feature engineering, model loading and CatBoost inference.

This is a standalone FastAPI application and the only place in the system that
touches CatBoost.  It owns:

* loading ``ml/model.cbm``, the 141-name feature list and the calibration curve;
* the planned-timetable index and the telemetry normaliser;
* building causal features for a batch of forecast points;
* the forward pass, plus the derived risk / class / recommendation / calibrated
  probability and the pattern detectors run on the built features.

The backend does **not** import any of this.  It talks to this service over HTTP:

    POST /predict   {"points": [...], "telemetry": [...] | null}
                 -> {"count": n, "predictions": [record, ...]}

A single batch endpoint is deliberate.  Feature building has a large fixed cost
per call (schedule merge, telemetry slice, frame assembly) that is almost
independent of the row count, so batching amortises it; the backend keeps its
micro-batcher and calls this once per coalesced group.  That is why there is no
separate single-prediction endpoint -- a one-row batch is the same code path.

Everything is a pure function of data available at or before each point's ``T``:
``time_fact_begin`` is never read, and the telemetry cut is applied per point
inside the feature builder.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from bisect import bisect_right
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from catboost import CatBoostRegressor

from src.predictor import build_features_from_frames, predict as model_predict
from src.predictor import select_feature_columns
from src.patterns import detect_all
from src.schedule_index import schedule_index

SERVICE_NAME = "predictor-ml"
SERVICE_VERSION = "2.0.0"


# --------------------------------------------------------------------------- #
# Canonical shapes (kept identical to the backend's, see src/runtime.py)
# --------------------------------------------------------------------------- #

TELEMETRY_COLUMNS = [
    "tr_id", "event_time", "location_valid", "lon", "lat", "alt", "speed", "heading",
]
POINT_COLUMNS = [
    "sample_id", "tr_id", "T", "target_stop_id", "target_time_begin", "cur_dev_s",
]

#: Hard ceiling on rows per call, so one oversized request cannot pin a worker.
MAX_BATCH_ROWS = int(os.getenv("ML_MAX_BATCH_ROWS", "512"))


def _empty_traffic() -> pd.DataFrame:
    return pd.DataFrame(columns=TELEMETRY_COLUMNS)


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Parse a timestamp column that may arrive as strings of mixed shape.

    The naive ``pd.to_datetime(series, errors="coerce")`` infers one format from
    the first element and applies it to the whole column, so a column that mixes
    ``2026-01-06T03:35:00`` with ``2026-01-06T03:35:00.462764`` yields ``NaT``
    for the fractional ones -- and the caller then drops them, silently changing
    every windowed feature.  In the validate telemetry 6 547 of 105 945 rows
    carry sub-second precision, so this was not a corner case: it made a live
    batch prediction differ from the same prediction computed locally.

    ``format="ISO8601"`` accepts both the ``T`` and space separators and the
    optional fraction, and parses the whole column in one pass; per-element
    ``format="mixed"`` is the fallback for anything it still cannot read.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    parsed = pd.to_datetime(series, errors="coerce", format="ISO8601")
    # Anything ISO8601 could not read (a legacy format, say) is retried
    # element by element rather than being silently dropped.  Note the dtype
    # check must not be ``object``: pandas 3 stores text in a dedicated string
    # dtype, so that test silently skipped the robust path.
    unresolved = parsed.isna() & series.notna()
    if bool(unresolved.any()):
        fallback = pd.to_datetime(series, errors="coerce", format="mixed")
        parsed = parsed.fillna(fallback)
    return parsed


def _ensure_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def normalise_telemetry(
    traffic: pd.DataFrame | list[dict[str, Any]] | None,
) -> pd.DataFrame:
    """Coerce arbitrary telemetry into the canonical, causal-ready frame."""
    if traffic is None:
        return _empty_traffic()
    frame = traffic.copy() if isinstance(traffic, pd.DataFrame) else pd.DataFrame(list(traffic))
    if frame.empty:
        return _empty_traffic()
    for column in TELEMETRY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["tr_id"] = _ensure_numeric(frame["tr_id"])
    frame = frame.dropna(subset=["tr_id"])
    frame["tr_id"] = frame["tr_id"].astype("int64")
    frame["event_time"] = _ensure_datetime(frame["event_time"])
    frame = frame.dropna(subset=["event_time"])
    for column in ("speed", "lon", "lat", "alt", "heading"):
        frame[column] = _ensure_numeric(frame[column])
    if frame["location_valid"].dtype == object:
        frame["location_valid"] = frame["location_valid"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    else:
        frame["location_valid"] = frame["location_valid"].fillna(False).astype(bool)
    return frame[TELEMETRY_COLUMNS].sort_values("event_time", kind="stable").reset_index(drop=True)


def normalise_points(points: Any) -> pd.DataFrame:
    """Coerce forecast points to the canonical frame the feature builder wants."""
    frame = points.copy() if isinstance(points, pd.DataFrame) else pd.DataFrame(list(points or []))
    if frame.empty:
        raise ValueError("points must not be empty")
    for column in POINT_COLUMNS:
        if column not in frame.columns:
            frame[column] = "" if column == "sample_id" else np.nan
    frame = frame.copy()
    frame["tr_id"] = pd.to_numeric(frame["tr_id"], errors="raise").astype("int64")
    frame["T"] = pd.to_datetime(frame["T"], errors="raise")
    frame["target_stop_id"] = pd.to_numeric(frame["target_stop_id"], errors="coerce")
    frame["cur_dev_s"] = pd.to_numeric(frame["cur_dev_s"], errors="raise").astype(float)
    frame["sample_id"] = frame["sample_id"].astype(str)
    generated = frame["sample_id"].isna() | frame["sample_id"].astype(str).eq("")
    if generated.any():
        frame.loc[generated, "sample_id"] = [f"ml_{i}" for i in range(1, int(generated.sum()) + 1)]
    frame["sample_id"] = frame["sample_id"].astype(str)
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Post-processing: identical semantics to the previous in-backend code
# --------------------------------------------------------------------------- #


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


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #


class ModelNotReady(RuntimeError):
    """Raised when inference is requested before the model finished loading."""


class MlEngine:
    """Owns the model, the schedule index and the inference pipeline.

    Loading takes a few seconds (schedule parse + CatBoost deserialise), so it
    runs in a background thread and every endpoint reports ``loading`` until it
    finishes.  Inference is serialised behind one lock: CatBoost prediction is
    fast and GIL-bound anyway, and a single lock makes the concurrency story
    trivial instead of a source of subtle races.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.model_path = self._path(os.getenv("MODEL_PATH", "ml/model.cbm"))
        self.features_path = self._path(os.getenv("FEATURES_PATH", "ml/features.json"))
        self.schedule_path = self._path(
            os.getenv("SCHEDULE_PATH", os.path.join("dataset", "validate", "schedule_plan.csv"))
        )
        # The calibration curve is a sibling of the model, so anchor it to the
        # model directory rather than to a project root: the two layouts differ
        # (ml-service/service.py in the repo, /app/service.py in the image) and
        # guessing the root silently disabled calibration in the container.
        self.prob_cal_path = self._path(
            os.getenv("PROB_CAL_PATH", str(self.model_path.parent / "prob_cal.json"))
        )
        self._lock = threading.Lock()
        self._ready = False
        self._error: str | None = None
        self._model: CatBoostRegressor | None = None
        self._features: list[str] = []
        self._schedule: pd.DataFrame | None = None
        self._schedule_index = None
        self._prob_cal: tuple[list[float], list[float]] | None = None
        self._load_seconds: float | None = None
        self.calls = 0
        self.rows = 0
        self.errors = 0
        self.thread = threading.Thread(target=self._load_guarded, daemon=True)
        self.thread.start()

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    # -- loading ---------------------------------------------------------- #

    def _load_guarded(self) -> None:
        try:
            self.load()
            self._ready = True
        except Exception as exc:  # noqa: BLE001 - surfaced through /health
            self._error = f"{type(exc).__name__}: {exc}"
            self._ready = False

    def load(self) -> None:
        started = time.perf_counter()
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")
        model = CatBoostRegressor()
        model.load_model(str(self.model_path))
        features = json.loads(self.features_path.read_text(encoding="utf-8"))
        if not isinstance(features, list) or not features:
            raise ValueError(f"invalid feature list: {self.features_path}")
        if not self.schedule_path.exists():
            raise FileNotFoundError(f"schedule not found: {self.schedule_path}")
        schedule = pd.read_csv(self.schedule_path, low_memory=False)
        with self._lock:
            self._model = model
            self._features = [str(name) for name in features]
            self._schedule = schedule
            self._schedule_index = schedule_index(schedule)
            self._prob_cal = self._load_prob_cal(self.prob_cal_path)
            self._load_seconds = time.perf_counter() - started

    @staticmethod
    def _load_prob_cal(path: Path) -> tuple[list[float], list[float]] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            xs = [float(v) for v in payload.get("X", [])]
            ps = [float(v) for v in payload.get("p", [])]
        except (OSError, ValueError, TypeError, AttributeError):
            return None
        if len(xs) != len(ps) or not xs:
            return None
        return xs, ps

    # -- calibration ------------------------------------------------------ #

    def p_late_for(self, delay_s: float) -> float | None:
        if self._prob_cal is None:
            return None
        value = _finite_float(delay_s)
        if value is None:
            return None
        xs, ps = self._prob_cal
        idx = max(0, min(bisect_right(xs, value) - 1, len(ps) - 1))
        return float(ps[idx])

    # -- inference -------------------------------------------------------- #

    def require_ready(self) -> None:
        if not self._ready:
            raise ModelNotReady(self._error or "model is still loading")

    def predict(
        self,
        points: Any,
        telemetry: Any = None,
    ) -> list[dict[str, Any]]:
        """Run the full pipeline for a batch of points.

        Returns one record per input point, in input order.  Raises
        :class:`ValueError` on malformed input and :class:`ModelNotReady` while
        the model is still loading.
        """
        self.require_ready()
        point_frame = normalise_points(points)
        if len(point_frame) > MAX_BATCH_ROWS:
            raise ValueError(
                f"batch too large: {len(point_frame)} rows, limit {MAX_BATCH_ROWS}"
            )
        telemetry_frame = (
            normalise_telemetry(telemetry) if telemetry is not None else _empty_traffic()
        )
        # Coarse pre-trim to the largest T in the batch.  The authoritative
        # per-point cut happens inside the feature builder.
        max_time = point_frame["T"].max()
        ids = set(point_frame["tr_id"].tolist())
        causal = telemetry_frame[
            telemetry_frame["tr_id"].isin(ids) & (telemetry_frame["event_time"] <= max_time)
        ].copy()

        with self._lock:
            model, features, index = self._model, self._features, self._schedule_index
        frame = build_features_from_frames(point_frame, causal, self._schedule, index)
        values = np.asarray(model_predict(model, frame, features), dtype=float)
        if len(values) != len(point_frame) or not np.isfinite(values).all():
            raise ValueError("model returned invalid predictions")

        records: list[dict[str, Any]] = []
        for pos in range(len(point_frame)):
            row = frame.iloc[pos]
            try:
                events = detect_all(row.to_dict())
            except Exception:  # noqa: BLE001 - detectors must stay total
                events = []
            delay = float(values[pos])
            records.append(
                {
                    "sample_id": str(point_frame["sample_id"].iloc[pos]),
                    "tr_id": int(point_frame["tr_id"].iloc[pos]),
                    "T": _json_value(point_frame["T"].iloc[pos]),
                    "target_stop_id": _json_value(point_frame["target_stop_id"].iloc[pos]),
                    "target_time_begin": _json_value(
                        point_frame["target_time_begin"].iloc[pos]
                    ),
                    "cur_dev_s": float(point_frame["cur_dev_s"].iloc[pos]),
                    "prediction": delay,
                    "p_late": self.p_late_for(delay),
                    "target_class": target_class_for(delay),
                    "risk": risk_for(delay),
                    "recommendation": recommendation_for(delay),
                    "pattern_events": events,
                }
            )
        self.calls += 1
        self.rows += len(records)
        return records

    # -- introspection ---------------------------------------------------- #

    def health(self) -> dict[str, Any]:
        with self._lock:
            features = len(self._features)
            stops = 0 if self._schedule is None else int(len(self._schedule))
            vehicles = 0 if self._schedule_index is None else len(self._schedule_index.by_vehicle)
        return {
            "status": "ok" if self._ready else ("error" if self._error else "loading"),
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "model": str(self.model_path),
            "model_loaded": self._ready,
            "error": self._error,
            "features": features,
            "schedule_stops": stops,
            "schedule_vehicles": vehicles,
            "calibrated": self._prob_cal is not None,
            "load_seconds": None if self._load_seconds is None else round(self._load_seconds, 3),
            "calls": self.calls,
            "rows": self.rows,
            "errors": self.errors,
        }


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #


class PredictPoint(BaseModel):
    sample_id: str | None = Field(default=None, description="ID точки; генерируется, если не передан")
    tr_id: int = Field(description="ID транспортного средства")
    T: str = Field(description="Момент прогноза — граница причинного среза")
    target_stop_id: int | None = Field(default=None, description="Целевая остановка")
    target_time_begin: str | None = Field(default=None, description="Плановое прибытие на цель")
    cur_dev_s: float = Field(description="Текущее отклонение от расписания, секунды")


class TelemetryEvent(BaseModel):
    tr_id: int
    event_time: str
    location_valid: bool = True
    lon: float | None = None
    lat: float | None = None
    alt: float | None = None
    speed: float | None = None
    heading: float | None = None


class PredictRequest(BaseModel):
    points: list[PredictPoint] = Field(min_length=1, description="Точки прогноза")
    telemetry: list[TelemetryEvent] | None = Field(
        default=None, description="Телеметрия, общая для всех точек пакета"
    )


class PatternEvent(BaseModel):
    type: str
    role: str
    confidence: float
    reason: str


class Prediction(BaseModel):
    sample_id: str
    tr_id: int
    T: str | None = None
    target_stop_id: int | None = None
    target_time_begin: str | None = None
    cur_dev_s: float
    prediction: float
    p_late: float | None = None
    target_class: str
    risk: str
    recommendation: str
    pattern_events: list[PatternEvent] = Field(default_factory=list)


class PredictResponse(BaseModel):
    count: int
    rows_per_call: float
    predict_ms: float
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model: str
    model_loaded: bool
    error: str | None = None
    features: int
    schedule_stops: int
    schedule_vehicles: int
    calibrated: bool
    load_seconds: float | None = None
    calls: int
    rows: int
    errors: int


def _default_root() -> Path:
    """Best guess at the project root.

    The service runs from two places: ``ml-service/service.py`` in a checkout
    (root is one level up) and ``/app/service.py`` in the image (root is the file's
    own directory).  Rather than hard-coding either, pick the first candidate that
    actually looks like the project, and allow ``ML_ROOT`` to override.
    """
    override = os.getenv("ML_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for candidate in candidates:
        if (candidate / "dataset" / "validate").is_dir() or (candidate / "ml").is_dir():
            return candidate
    return here.parent


def _engine() -> MlEngine:
    return app.state.engine


def create_app(root: str | Path | None = None) -> FastAPI:
    application = FastAPI(
        title="Предиктор задержек — ML service",
        version=SERVICE_VERSION,
        description=(
            "Feature engineering, CatBoost inference and pattern detection. "
            "Вызывается из backend по HTTP; напрямую в backend не импортируется."
        ),
    )
    project_root = Path(root) if root is not None else _default_root()
    application.state.engine = MlEngine(project_root)

    @application.get("/health", response_model=HealthResponse, tags=["service"])
    async def health() -> dict[str, Any]:
        return _engine().health()

    @application.get("/models", tags=["service"])
    async def models() -> dict[str, Any]:
        engine = _engine()
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "model": str(engine.model_path),
            "features": list(engine._features),
            "feature_count": len(engine._features),
            "calibrated": engine._prob_cal is not None,
            "ready": engine._ready,
        }

    @application.post(
        "/predict",
        response_model=PredictResponse,
        summary="Пакетный прогноз (features + CatBoost + паттерны)",
        tags=["prediction"],
    )
    async def predict(payload: Annotated[PredictRequest, Body()]) -> Any:
        engine = _engine()
        started = time.perf_counter()
        try:
            records = engine.predict(
                [point.model_dump() for point in payload.points],
                [event.model_dump() for event in payload.telemetry] if payload.telemetry else None,
            )
        except ModelNotReady as exc:
            return JSONResponse(status_code=503, content={"detail": str(exc)})
        except (ValueError, TypeError, KeyError) as exc:
            engine.errors += 1
            return JSONResponse(status_code=422, content={"detail": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:  # noqa: BLE001 - never leak a stack trace as a 500
            engine.errors += 1
            return JSONResponse(
                status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"}
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "count": len(records),
            "rows_per_call": round(len(records) / max(elapsed_ms / 1000.0, 1e-9), 2),
            "predict_ms": round(elapsed_ms, 2),
            "predictions": records,
        }

    return application


app = create_app()


def main() -> None:  # pragma: no cover - container entry point
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("ML_HOST", "0.0.0.0"),
        port=int(os.getenv("ML_PORT", "8001")),
        log_level=os.getenv("ML_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
