from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from src.runtime import InferenceService, StreamSimulator
from src.ndtp_server import NDTPReceiver
from src.batching import MicroBatcher

DEFAULT_SPEED = float(os.getenv("SIMULATION_SPEED", "60"))
TCP_PORT = int(os.getenv("TCP_PORT", "9201"))
# Троттлинг live-прогнозов: пересчёт для юнита не чаще TTL, повторные опросы
# /stream/status без новых данных отдают кэшированный прогноз.
LIVE_PREDICT_TTL = float(os.getenv("LIVE_PREDICT_TTL", "10"))
# Окно сбора микро-батча: сколько ждать соседние запросы перед общим вызовом.
BATCH_WINDOW_MS = float(os.getenv("PREDICT_BATCH_WINDOW_MS", "4"))
# Потолок строк в одном батче, ограничивающий пиковую память фрейма признаков.
PREDICT_BATCH_MAX_ROWS = int(os.getenv("PREDICT_BATCH_MAX_ROWS", "256"))

app = FastAPI(title="Transport Delay MVP", version="1.1.0")

# Кэш прогнозов живого потока: (unit_id, event_time_iso) -> (mono_ts, forecast).
# forecast — только прогнозная часть юнита (позиция/двери всегда свежие из строк).
_LIVE_FORECAST_CACHE: dict[tuple[int, str], tuple[float, dict[str, Any]]] = {}


def _reset_live_forecast_cache() -> None:
    """Сброс кэша live-прогнозов (для тестов)."""
    _LIVE_FORECAST_CACHE.clear()


# Прогнозная часть юнита, покрытая кэшем (позиция/двери/возраст всегда свежие).
_FORECAST_KEYS = (
    "target_stop_id",
    "target_time_begin",
    "target_status",
    "target_note",
    "cur_dev_hint",
    "prediction",
    "risk",
    "target_class",
    "recommendation",
    "p_late",
    "pattern_events",
)


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #


class Metrics:
    """Lightweight request/inference accounting, no external dependency.

    Latency is kept as a count/sum/max triple plus a coarse histogram rather
    than every sample: a bounded structure can never grow with traffic, and the
    median is still recoverable from the buckets.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.started_at = time.time()
        self.counts: dict[str, int] = defaultdict(int)
        self.errors: dict[str, int] = defaultdict(int)
        self.total_s: dict[str, float] = defaultdict(float)
        self.max_s: dict[str, float] = defaultdict(float)
        self.histogram: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.inference = {"calls": 0, "rows": 0, "total_s": 0.0, "max_s": 0.0}

    def observe(self, name: str, seconds: float, *, error: bool = False) -> None:
        self.counts[name] += 1
        self.total_s[name] += seconds
        self.max_s[name] = max(self.max_s[name], seconds)
        self.histogram[name][_bucket(seconds)] += 1
        if error:
            self.errors[name] += 1

    def observe_inference(self, seconds: float, rows: int) -> None:
        self.inference["calls"] += 1
        self.inference["rows"] += rows
        self.inference["total_s"] += seconds
        self.inference["max_s"] = max(self.inference["max_s"], seconds)

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            endpoints = {}
            for name, count in self.counts.items():
                endpoints[name] = {
                    "calls": count,
                    "errors": self.errors.get(name, 0),
                    "mean_s": round(self.total_s[name] / count, 4) if count else 0.0,
                    "max_s": round(self.max_s[name], 4),
                    "histogram": dict(sorted(self.histogram[name].items())),
                }
            inference = dict(self.inference)
            inference["mean_s"] = (
                round(inference["total_s"] / inference["calls"], 4) if inference["calls"] else 0.0
            )
            inference["mean_rows"] = (
                round(inference["rows"] / inference["calls"], 2) if inference["calls"] else 0.0
            )
            inference["rows_per_s"] = (
                round(inference["rows"] / inference["total_s"], 1) if inference["total_s"] else 0.0
            )
            return {
                "uptime_s": round(time.time() - self.started_at, 1),
                "endpoints": endpoints,
                "inference": inference,
            }


METRICS = Metrics()

_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def _bucket(seconds: float) -> str:
    for edge in _BUCKETS:
        if seconds < edge:
            return f"<{round(edge * 1000)}ms"
    return ">=5000ms"


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Time every request, tag the response, and count failures.

    Executed on the event loop but does no I/O of its own, so it costs
    microseconds.  Inference time is recorded separately by the service, which
    knows how much of the request was the model.
    """
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        METRICS.observe(request.url.path, time.perf_counter() - started, error=True)
        raise
    elapsed = time.perf_counter() - started
    METRICS.observe(request.url.path, elapsed, error=response.status_code >= 400)
    response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.1f}"
    return response


# --------------------------------------------------------------------------- #
# Схемы запросов
# --------------------------------------------------------------------------- #


class TelemetryEvent(BaseModel):
    tr_id: int
    event_time: datetime
    location_valid: bool = True
    lon: float | None = None
    lat: float | None = None
    alt: float | None = None
    speed: float | None = None
    heading: float | None = None

    class Config:
        extra = "ignore"


class PredictPoint(BaseModel):
    sample_id: str | None = None
    tr_id: int
    T: datetime
    target_stop_id: int
    target_time_begin: datetime
    cur_dev_s: float
    telemetry: list[TelemetryEvent] = Field(default_factory=list)

    class Config:
        extra = "ignore"


class PredictRequest(BaseModel):
    points: list[PredictPoint] = Field(min_length=1)
    telemetry: list[TelemetryEvent] = Field(default_factory=list)

    class Config:
        extra = "ignore"


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


# --------------------------------------------------------------------------- #
# Зависимости
# --------------------------------------------------------------------------- #


def _get_runtime() -> InferenceService:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        runtime = InferenceService()
        app.state.runtime = runtime
    return runtime


async def _get_batcher() -> MicroBatcher:
    batcher = getattr(app.state, "batcher", None)
    if batcher is None:
        batcher = MicroBatcher(
            _get_runtime(),
            max_batch_rows=PREDICT_BATCH_MAX_ROWS,
            window_ms=BATCH_WINDOW_MS,
        )
        app.state.batcher = batcher
    if batcher._task is None or batcher._task.done():
        await batcher.start()
    return batcher


async def _get_simulator() -> StreamSimulator:
    simulator = getattr(app.state, "simulator", None)
    if simulator is None:
        simulator = _get_runtime().stream_simulator(DEFAULT_SPEED)
        app.state.simulator = simulator
    if simulator._task is None or simulator._task.done():
        await simulator.start()
    return simulator


def _parse_request(payload: Any) -> PredictRequest:
    if isinstance(payload, list):
        payload = {"points": payload}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="request body must be an object or a list")
    try:
        if hasattr(PredictRequest, "model_validate"):
            return PredictRequest.model_validate(payload)
        return PredictRequest.parse_obj(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


KNOWN_ROUTES = [
    "/",
    "/docs",
    "/health",
    "/predict",
    "/generate_submission",
    "/stream/status",
    "/models",
    "/v1/models",
    "/ndtp/status",
    "/cascade",
    "/metrics",
]


def _get_ndtp() -> NDTPReceiver | None:
    return getattr(app.state, "ndtp", None)


# --------------------------------------------------------------------------- #
# Live-агрегация
# --------------------------------------------------------------------------- #


async def _live_units() -> list[dict[str, Any]]:
    """Latest position per connected NDTP unit (live emulator feed).

    Каждый юнит обогащается:
      - tr_id через UNIT_TR_MAP (env JSON {"unitId": tr_id}),
      - door_open из ячеек Crown03/Irma04 (True/False/None),
      - целью = первая плановая остановка из расписания по tr_id
        в окне (T+600, T+900]; вне окна — prediction=None + target_note,
      - прогнозом задержки через InferenceService.predict_records.

    Честность live-прогноза: фактического отклонения cur_dev_s для живых
    юнитов нет (нет time_fact), поэтому в точку подставляется нейтральный
    плейсхолдер 0.0 — модель тяжело опирается на эту подсказку и тянет
    прогноз к «по графику». Это задекларировано меткой cur_dev_hint="none"
    рядом с прогнозом (бейдж «без подсказки» в дашборде), а не молча.

    Троттлинг: прогноз кэшируется по (unit_id, event_time) на LIVE_PREDICT_TTL
    секунд — повторные опросы без новых данных пересчёта не вызывают
    (forecast_cached=True).
    """
    receiver = _get_ndtp()
    if receiver is None:
        return []
    runtime = _get_runtime()
    mapping = _unit_tr_map()
    try:
        receiver.set_tr_map(mapping)
    except AttributeError:
        pass
    rows = await receiver.snapshot_rows()
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            latest[int(row.get("unit_id", -1))] = row
        except (TypeError, ValueError):
            continue
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    units: list[dict[str, Any]] = []
    for unit_id in sorted(latest):
        row = latest[unit_id]
        event_time = row.get("event_time")
        age = (now - event_time).total_seconds() if event_time is not None else None
        tr_id = mapping.get(unit_id, row.get("tr_id", unit_id))
        try:
            tr_id = int(tr_id)
        except (TypeError, ValueError):
            tr_id = int(unit_id)
        door = row.get("door_open")
        units.append(
            {
                "unit_id": unit_id,
                "tr_id": tr_id,
                "mapped": unit_id in mapping,
                "lon": row.get("lon"),
                "lat": row.get("lat"),
                "speed": row.get("speed"),
                "heading": row.get("heading"),
                "location_valid": bool(row.get("location_valid", False)),
                "event_time": event_time.isoformat() if hasattr(event_time, "isoformat") else event_time,
                "_event_time": event_time,
                "age_s": age,
                "door_open": bool(door) if door is not None else None,
                "source": "live",
            }
        )
    try:
        batcher = await _get_batcher()
    except Exception:
        batcher = None
    if batcher is None:
        for unit in units:
            unit.pop("_event_time", None)
            unit.update(_no_model_forecast())
        return units
    try:
        points: list[dict[str, Any]] = []
        point_unit_idx: list[int] = []
        cached_idx: set[int] = set()
        telemetry: list[dict[str, Any]] = []
        now_mono = time.monotonic()
        for idx, unit in enumerate(units):
            event_time = unit.pop("_event_time", None)
            event_key = unit.get("event_time") or ""
            unit["forecast_cached"] = False
            if event_key:
                hit = _LIVE_FORECAST_CACHE.get((unit["unit_id"], event_key))
                if hit is not None and now_mono - hit[0] <= LIVE_PREDICT_TTL:
                    # Новых данных нет и кэш свежий — пересчёт не нужен.
                    unit.update(hit[1])
                    unit["forecast_cached"] = True
                    cached_idx.add(idx)
                    continue
            target_id, target_time, target_status = runtime.live_target(unit["tr_id"], event_time)
            unit["target_stop_id"] = target_id
            unit["target_time_begin"] = target_time.isoformat() if hasattr(target_time, "isoformat") else target_time
            unit["_target_time"] = target_time
            unit["target_status"] = target_status
            unit["target_note"] = _target_note(target_status)
            unit["cur_dev_hint"] = None
            if target_status != "ok":
                unit.update({"prediction": None, "risk": None, "target_class": None, "recommendation": None})
                continue
            points.append(
                {
                    "tr_id": unit["tr_id"],
                    "T": unit["event_time"],
                    "target_stop_id": target_id,
                    "target_time_begin": unit["target_time_begin"],
                    # ВНИМАНИЕ: фактического отклонения для live-юнитов нет.
                    # 0.0 — нейтральный плейсхолдер, а не измерение: модель
                    # опирается на cur_dev_s и смещает прогноз к «по графику».
                    # Метка cur_dev_hint="none" + бейдж «без подсказки» в UI
                    # делают это явным рядом с каждым live-прогнозом.
                    "cur_dev_s": 0.0,
                }
            )
            unit["cur_dev_hint"] = "none"
            point_unit_idx.append(idx)
        for row in rows:
            try:
                uid = int(row.get("unit_id", -1))
            except (TypeError, ValueError):
                continue
            event_time = row.get("event_time")
            telemetry.append(
                {
                    "tr_id": mapping.get(uid, row.get("tr_id", uid)),
                    "event_time": event_time.isoformat() if hasattr(event_time, "isoformat") else event_time,
                    "location_valid": bool(row.get("location_valid", False)),
                    "lon": row.get("lon"),
                    "lat": row.get("lat"),
                    "alt": row.get("alt"),
                    "speed": row.get("speed"),
                    "heading": row.get("heading"),
                }
            )
        # Привязка прогнозов — строго по индексу точки, а не по tr_id:
        # несколько юнитов могут маппиться на один tr_id, а юниты вне
        # горизонта точек не строят и чужой прогноз получать не должны.
        predicted: dict[int, dict[str, Any]] = {}
        if points:
            started = time.perf_counter()
            result = await batcher.predict_records(points, telemetry or None)
            METRICS.observe_inference(time.perf_counter() - started, len(points))
            for pos, record in zip(point_unit_idx, result):
                predicted[pos] = record
        for idx, unit in enumerate(units):
            unit.pop("_target_time", None)
            if idx in cached_idx:
                continue
            record = predicted.get(idx)
            if record is None:
                if unit.get("target_status") != "ok":
                    unit.update({"prediction": None, "risk": None, "target_class": None, "recommendation": None})
                else:
                    unit.update(
                        {
                            "prediction": None,
                            "risk": "unknown",
                            "target_class": None,
                            "recommendation": "прогноз недоступен",
                        }
                    )
            else:
                unit.update(
                    {
                        "prediction": record.get("prediction"),
                        "risk": record.get("risk"),
                        "target_class": record.get("target_class"),
                        "recommendation": record.get("recommendation"),
                    }
                )
                # Passthrough полей Person A (p_late/pattern_events): сегодня
                # runtime их не отдаёт — ключи просто отсутствуют; после мержа
                # prob-wire подхватятся автоматически, без правок здесь.
                if "p_late" in record:
                    unit["p_late"] = record["p_late"]
                if "pattern_events" in record:
                    unit["pattern_events"] = record["pattern_events"]
                event_key = unit.get("event_time") or ""
                if event_key:
                    # Evict stale entries of the same unit: cache holds only
                    # the latest event_time, otherwise it grows forever.
                    for key in [k for k in _LIVE_FORECAST_CACHE if k[0] == unit["unit_id"]]:
                        del _LIVE_FORECAST_CACHE[key]
                    _LIVE_FORECAST_CACHE[(unit["unit_id"], event_key)] = (
                        now_mono, {key: unit.get(key) for key in _FORECAST_KEYS}
                    )
    except Exception:
        for unit in units:
            unit.pop("_event_time", None)
            unit.pop("_target_time", None)
            unit.setdefault("prediction", None)
            unit.setdefault("risk", None)
            unit.setdefault("target_stop_id", None)
            unit.setdefault("target_time_begin", None)
            unit.setdefault("target_status", "no_model")
            unit.setdefault("target_note", "прогноз недоступен")
            unit.setdefault("cur_dev_hint", None)
            unit.setdefault("forecast_cached", False)
    return units


def _no_model_forecast() -> dict[str, Any]:
    return {
        "target_stop_id": None,
        "target_time_begin": None,
        "target_status": "no_model",
        "target_note": "прогноз недоступен: нет модели/расписания",
        "cur_dev_hint": None,
        "prediction": None,
        "risk": None,
        "target_class": None,
        "recommendation": None,
        "forecast_cached": False,
    }


def _unit_tr_map() -> dict[int, int]:
    raw = os.getenv("UNIT_TR_MAP", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    mapping: dict[int, int] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            try:
                mapping[int(key)] = int(value)
            except (TypeError, ValueError):
                continue
    return mapping


def _target_note(status: str) -> str | None:
    """Человекочитаемая пометка статуса цели для витрины."""
    return {
        "ok": None,
        "no_schedule": "нет расписания для ТС",
        "no_time": "нет времени события",
        "out_of_horizon": "цель вне горизонта 10–15 мин",
        "no_model": "прогноз недоступен",
    }.get(status, "прогноз недоступен")


# --------------------------------------------------------------------------- #
# Обработчики
# --------------------------------------------------------------------------- #


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "detail": "not found",
            "path": request.url.path,
            "available": KNOWN_ROUTES,
        },
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/models")
def models() -> dict[str, Any]:
    runtime = _get_runtime()
    return {
        "model": str(runtime.model_path),
        "features": len(runtime.features),
        "validate_points": len(runtime.points),
    }


@app.get("/v1/models")
def v1_models() -> dict[str, Any]:
    info = models()
    return {
        "object": "list",
        "data": [
            {
                "id": Path(info["model"]).stem,
                "object": "model",
                "owned_by": "hackathon",
                "features": info["features"],
            }
        ],
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"service": app.title, "docs": "/docs"}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Healthcheck: must stay cheap, so nothing heavy is warmed up here."""
    try:
        runtime = await asyncio.to_thread(_get_runtime)
        return {
            "status": "ok",
            "model": str(runtime.model_path),
            "features": len(runtime.features),
            "validate_points": len(runtime.points),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict")
async def predict(payload: Annotated[Any, Body()]) -> dict[str, Any]:
    """Batch prediction endpoint.

    The request is parsed on the event loop (pure CPU, microseconds), then the
    actual inference is handed to the :class:`MicroBatcher`, which runs it in a
    worker thread and coalesces it with any concurrent request.  Nothing here
    blocks the loop.
    """
    request = _parse_request(payload)
    point_rows = [_dump(point) for point in request.points]
    telemetry_rows = [_dump(event) for event in request.telemetry]
    started = time.perf_counter()
    try:
        batcher = await _get_batcher()
        predictions = await batcher.predict_records(point_rows, telemetry_rows or None)
    except HTTPException:
        raise
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    METRICS.observe_inference(time.perf_counter() - started, len(point_rows))
    return {"count": len(predictions), "predictions": predictions}


@app.post("/generate_submission")
async def generate_submission() -> dict[str, Any]:
    batcher = await _get_batcher()
    started = time.perf_counter()
    try:
        result = await batcher.predict_records(
            _get_runtime().points.to_dict("records"),
            _get_runtime().traffic,
            bypass=True,
        )
        payload = _build_submission(result)
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    METRICS.observe_inference(time.perf_counter() - started, len(result))
    return payload


@app.get("/generate_submission")
async def generate_submission_get() -> dict[str, Any]:
    return await generate_submission()


def _build_submission(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate coverage and write submission.csv from already-computed records."""
    import pandas as pd

    runtime = _get_runtime()
    expected = runtime.points["sample_id"].astype(str)
    actual = pd.Series([str(record.get("sample_id")) for record in records], dtype=str)
    if len(records) != len(runtime.points):
        raise ValueError("submission coverage mismatch")
    if actual.duplicated().any() or not actual.equals(expected.reset_index(drop=True)):
        raise ValueError("submission sample_id coverage mismatch")
    values = [float(record.get("prediction")) for record in records]
    if any(not (value == value) or value in (float("inf"), float("-inf")) for value in values):
        raise ValueError("submission contains non-finite predictions")
    destination = runtime.root / "submission.csv"
    frame = pd.DataFrame({"sample_id": actual, "prediction": values})
    frame.to_csv(destination, index=False, sep=";", float_format="%.6f")
    return {
        "path": str(destination),
        "rows": len(frame),
        "first_rows": [
            {"sample_id": str(row["sample_id"]), "prediction": float(row["prediction"])}
            for _, row in frame.head(5).iterrows()
        ],
    }


@app.get("/stream/status")
async def stream_status(
    speed: float | None = Query(default=None, gt=0),
    reset: bool = Query(default=False),
) -> dict[str, Any]:
    """Replay dashboard feed.

    The simulator does its work in a background task; this handler only reads
    the already-computed state, so the loop is never held for more than the
    status assembly itself.
    """
    simulator = await _get_simulator()
    try:
        if reset:
            simulator.reset(speed)
        elif speed is not None:
            simulator.set_speed(speed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    data = simulator.status()
    data["live_units"] = await _live_units()
    data["batching"] = (await _get_batcher()).stats.as_dict()
    return data


@app.get("/ndtp/status")
async def ndtp_status() -> dict[str, Any]:
    receiver = _get_ndtp()
    if receiver is None:
        return {"enabled": False, "port": TCP_PORT}
    return receiver.status()


@app.get("/cascade")
async def cascade() -> dict[str, Any]:
    """Cascading-delay view of the planned route network.

    Returns the static graph description (vertex / edge counts, corridor pairs,
    active regime) plus the current solved view: which routes are late, which of
    them get dragged further behind by an intersecting route, and the per-route
    ETA projection with the stop at which the fleet claws the delay back.
    """
    runtime = await asyncio.to_thread(_get_runtime)
    engine = await asyncio.to_thread(runtime.cascade_engine)
    payload: dict[str, Any] = {"graph": engine.graph.network(), "view": None}
    simulator = getattr(app.state, "simulator", None)
    if simulator is not None:
        try:
            payload["view"] = simulator.cascade_view()
        except Exception as exc:  # noqa: BLE001 - never fail the whole response
            payload["view"] = {"enabled": False, "reason": str(exc)}
    return payload


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Runtime counters: per-endpoint latency, inference time, batching efficiency."""
    payload = await METRICS.snapshot()
    try:
        payload["batching"] = (await _get_batcher()).stats.as_dict()
    except Exception:  # noqa: BLE001
        payload["batching"] = None
    runtime = getattr(app.state, "runtime", None)
    if runtime is not None:
        payload["model"] = {
            "path": str(runtime.model_path),
            "features": len(runtime.features),
            "traffic_rows": int(len(runtime.traffic)),
            "schedule_rows": int(len(runtime.schedule)),
            "predict_points": int(len(runtime.points)),
        }
    return payload


# --------------------------------------------------------------------------- #
# Жизненный цикл
# --------------------------------------------------------------------------- #


@app.on_event("startup")
async def on_startup() -> None:
    await _get_batcher()
    await _get_simulator()
    try:
        receiver = NDTPReceiver(tr_map=_unit_tr_map())
        await receiver.start("0.0.0.0", TCP_PORT)
        app.state.ndtp = receiver
    except OSError as exc:
        app.state.ndtp = None
        print(f"NDTP TCP server not started on port {TCP_PORT}: {exc}", flush=True)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    with contextlib.suppress(Exception):
        batcher = getattr(app.state, "batcher", None)
        if batcher is not None:
            await batcher.stop()
    simulator = getattr(app.state, "simulator", None)
    if simulator is not None:
        await simulator.stop()
    receiver = _get_ndtp()
    if receiver is not None:
        await receiver.stop()
