from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from src.runtime import InferenceService, StreamSimulator
from src.ndtp_server import NDTPReceiver

DEFAULT_SPEED = float(os.getenv("SIMULATION_SPEED", "60"))
TCP_PORT = int(os.getenv("TCP_PORT", "9201"))
# Троттлинг live-прогнозов: пересчёт для юнита не чаще TTL, повторные опросы
# /stream/status без новых данных отдают кэшированный прогноз.
LIVE_PREDICT_TTL = float(os.getenv("LIVE_PREDICT_TTL", "10"))
app = FastAPI(title="Transport Delay MVP", version="1.0.0")

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


def _get_runtime() -> InferenceService:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        runtime = InferenceService()
        app.state.runtime = runtime
    return runtime


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
]


def _get_ndtp() -> NDTPReceiver | None:
    return getattr(app.state, "ndtp", None)


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


# Горизонт цели, на котором училась модель: плановая остановка строго
# в окне (T+600, T+900] (см. lead_minus_660 = lead_s - 660 в фичах).
# Цель ближе/дальше — out-of-distribution: lead_s улетает, прогноз мусорный,
# поэтому вне окна прогноз не выдаём (prediction=None + пометка).
LIVE_TARGET_MIN_S = 600
LIVE_TARGET_MAX_S = 900


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
    # Прогнозы для живого потока (best-effort: без расписания/модели — без прогноза).
    try:
        runtime = _get_runtime()
    except Exception:
        for unit in units:
            unit.pop("_event_time", None)
            unit.update(
                {
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
            )
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
            target_id, target_time, target_status = _live_target(
                runtime, unit["tr_id"], event_time
            )
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
            result = await asyncio.to_thread(runtime.predict_records, points, telemetry or None)
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


def _target_note(status: str) -> str | None:
    """Человекочитаемая пометка статуса цели для витрины."""
    return {
        "ok": None,
        "no_schedule": "нет расписания для ТС",
        "no_time": "нет времени события",
        "out_of_horizon": "цель вне горизонта 10–15 мин",
        "no_model": "прогноз недоступен",
    }.get(status, "прогноз недоступен")


def _live_target(runtime: Any, tr_id: int, event_time: Any) -> tuple[Any, Any, str]:
    """Первая плановая остановка по tr_id в окне (T+600, T+900].

    Возвращает (target_stop_id, target_time_begin, status), где status:
      "ok" — цель в горизонте, можно прогнозировать;
      "no_schedule" — tr_id нет в расписании;
      "no_time" — время события отсутствует/не парсится, окно не проверить;
      "out_of_horizon" — в окне остановок нет (ближайшая через <600с или
        только дальние >900с): цель OOD, прогноз не выдаём.
    """
    try:
        import pandas as pd

        schedule = runtime.schedule
        frame = schedule[pd.to_numeric(schedule["tr_id"], errors="coerce") == tr_id].copy()
        if frame.empty:
            return None, None, "no_schedule"
        frame["time_begin"] = pd.to_datetime(frame["time_begin"], errors="coerce")
        frame = frame.dropna(subset=["time_begin"]).sort_values("time_begin", kind="stable")
        if frame.empty:
            return None, None, "no_schedule"
        ref = pd.to_datetime(event_time, errors="coerce") if event_time is not None else None
        if ref is None or pd.isna(ref):
            return None, None, "no_time"
        window = frame[
            (frame["time_begin"] > ref + pd.Timedelta(seconds=LIVE_TARGET_MIN_S))
            & (frame["time_begin"] <= ref + pd.Timedelta(seconds=LIVE_TARGET_MAX_S))
        ]
        if window.empty:
            return None, None, "out_of_horizon"
        row = window.iloc[0]
        target_id = row.get("tt_action_item_id")
        try:
            target_id = int(target_id)
        except (TypeError, ValueError):
            return None, None, "no_schedule"
        target_time = row.get("time_begin")
        if pd.isna(target_time):
            return None, None, "no_schedule"
        return target_id, target_time, "ok"
    except Exception:
        return None, None, "no_model"


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
def health() -> dict[str, Any]:
    try:
        runtime = _get_runtime()
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
    request = _parse_request(payload)
    point_rows = [_dump(point) for point in request.points]
    telemetry_rows = [_dump(event) for event in request.telemetry]
    try:
        predictions = await asyncio.to_thread(
            _get_runtime().predict_records,
            point_rows,
            telemetry_rows,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"count": len(predictions), "predictions": predictions}


@app.post("/generate_submission")
async def generate_submission() -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(_get_runtime().generate_submission)
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@app.get("/generate_submission")
async def generate_submission_get() -> dict[str, Any]:
    return await generate_submission()


@app.get("/stream/status")
async def stream_status(
    speed: float | None = Query(default=None, gt=0),
    reset: bool = Query(default=False),
) -> dict[str, Any]:
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
    runtime = _get_runtime()
    engine = runtime.cascade_engine()
    payload: dict[str, Any] = {"graph": engine.graph.network(), "view": None}
    simulator = getattr(app.state, "simulator", None)
    if simulator is not None:
        try:
            payload["view"] = simulator.cascade_view()
        except Exception as exc:  # noqa: BLE001 - never fail the whole response
            payload["view"] = {"enabled": False, "reason": str(exc)}
    return payload


@app.on_event("startup")
async def on_startup() -> None:
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
    simulator = getattr(app.state, "simulator", None)
    if simulator is not None:
        await simulator.stop()
    receiver = _get_ndtp()
    if receiver is not None:
        await receiver.stop()
