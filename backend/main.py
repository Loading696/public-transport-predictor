from __future__ import annotations

import asyncio
import json
import os
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
app = FastAPI(title="Transport Delay MVP", version="1.0.0")


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


async def _live_units() -> list[dict[str, Any]]:
    """Latest position per connected NDTP unit (live emulator feed).

    Каждый юнит обогащается:
      - tr_id через UNIT_TR_MAP (env JSON {"unitId": tr_id}),
      - door_open из ячеек Crown03/Irma04 (True/False/None),
      - целью = ближайшая плановая остановка из расписания по tr_id
        (первая time_begin > event_time, иначе последняя),
      - прогнозом задержки через InferenceService.predict_records.
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
                    "prediction": None,
                    "risk": None,
                    "target_class": None,
                    "recommendation": None,
                }
            )
        return units
    try:
        points: list[dict[str, Any]] = []
        telemetry: list[dict[str, Any]] = []
        for unit in units:
            target_id, target_time = _live_target(runtime, unit["tr_id"], unit.pop("_event_time", None))
            unit["target_stop_id"] = target_id
            unit["target_time_begin"] = target_time.isoformat() if hasattr(target_time, "isoformat") else target_time
            unit["_target_time"] = target_time
            if target_id is None or target_time is None:
                unit.update({"prediction": None, "risk": None, "target_class": None, "recommendation": None})
                continue
            points.append(
                {
                    "tr_id": unit["tr_id"],
                    "T": unit["event_time"],
                    "target_stop_id": target_id,
                    "target_time_begin": unit["target_time_begin"],
                    "cur_dev_s": 0.0,
                }
            )
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
        predictions: dict[int, dict[str, Any]] = {}
        if points:
            result = await asyncio.to_thread(runtime.predict_records, points, telemetry or None)
            for record in result:
                try:
                    predictions[int(record.get("tr_id"))] = record
                except (TypeError, ValueError):
                    continue
        for unit in units:
            record = predictions.get(unit["tr_id"])
            target_time = unit.pop("_target_time", None)
            if record is None:
                if unit.get("target_stop_id") is None:
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
    except Exception:
        for unit in units:
            unit.pop("_event_time", None)
            unit.pop("_target_time", None)
            unit.setdefault("prediction", None)
            unit.setdefault("risk", None)
            unit.setdefault("target_stop_id", None)
            unit.setdefault("target_time_begin", None)
    return units


def _live_target(runtime: Any, tr_id: int, event_time: Any) -> tuple[Any, Any]:
    """Ближайшая плановая остановка из расписания по tr_id.

    Первая остановка с time_begin > event_time; если таких нет — последняя;
    если tr_id нет в расписании или время не распарсилось — (None, None).
    """
    try:
        import pandas as pd

        schedule = runtime.schedule
        frame = schedule[pd.to_numeric(schedule["tr_id"], errors="coerce") == tr_id].copy()
        if frame.empty:
            return None, None
        frame["time_begin"] = pd.to_datetime(frame["time_begin"], errors="coerce")
        frame = frame.dropna(subset=["time_begin"]).sort_values("time_begin", kind="stable")
        if frame.empty:
            return None, None
        ref = pd.to_datetime(event_time, errors="coerce") if event_time is not None else None
        if ref is None or pd.isna(ref):
            # Без времени события берём первую плановую остановку как цель.
            row = frame.iloc[0]
        else:
            future = frame[frame["time_begin"] > ref]
            row = future.iloc[0] if not future.empty else frame.iloc[-1]
        target_id = row.get("tt_action_item_id")
        try:
            target_id = int(target_id)
        except (TypeError, ValueError):
            return None, None
        target_time = row.get("time_begin")
        if pd.isna(target_time):
            return None, None
        return target_id, target_time
    except Exception:
        return None, None


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
