from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from pathlib import Path

from src.runtime import InferenceService, StreamSimulator

DEFAULT_SPEED = float(os.getenv("SIMULATION_SPEED", "60"))
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
]


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
    return simulator.status()


@app.on_event("startup")
async def on_startup() -> None:
    await _get_simulator()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    simulator = getattr(app.state, "simulator", None)
    if simulator is not None:
        await simulator.stop()
