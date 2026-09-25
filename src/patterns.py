"""Event detectors over causal model features (Person A).

Pure functions: input is a mapping of already-built causal features
(see src/predictor.py feature list), output follows the cross-branch contract:

    {"type": "dwell" | "speed_drop" | "backlog", "confidence": 0..1, "reason": str}

Conventions:
  - never raise on NaN/missing keys: unknown -> confidence 0.0 or None event;
  - thresholds are module constants so Person B can display them;
  - no I/O, no model loading here (calibration lives in ml/calibrate.py).

Feature names used (all causal, event_time <= T):
  speed_{60,180,300,600,900,1800}s_{mean,moving_frac,le_5},
  gps_valid_300s, last_event_age_s, cur_dev_s, target_distance_km,
  planned_stops_between, telemetry_points.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

DWELL_SPEED_KMH = 3.0
DWELL_MOVE_FRAC = 0.2
DWELL_WINDOW = "speed_300s"
DROP_SHORT_WINDOW = "speed_300s"
DROP_LONG_WINDOW = "speed_1800s"
DROP_RATIO = 0.5
DROP_LONG_MIN_KMH = 10.0
STALE_AGE_S = 900.0
STALE_GPS_VALID = 0.3
BACKLOG_S = 120.0


def _num(feat: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(feat.get(key))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def detect_dwell(feat: Mapping[str, Any]) -> dict[str, Any] | None:
    """Long boarding / jam standstill: low mean speed + barely moving.

    TODO(Person A): tune thresholds on train labels; consider le_5 share.
    """
    mean = _num(feat, f"{DWELL_WINDOW}_mean")
    move = _num(feat, f"{DWELL_WINDOW}_moving_frac")
    if mean is None or move is None:
        return None
    if mean <= DWELL_SPEED_KMH and move <= DWELL_MOVE_FRAC:
        confidence = _clip01((DWELL_SPEED_KMH - mean) / DWELL_SPEED_KMH * 0.7 + (DWELL_MOVE_FRAC - move) * 1.5)
        return {
            "type": "dwell",
            "confidence": round(confidence, 3),
            "reason": f"простой: средняя скорость {mean:.1f} км/ч, доля движения {move:.2f} за 5 мин",
        }
    return None


def detect_speed_drop(feat: Mapping[str, Any]) -> dict[str, Any] | None:
    """Anomalous slowdown on approach: short-window mean collapsed vs long window.

    TODO(Person A): validate precision/recall vs late labels; try 600s/1800s pairs.
    """
    short = _num(feat, f"{DROP_SHORT_WINDOW}_mean")
    long = _num(feat, f"{DROP_LONG_WINDOW}_mean")
    if short is None or long is None or long < DROP_LONG_MIN_KMH:
        return None
    if short < DROP_RATIO * long:
        confidence = _clip01((long - short) / long)
        return {
            "type": "speed_drop",
            "confidence": round(confidence, 3),
            "reason": f"просадка скорости: {short:.1f} vs {long:.1f} км/ч на длинном окне",
        }
    return None


def detect_backlog(feat: Mapping[str, Any]) -> dict[str, Any] | None:
    """Already-accumulated schedule backlog carrying into the target."""
    cur = _num(feat, "cur_dev_s")
    if cur is None:
        return None
    if cur >= BACKLOG_S:
        return {
            "type": "backlog",
            "confidence": round(_clip01(cur / 300.0), 3),
            "reason": f"накопленное отставание {cur:.0f} с сохраняется к цели",
        }
    return None


def detect_stale(feat: Mapping[str, Any]) -> dict[str, Any] | None:
    """Stale or lost telemetry: last event far behind T or GPS mostly invalid."""
    age = _num(feat, "last_event_age_s")
    gps = _num(feat, "gps_valid_300s")
    if age is not None and age > STALE_AGE_S:
        return {
            "type": "stale",
            "confidence": round(_clip01(age / 3600.0), 3),
            "reason": f"телеметрия устарела: последнее событие {age:.0f} с назад",
        }
    if gps is not None and gps < STALE_GPS_VALID:
        return {
            "type": "stale",
            "confidence": round(_clip01(1.0 - gps / STALE_GPS_VALID), 3),
            "reason": f"потеря GPS: доля валидных координат {gps:.2f} за 5 мин",
        }
    return None


def detect_all(feat: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Run every detector; strongest first. Never raises on bad input."""
    events = []
    for detector in (detect_stale, detect_dwell, detect_speed_drop, detect_backlog):
        try:
            event = detector(feat)
        except Exception:  # noqa: BLE001 - detectors must be total functions
            continue
        if event is not None:
            events.append(event)
    return sorted(events, key=lambda item: item["confidence"], reverse=True)


def cause_scores(feat: Mapping[str, Any]) -> dict[str, float]:
    """Per-cause confidence map for the incident card. Keys are stable API."""
    scores = {"dwell": 0.0, "speed_drop": 0.0, "backlog": 0.0, "stale": 0.0}
    for event in detect_all(feat):
        scores[event["type"]] = max(scores[event["type"]], float(event["confidence"]))
    return scores
