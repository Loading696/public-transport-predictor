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

Roles (measured on test, late_rate=24.4%):
  backlog     — strong_signal (P=0.46, R=0.51): the only real predictor;
  dwell       — weak_signal   (P=0.24, R=0.19): at base rate, no signal;
  speed_drop  — weak_signal   (P=0.25, R=0.07): at base rate, tiny support;
  stale       — data_quality  (P=0.00, R=0.00): NOT a delay predictor, flags
                untrustworthy telemetry. Excluded from cause_scores().

Category split (the contract the dispatcher relies on):

  CAUSE    backlog, dwell, speed_drop — explain *why the vehicle is late*.
           Ranked among themselves via :data:`CAUSE_RANK` (strong, then weak).
  QUALITY  stale — says *how far the telemetry can be trusted*. It is never a
           reason for a delay and is never ranked against a cause, so a stale
           feed sitting next to a real backlog leaves the backlog as the cause
           and reports the staleness separately as a data-quality warning.

Use :func:`split_events` (or :func:`rank_cause_events` and
:func:`quality_events`) to get the two categories apart.  Putting them in one
ordered list and picking the head is what used to let ``stale`` outrank
``backlog`` whenever both fired.
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

ROLE_WEAK = "weak_signal"
ROLE_STRONG = "strong_signal"
ROLE_QUALITY = "data_quality"
EVENT_ROLE = {
    "dwell": ROLE_WEAK,
    "speed_drop": ROLE_WEAK,
    "backlog": ROLE_STRONG,
    "stale": ROLE_QUALITY,
}
DELAY_TYPES = ("dwell", "speed_drop", "backlog")
QUALITY_TYPES = ("stale",)

#: Ranking of *cause* events, strongest category first.  Only cause events are
#: ranked: quality events are partitioned out beforehand and are never compared
#: against a cause.  A stale feed is not a reason for a delay, so it must not be
#: able to outrank one -- ``ROLE_QUALITY`` is deliberately absent from this map.
CAUSE_RANK = {ROLE_STRONG: 0, ROLE_WEAK: 1}


def is_cause_event(event: Mapping[str, Any]) -> bool:
    """True when the event explains *why the vehicle is late*.

    Decided by the event's declared ``role`` first (the contract the detectors
    publish) and by the type set second, so an event that arrives without a role
    -- or with a role the module does not know -- is still classified by what it
    actually is rather than silently dropped.
    """
    if not isinstance(event, Mapping):
        return False
    role = event.get("role")
    if role == ROLE_QUALITY:
        return False
    kind = str(event.get("type", ""))
    if kind in QUALITY_TYPES:
        return False
    return role in CAUSE_RANK or kind in DELAY_TYPES


def is_quality_event(event: Mapping[str, Any]) -> bool:
    """True when the event is a statement about telemetry trustworthiness."""
    if not isinstance(event, Mapping):
        return False
    if event.get("role") == ROLE_QUALITY:
        return True
    return str(event.get("type", "")) in QUALITY_TYPES and not is_cause_event(event)


def _confidence(event: Mapping[str, Any]) -> float:
    try:
        value = float(event.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _normalised(event: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of an event with ``confidence`` coerced to a float in [0, 1].

    The detectors always emit a valid float, but these helpers also run over
    events assembled elsewhere, and a raw non-numeric confidence would otherwise
    travel into the API payload and break its schema.
    """
    out = dict(event)
    out["confidence"] = _clip01(_confidence(event))
    return out


def rank_cause_events(events: Any) -> list[dict[str, Any]]:
    """Cause events only, strongest first: strong category, then confidence.

    Ties are broken by the event type so the order is deterministic regardless of
    the order the detectors happened to fire in.
    """
    if not isinstance(events, (list, tuple)):
        return []
    causes = [_normalised(e) for e in events if is_cause_event(e)]
    return sorted(
        causes,
        key=lambda e: (
            CAUSE_RANK.get(str(e.get("role")), len(CAUSE_RANK)),
            -_confidence(e),
            str(e.get("type", "")),
        ),
    )


def quality_events(events: Any) -> list[dict[str, Any]]:
    """Data-quality events only, most severe first by confidence."""
    if not isinstance(events, (list, tuple)):
        return []
    found = [_normalised(e) for e in events if is_quality_event(e)]
    return sorted(found, key=lambda e: (-_confidence(e), str(e.get("type", ""))))


def split_events(events: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition events into ``(ranked_causes, quality)``.

    This is the single place the two categories are separated.  Callers must rank
    causes and report quality independently -- mixing them in one ordered list is
    what previously let ``stale`` outrank ``backlog``.
    """
    return rank_cause_events(events), quality_events(events)


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
            "role": EVENT_ROLE["dwell"],
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
            "role": EVENT_ROLE["speed_drop"],
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
            "role": EVENT_ROLE["backlog"],
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
            "role": EVENT_ROLE["stale"],
            "confidence": round(_clip01(age / 3600.0), 3),
            "reason": f"телеметрия устарела: последнее событие {age:.0f} с назад",
        }
    if gps is not None and gps < STALE_GPS_VALID:
        return {
            "type": "stale",
            "role": EVENT_ROLE["stale"],
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
    """Delay-cause confidence map (data_quality events excluded, see stale)."""
    scores = {t: 0.0 for t in DELAY_TYPES}
    for event in detect_all(feat):
        if event["type"] in scores:
            scores[event["type"]] = max(scores[event["type"]], float(event["confidence"]))
    return scores


def quality_flags(feat: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Data-quality events (role == data_quality): stale telemetry, not delays."""
    return quality_events(detect_all(feat))
