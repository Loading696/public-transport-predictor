"""Cause-selection tests: a data-quality signal must never become the reason.

The bug this guards against
---------------------------
``cause_from_events`` used to sort every event into one list keyed by role, with
``data_quality`` ranked *first*::

    key=lambda e: ({"data_quality": 0, "strong_signal": 1}.get(e.get("role"), 2), -confidence)

So whenever ``stale`` fired alongside any real pattern it won outright and the
incident reported "телеметрия недостоверна" as the cause -- the dispatcher was
told the *reason* for the delay was a broken feed.  That is backwards: ``stale``
has measured precision 0.00 as a delay predictor; it is a statement about how far
the telemetry can be trusted, not about why the bus is late.

The contract now
----------------
  CAUSE    backlog / dwell / speed_drop  -> ranked among themselves
  QUALITY  stale                         -> never ranked against a cause

A stale feed next to a real backlog keeps the backlog as the cause and surfaces
the staleness separately, as a data-quality warning.

Run: py ml/test_cause_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.patterns import (  # noqa: E402
    CAUSE_RANK,
    DELAY_TYPES,
    is_cause_event,
    is_quality_event,
    quality_events,
    rank_cause_events,
    split_events,
)
from src.runtime import (  # noqa: E402
    CAUSE_UNKNOWN_UNTRUSTWORTHY,
    diagnose_events,
)

STALE = {"type": "stale", "role": "data_quality", "confidence": 0.4,
         "reason": "телеметрия устарела: последнее событие 1200 с назад"}
STALE_LOUD = {"type": "stale", "role": "data_quality", "confidence": 0.95,
              "reason": "потеря GPS: доля валидных координат 0.02 за 5 мин"}
BACKLOG = {"type": "backlog", "role": "strong_signal", "confidence": 0.93,
           "reason": "накопленное отставание 280 с сохраняется к цели"}
DWELL = {"type": "dwell", "role": "weak_signal", "confidence": 0.62,
         "reason": "простой: средняя скорость 1.1 км/ч"}
DROP = {"type": "speed_drop", "role": "weak_signal", "confidence": 0.55,
        "reason": "просадка скорости: 6.0 vs 30.0 км/ч"}
DROP_WEAK = {"type": "speed_drop", "role": "weak_signal", "confidence": 0.11,
             "reason": "просадка скорости: 27.0 vs 30.0 км/ч"}


def _cause_type(diag) -> str | None:
    return diag.cause_type


def test_categories_are_disjoint() -> None:
    """Every known event is exactly one of: cause, quality. Never both."""
    for kind in DELAY_TYPES:
        event = {"type": kind, "role": "strong_signal" if kind == "backlog" else "weak_signal"}
        assert is_cause_event(event) and not is_quality_event(event), event
    stale = {"type": "stale", "role": "data_quality"}
    assert is_quality_event(stale) and not is_cause_event(stale), stale
    # A quality-typed event stays quality even if it arrives without a role...
    assert is_quality_event({"type": "stale"}), "stale without role"
    # ...and a cause-typed event stays a cause even with an unknown role.
    assert is_cause_event({"type": "backlog", "role": "brand_new"}), "unknown role"
    assert "data_quality" not in CAUSE_RANK, CAUSE_RANK
    print("   disjoint:  категории не пересекаются, data_quality не ранжируется")


def test_split_never_mixes_categories() -> None:
    events = [STALE, BACKLOG, DWELL, STALE_LOUD]
    causes, quality = split_events(events)
    assert [c["type"] for c in causes] == ["backlog", "dwell"], causes
    assert {q["type"] for q in quality} == {"stale"} and len(quality) == 2, quality
    assert all(is_cause_event(c) for c in causes)
    assert all(is_quality_event(q) for q in quality)
    print(f"   split:     causes={[c['type'] for c in causes]} quality={[q['type'] for q in quality]}")


def test_stale_plus_backlog_keeps_backlog() -> None:
    """The headline case from the bug report."""
    for stale_event in (STALE, STALE_LOUD):
        diag = diagnose_events([stale_event, BACKLOG])
        assert _cause_type(diag) == "backlog", diag
        assert diag.cause_role == "strong_signal", diag
        assert "отставание" in diag.cause, diag.cause
        # The quality complaint must survive, just not as the cause.
        assert diag.quality["stale"] is True, diag.quality
        assert diag.quality["status"] == "degraded", diag.quality
        assert diag.quality["events"] and diag.quality["events"][0]["type"] == "stale"
        assert diag.additional == [], diag.additional
    # Even a maximal-confidence stale must not win.
    diag = diagnose_events([{"type": "stale", "role": "data_quality", "confidence": 1.0}, BACKLOG])
    assert _cause_type(diag) == "backlog", diag
    print("   stale+backlog: причиной остаётся backlog, stale — предупреждение о качестве")


def test_stale_plus_dwell() -> None:
    diag = diagnose_events([STALE, DWELL])
    assert _cause_type(diag) == "dwell", diag
    assert diag.cause_role == "weak_signal", diag
    assert diag.quality["stale"] is True
    print("   stale+dwell:    причина dwell (weak_signal), stale отдельно")


def test_stale_plus_speed_drop() -> None:
    diag = diagnose_events([STALE_LOUD, DROP])
    assert _cause_type(diag) == "speed_drop", diag
    assert diag.cause_role == "weak_signal", diag
    assert diag.quality["stale"] is True
    print("   stale+speed_drop: причина speed_drop (weak_signal), stale отдельно")


def test_strong_beats_weak_regardless_of_confidence() -> None:
    """A strong signal outranks a more confident weak one: role first, then conf."""
    loud_weak = {**DWELL, "confidence": 0.99}
    diag = diagnose_events([STALE, loud_weak, BACKLOG])
    assert _cause_type(diag) == "backlog", diag
    assert [a["type"] for a in diag.additional] == ["dwell"], diag.additional
    # Without the strong signal the confident weak one leads.
    diag = diagnose_events([STALE, loud_weak, DROP])
    assert _cause_type(diag) == "dwell", diag
    assert [a["type"] for a in diag.additional] == ["speed_drop"], diag.additional
    print("   ranking:    strong > weak, затем по confidence; остальные — дополнительные")


def test_multiple_strong_signals() -> None:
    """Two strong-class events: highest confidence leads, the other is kept."""
    second_strong = {**BACKLOG, "type": "dwell", "role": "strong_signal", "confidence": 0.4}
    diag = diagnose_events([second_strong, BACKLOG, STALE])
    assert _cause_type(diag) == "backlog", diag
    assert [a["type"] for a in diag.additional] == ["dwell"], diag.additional
    assert diag.quality["stale"] is True
    # Order of arrival must not change the outcome.
    for order in ([STALE, BACKLOG, second_strong], [second_strong, STALE, BACKLOG]):
        other = diagnose_events(order)
        assert _cause_type(other) == "backlog" and [a["type"] for a in other.additional] == ["dwell"]
    print("   multi:      два strong — лидирует увереннейший, второй не потерян")


def test_all_three_causes_together() -> None:
    diag = diagnose_events([STALE, DROP_WEAK, DWELL, BACKLOG])
    assert _cause_type(diag) == "backlog", diag
    assert [a["type"] for a in diag.additional] == ["dwell", "speed_drop"], diag.additional
    assert diag.quality["stale"] is True
    assert all(a["reason"] for a in diag.additional), diag.additional
    print("   full:       backlog + dwell + speed_drop + stale — ничего не потеряно")


def test_stale_only_is_not_a_cause() -> None:
    """With only quality evidence the pattern is undetermined, and says so."""
    diag = diagnose_events([STALE])
    assert diag.cause == CAUSE_UNKNOWN_UNTRUSTWORTHY, diag
    assert diag.cause_role == "data_quality", diag
    assert diag.cause_type is None, diag
    assert diag.quality["stale"] is True
    assert diag.quality["status"] == "degraded"
    print("   stale only: причина не определена, качество помечено как degraded")


def test_no_events_is_plain_unknown() -> None:
    diag = diagnose_events([])
    assert diag.cause_role == "none", diag
    assert diag.cause_type is None
    assert diag.additional == []
    assert diag.quality["status"] == "ok" and diag.quality["stale"] is False
    assert diagnose_events(None).cause_role == "none"
    assert diagnose_events("garbage").cause_role == "none"
    print("   empty:     нет событий -> причина неизвестна, качество ok")


def test_freshness_is_reported_separately() -> None:
    """Quality block carries the measured numbers, not just a boolean."""
    diag = diagnose_events(
        [STALE, BACKLOG], last_event_age_s=1420.0, gps_valid=0.18
    )
    assert diag.quality["last_event_age_s"] == 1420.0, diag.quality
    assert diag.quality["gps_valid_300s"] == 0.18, diag.quality
    clean = diagnose_events([BACKLOG], last_event_age_s=12.0, gps_valid=0.98)
    assert clean.quality["status"] == "ok" and clean.quality["stale"] is False
    assert _cause_type(clean) == "backlog"
    print("   freshness:  возраст и доля GPS передаются в API вместе с причиной")


def test_rank_helpers_are_total() -> None:
    assert rank_cause_events(None) == []
    assert quality_events(None) == []
    assert rank_cause_events([1, "x", None]) == []
    # A junk event with an unusable confidence must not raise.
    junk = [{"type": "backlog", "role": "strong_signal", "confidence": "oops"}]
    ranked = rank_cause_events(junk)
    assert ranked and ranked[0]["confidence"] == 0.0, ranked
    assert is_cause_event(junk[0]) and not is_quality_event(junk[0])
    print("   total:      мусор на входе не роняет ни один путь")


def test_detector_semantics_unchanged() -> None:
    """The fix is a composition change: detectors must still fire as before."""
    from src.patterns import cause_scores, detect_all, quality_flags

    both = {
        "speed_300s_mean": 25.0, "speed_300s_moving_frac": 0.9,
        "speed_1800s_mean": 24.0, "cur_dev_s": 280.0,
        "last_event_age_s": 1200.0, "gps_valid_300s": 0.9,
    }
    events = detect_all(both)
    kinds = {e["type"] for e in events}
    assert kinds == {"backlog", "stale"}, events
    # Roles unchanged: stale is still data_quality, backlog still strong_signal.
    roles = {e["type"]: e["role"] for e in events}
    assert roles["stale"] == "data_quality" and roles["backlog"] == "strong_signal", roles
    # Both are still reported to the caller, and stale is still excluded from scores.
    assert "stale" not in cause_scores(both), cause_scores(both)
    assert quality_flags(both) and quality_flags(both)[0]["type"] == "stale"
    diag = diagnose_events(events)
    assert _cause_type(diag) == "backlog", diag
    assert diag.quality["stale"] is True, diag.quality
    print("   detectors:  роли и срабатывания не изменились, изменилась только композиция")


def main() -> None:
    tests = [
        test_categories_are_disjoint,
        test_split_never_mixes_categories,
        test_stale_plus_backlog_keeps_backlog,
        test_stale_plus_dwell,
        test_stale_plus_speed_drop,
        test_strong_beats_weak_regardless_of_confidence,
        test_multiple_strong_signals,
        test_all_three_causes_together,
        test_stale_only_is_not_a_cause,
        test_no_events_is_plain_unknown,
        test_freshness_is_reported_separately,
        test_rank_helpers_are_total,
        test_detector_semantics_unchanged,
    ]
    for test in tests:
        print(f"[run ] {test.__name__}")
        test()
    print("ALL CAUSE-SELECTION TESTS PASSED")


if __name__ == "__main__":
    main()
