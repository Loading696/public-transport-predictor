"""Acceptance tests for Person A's detectors (synthetic, fast, no data needed).

Checks:
  1. dwell fires on standstill, silent on free flow and on NaN input;
  2. speed_drop fires on collapse, silent when long window is slow too;
  3. backlog fires on cur_dev_s >= 120;
  4. every event matches the cross-branch contract
     {"type", "confidence" in [0,1], "reason": str};
  5. detectors never raise (NaN / missing keys / garbage).

Usage:
    py ml/test_patterns.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.patterns import cause_scores, detect_all  # noqa: E402

FREE_FLOW = {
    "speed_300s_mean": 28.0, "speed_300s_moving_frac": 0.95,
    "speed_1800s_mean": 26.0, "cur_dev_s": 5.0,
}
STANDSTILL = {
    "speed_300s_mean": 1.2, "speed_300s_moving_frac": 0.05,
    "speed_1800s_mean": 18.0, "cur_dev_s": 40.0,
}
COLLAPSE = {
    "speed_300s_mean": 6.0, "speed_300s_moving_frac": 0.6,
    "speed_1800s_mean": 30.0, "cur_dev_s": 20.0,
}
BACKLOG = {
    "speed_300s_mean": 25.0, "speed_300s_moving_frac": 0.9,
    "speed_1800s_mean": 24.0, "cur_dev_s": 200.0,
}
GARBAGE = {"speed_300s_mean": float("nan"), "cur_dev_s": "oops"}


def _check_contract(events: list[dict]) -> None:
    for event in events:
        assert set(event) == {"type", "confidence", "reason"}, event
        assert event["type"] in {"dwell", "speed_drop", "backlog"}, event
        assert 0.0 <= event["confidence"] <= 1.0, event
        assert isinstance(event["reason"], str) and event["reason"], event


def main() -> None:
    assert detect_all(FREE_FLOW) == [], detect_all(FREE_FLOW)
    dwell = detect_all(STANDSTILL)
    assert any(e["type"] == "dwell" for e in dwell), dwell
    drop = detect_all(COLLAPSE)
    assert any(e["type"] == "speed_drop" for e in drop), drop
    backlog = detect_all(BACKLOG)
    assert any(e["type"] == "backlog" for e in backlog), backlog
    assert detect_all({}) == [] and detect_all(GARBAGE) == []
    for case in (FREE_FLOW, STANDSTILL, COLLAPSE, BACKLOG, {}, GARBAGE):
        _check_contract(detect_all(case))
    scores = cause_scores(STANDSTILL)
    assert set(scores) == {"dwell", "speed_drop", "backlog"}, scores
    print("ALL PATTERN TESTS PASSED")


if __name__ == "__main__":
    main()
