"""Tests for the dwell (standstill) feature in src/predictor (Person A).

Checks the causal semantics: only the supplied prefix is scanned, speed
threshold 3 km/h, seconds derived from event timestamps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predictor import _dwell_stats  # noqa: E402

BASE = 1_000_000_000  # seconds -> ns


def main() -> None:
    times = np.array([0, 10, 20, 30, 40, 50], dtype="int64") * BASE
    # 20s stall between t=10..30, then moving again.
    dwell, age = _dwell_stats(np.array([25.0, 1.0, 0.5, 0.0, 30.0, 28.0]), times)
    assert dwell == 20.0, dwell
    assert age == 40.0, age

    # No standstill: zero, no onset.
    dwell, age = _dwell_stats(np.array([25.0, 30.0, 40.0]), times[:3])
    assert dwell == 0.0 and np.isnan(age), (dwell, age)

    # Trailing stall is still detected, counted to the last event.
    dwell, age = _dwell_stats(np.array([0.0, 0.0, 0.0, 0.0]), times[:4])
    assert dwell == 30.0 and age == 30.0, (dwell, age)

    # Boundary: exactly 3 km/h counts as stopped.
    dwell, _ = _dwell_stats(np.array([3.0, 3.0]), times[:2])
    assert dwell == 10.0, dwell
    dwell, _ = _dwell_stats(np.array([3.01, 3.01]), times[:2])
    assert dwell == 0.0, dwell

    # NaN speeds are not treated as stops and do not break a run.
    dwell, _ = _dwell_stats(np.array([np.nan, 0.0, 0.0]), times[:3])
    assert dwell == 10.0, dwell

    # Empty input.
    dwell, age = _dwell_stats(np.array([]), times)
    assert np.isnan(dwell) and np.isnan(age), (dwell, age)

    print("ALL DWELL TESTS PASSED")


if __name__ == "__main__":
    main()
