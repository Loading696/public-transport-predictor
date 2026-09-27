"""Tests for the shared timetable module, ``src/schedule_index``.

This module exists because two services read the same plan: the backend needs
it to pick a target stop, measure a live deviation and build the cascade graph,
and the ML service needs it for static features.  Neither may depend on the
other's code, so the plan parsing lives here -- free of CatBoost.

The risk with such a module is a silent transcription error.  It already happened
once: during the extraction from ``src.predictor``, ``haversine_km`` came out
using ``lon2`` where the latitude belongs, which turned a zero-metre distance
into 2011 km and silently corrupted every ``path_distance_*`` feature.  The
tests below pin the geometry against values anyone can check by hand, so a
re-transcription fails loudly instead of quietly producing plausible numbers.

Run: py ml\\test_schedule_index.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schedule_index import (  # noqa: E402
    EARTH_RADIUS_KM,
    ScheduleIndex,
    haversine_km,
    parse_point_wkt,
    schedule_index,
)

KMS_PER_DEG_LAT = math.pi * EARTH_RADIUS_KM / 180.0


def test_haversine_zero_and_symmetry() -> None:
    """The two properties a transcription error destroys first."""
    same = haversine_km(np.array([37.61]), np.array([55.70]), 37.61, 55.70)
    assert same[0] == 0.0, f"a point is {same[0]} km from itself: {same[0] * 1000:.0f} m"

    lon1 = np.array([37.60, 37.62, 37.65])
    lat1 = np.array([55.70, 55.72, 55.68])
    forward = haversine_km(lon1, lat1, 37.61, 55.71)
    backward = haversine_km(np.array([37.61]), np.array([55.71]), lon1, lat1)
    assert np.allclose(forward, backward), (forward, backward)
    print("   haversine: self-distance 0, symmetric")


def test_haversine_known_distances() -> None:
    """Pinned against values derivable by hand.

    One degree of latitude is a fixed 111.19 km anywhere; a degree of longitude
    is that times cos(latitude).  If the arguments are transposed these come out
    as thousands of kilometres, which is exactly the failure being guarded.
    """
    # 0.01 deg of latitude at any longitude
    north = haversine_km(np.array([37.61]), np.array([55.70]), 37.61, 55.71)
    expected = 0.01 * KMS_PER_DEG_LAT
    assert abs(north[0] - expected) < 0.01, (north[0], expected)

    # 0.01 deg of longitude at 55.7 N
    east = haversine_km(np.array([37.60]), np.array([55.70]), 37.61, 55.70)
    expected_lon = 0.01 * KMS_PER_DEG_LAT * math.cos(math.radians(55.70))
    assert abs(east[0] - expected_lon) < 0.01, (east[0], expected_lon)

    # A transposed call would be ~2000 km; make that explicit so the intent is
    # clear if someone "simplifies" the signature again.
    transposed = haversine_km(np.array([37.60]), np.array([55.70]), 37.60, 55.71)
    assert transposed[0] < 5.0, f"transposed arguments would give {transposed[0]:.0f} km"
    print(
        f"   haversine: 0.01deg lat = {north[0] * 1000:.0f} m, "
        f"0.01deg lon = {east[0] * 1000:.0f} m (hand-checkable)"
    )


def test_haversine_matches_reference_formula() -> None:
    """Against a plain scalar implementation, not another copy of this one."""
    def reference(lon_a, lat_a, lon_b, lat_b):
        phi1, phi2 = math.radians(lat_a), math.radians(lat_b)
        dphi = math.radians(lat_b - lat_a)
        dlam = math.radians(lon_b - lon_a)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return 2 * 6371.0088 * math.asin(math.sqrt(a))

    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(500):
        lon1 = rng.uniform(37.0, 38.0, size=4)
        lat1 = rng.uniform(55.0, 56.0, size=4)
        lon2, lat2 = float(rng.uniform(37.0, 38.0)), float(rng.uniform(55.0, 56.0))
        got = haversine_km(lon1, lat1, lon2, lat2)
        for i in range(4):
            worst = max(worst, abs(float(got[i]) - reference(lon1[i], lat1[i], lon2, lat2)))
    assert worst < 1e-9, f"max deviation from the reference formula: {worst}"
    print(f"   haversine: matches a scalar reference on 500 random pairs (max {worst:.1e} km)")


def test_parse_point_wkt() -> None:
    assert parse_point_wkt("POINT (37.859122 55.730342)") == (37.859122, 55.730342)
    assert parse_point_wkt("POINT(37.1 55.2)") == (37.1, 55.2)
    for junk in ("", "LINESTRING (1 2)", None, 42, "POINT ()"):
        lon, lat = parse_point_wkt(junk)
        assert math.isnan(lon) and math.isnan(lat), (junk, lon, lat)
    # Memoised: the same string must give the same value every time.
    assert parse_point_wkt("POINT (1 2)") == (1.0, 2.0)
    print("   wkt:       parses POINT, nan for junk, memoised")


def _sample_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tt_action_item_id": [1, 2, 3, 1, 2],
            "tr_id": [10, 10, 10, 20, 20],
            "time_begin": pd.to_datetime(
                [
                    "2026-01-06 08:00:00", "2026-01-06 08:01:00", "2026-01-06 08:02:00",
                    "2026-01-06 09:00:00", "2026-01-06 09:05:00",
                ]
            ),
            "geom": [
                "POINT (37.600 55.700)",
                "POINT (37.610 55.700)",
                "POINT (37.620 55.700)",
                "POINT (37.600 55.700)",
                "POINT (37.650 55.750)",
            ],
            "building_address": ["a", "b", "c", "d", "e"],
        }
    )


def test_schedule_index_contents() -> None:
    frame = _sample_schedule()
    index = schedule_index(frame)
    assert isinstance(index, ScheduleIndex)
    prepared = index.prepared
    assert len(prepared) == 5, prepared
    assert set(prepared["stop_idx"]) == {0, 1, 2, 0, 1}, prepared["stop_idx"].tolist()
    assert prepared["route_stop_count"].tolist() == [3, 3, 3, 2, 2]
    # inter-stop gaps
    gaps = prepared[prepared["tr_id"] == 10]["planned_gap_prev_s"].tolist()
    assert math.isnan(gaps[0]) and gaps[1] == 60.0 and gaps[2] == 60.0, gaps
    # leg distance along the route, computed with the haversine
    legs = prepared[prepared["tr_id"] == 10]["leg_distance_km"].tolist()
    expected = 0.01 * KMS_PER_DEG_LAT * math.cos(math.radians(55.70))
    assert abs(legs[0] - expected) < 0.01, (legs[0], expected)
    assert math.isnan(legs[2]), legs
    assert set(index.by_vehicle) == {10, 20}
    assert len(index.by_vehicle[10]) == 3
    print(f"   index:     stop_idx/route_stop_count/gaps/legs correct (leg={legs[0] * 1000:.0f} m)")


def test_schedule_index_is_memoised() -> None:
    frame = _sample_schedule()
    first = schedule_index(frame)
    second = schedule_index(frame)
    assert first is second, "the index should be cached on the frame"
    assert schedule_index(first) is first, "an index should pass through unchanged"
    print("   index:     memoised on the frame, pass-through for an index")


def test_module_has_no_ml_dependency() -> None:
    """The point of the module: importing it must not drag in the ML stack."""
    import subprocess

    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "import src.schedule_index;"
        "print(int('catboost' in sys.modules), int('sklearn' in sys.modules))"
        % ROOT
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "0 0", out.stdout
    print("   purity:    importing it pulls in neither catboost nor sklearn")


def main() -> None:
    tests = [
        test_haversine_zero_and_symmetry,
        test_haversine_known_distances,
        test_haversine_matches_reference_formula,
        test_parse_point_wkt,
        test_schedule_index_contents,
        test_schedule_index_is_memoised,
        test_module_has_no_ml_dependency,
    ]
    for test in tests:
        print(f"[run ] {test.__name__}")
        test()
    print("ALL SCHEDULE-INDEX TESTS PASSED")


if __name__ == "__main__":
    main()
