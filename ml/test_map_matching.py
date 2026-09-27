"""Tests for the simple GPS map matching in src/runtime.snap_to_polyline.

Checks the projection geometry itself (perpendicular snap onto a segment,
clamping at segment ends, nearest-of-several-segments) rather than any real
route data -- the function is deliberately just 2-D point-to-polyline
projection, no road graph involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime import snap_to_polyline  # noqa: E402


def main() -> None:
    # Two-stop route running east along a fixed latitude. A fix offset to
    # the north of the midpoint should snap straight down onto the segment.
    polyline = [[55.70, 37.60], [55.70, 37.62]]
    result = snap_to_polyline(37.61, 55.705, polyline)
    assert result is not None
    assert abs(result["lat"] - 55.70) < 1e-6, result
    assert abs(result["lon"] - 37.61) < 1e-6, result
    assert result["snap_distance_m"] > 0, result

    # A fix exactly on the polyline snaps to itself with ~0 distance.
    on_route = snap_to_polyline(37.61, 55.70, polyline)
    assert on_route is not None
    assert on_route["snap_distance_m"] < 1.0, on_route

    # A fix past the last stop clamps to the segment end, not an extrapolation.
    beyond = snap_to_polyline(37.63, 55.70, polyline)
    assert beyond is not None
    assert abs(beyond["lon"] - 37.62) < 1e-6, beyond

    # Three-stop route with a turn: the fix must snap to the nearer leg.
    turn = [[55.70, 37.60], [55.70, 37.62], [55.72, 37.62]]
    near_second_leg = snap_to_polyline(37.625, 55.71, turn)
    assert near_second_leg is not None
    assert abs(near_second_leg["lon"] - 37.62) < 1e-4, near_second_leg
    assert 55.70 <= near_second_leg["lat"] <= 55.72, near_second_leg

    # Degenerate inputs never raise, they just decline to match.
    assert snap_to_polyline(None, 55.70, polyline) is None
    assert snap_to_polyline(37.61, 55.70, []) is None
    assert snap_to_polyline(37.61, 55.70, [[55.70, 37.60]]) is None

    print("ALL MAP MATCHING TESTS PASSED")


if __name__ == "__main__":
    main()
