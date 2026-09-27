"""Planned-timetable parsing shared by the backend and the ML service.

This module exists so the two services can agree on *what the plan says* without
either importing the other's code.  It is deliberately free of CatBoost, of model
loading and of any inference: pure pandas/numpy over the static schedule.

The backend needs the plan to pick a 10-15 minute target stop, to measure a live
vehicle's deviation from it and to build the cascade route graph.  The ML service
needs it to derive static features and inter-stop geometry.  Both sides therefore
read the same ``schedule_plan.csv`` through the same code -- one file, two
containers -- which is a shared library, not a runtime coupling: the model and
the inference still cross the process boundary over HTTP.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

#: Attribute key used to memoise a prepared schedule on the DataFrame itself.
#: The schedule is static for the lifetime of the process, so parsing WKT,
#: sorting and computing inter-stop geometry once removes a large share of the
#: latency of every subsequent call.
_SCHEDULE_INDEX_KEY = "_prepared_schedule"


@lru_cache(maxsize=200_000)
def _parse_point_wkt_cached(value: str) -> tuple[float, float]:
    match = re.search(r"POINT\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", value)
    if not match:
        return math.nan, math.nan
    return float(match.group(1)), float(match.group(2))


def parse_point_wkt(value: object) -> tuple[float, float]:
    """Return ``(lon, lat)`` from the POINT WKT used by the schedule.

    Memoised on the raw string: the schedule is parsed repeatedly (feature
    building, route network, cascade graph) but every stop repeats the same
    handful of coordinate strings, so a plain LRU turns a regex scan of 5.5k
    rows into a dict lookup.
    """
    if not isinstance(value, str):
        return math.nan, math.nan
    return _parse_point_wkt_cached(value)


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: float, lat2: float) -> np.ndarray:
    """Vectorized great-circle distance in kilometres."""
    lon1 = np.asarray(lon1, dtype=float)
    lat1 = np.asarray(lat1, dtype=float)
    phi1 = np.deg2rad(lat1)
    lat2_arr = np.asarray(lat2, dtype=float)
    phi2 = np.deg2rad(lat2_arr)
    dphi = np.deg2rad(lat2_arr - lat1)
    dlambda = np.deg2rad(np.asarray(lon2, dtype=float) - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _haversine_grid_km(
    fix_lons: np.ndarray,
    fix_lats: np.ndarray,
    stop_lons: np.ndarray,
    stop_lats: np.ndarray,
) -> np.ndarray:
    """(fixes x stops) distance matrix, km.

    :func:`haversine_km` broadcasts a scalar second point against the first and
    so cannot express a full matrix; map matching and deviation estimation need
    the full matrix to pick the nearest stop per fix in one vectorised pass.
    """
    phi1 = np.deg2rad(fix_lats)[:, None]
    phi2 = np.deg2rad(stop_lats)[None, :]
    dphi = phi2 - phi1
    dlam = np.deg2rad(stop_lons)[None, :] - np.deg2rad(fix_lons)[:, None]
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class ScheduleIndex:
    """Pre-computed, purely static view of the planned timetable.

    The schedule never changes during inference, yet it used to be re-derived on
    *every* call: parse 5 558 WKT points, sort by ``(tr_id, time_begin, stop_id)``,
    compute ``cumcount``/inter-stop gaps and run a haversine per leg.  That was
    ~40% of the total request latency for zero information gain.

    Everything here is a function of the plan alone -- the actual-arrival column
    is not even present in this subset -- so hoisting it out of the request path
    cannot introduce leakage.  The heavy work happens once per process; per-request
    the object only serves a merge.

    Attributes
    ----------
    prepared : pandas.DataFrame
        Sorted schedule with ``stop_lon``/``stop_lat`` plus the static
        ``stop_idx``, ``route_stop_count``, ``planned_gap_prev_s``,
        ``planned_gap_next_s`` and ``leg_distance_km`` columns.
    by_vehicle : dict
        ``tr_id -> DataFrame`` slice, reused by the snapshot builder so it stops
        copying the whole schedule per forecast point.
    """

    __slots__ = ("prepared", "by_vehicle")

    def __init__(self, schedule: pd.DataFrame) -> None:
        frame = schedule.copy()
        frame["tr_id"] = pd.to_numeric(frame["tr_id"], errors="coerce")
        frame = frame.dropna(subset=["tr_id"])
        frame["tr_id"] = frame["tr_id"].astype("int64")
        frame["tt_action_item_id"] = pd.to_numeric(frame["tt_action_item_id"], errors="coerce")
        frame = frame.dropna(subset=["tt_action_item_id"])
        frame["tt_action_item_id"] = frame["tt_action_item_id"].astype("int64")
        frame["time_begin"] = pd.to_datetime(frame["time_begin"], errors="coerce")
        frame = frame.dropna(subset=["time_begin"])

        if "geom" in frame.columns:
            parsed = [parse_point_wkt(value) for value in frame["geom"]]
            frame["stop_lon"] = [item[0] for item in parsed]
            frame["stop_lat"] = [item[1] for item in parsed]
        else:
            frame["stop_lon"] = math.nan
            frame["stop_lat"] = math.nan

        frame = frame.sort_values(
            ["tr_id", "time_begin", "tt_action_item_id"], kind="stable"
        ).reset_index(drop=True)

        grouped = frame.groupby("tr_id", sort=False)
        frame["stop_idx"] = grouped.cumcount()
        frame["route_stop_count"] = grouped["tt_action_item_id"].transform("size")
        frame["planned_gap_prev_s"] = grouped["time_begin"].diff().dt.total_seconds()
        frame["planned_gap_next_s"] = grouped["time_begin"].diff(-1).dt.total_seconds()

        lons = frame["stop_lon"].to_numpy(dtype=float)
        lats = frame["stop_lat"].to_numpy(dtype=float)
        legs = np.full(len(frame), np.nan)
        tr_ids = frame["tr_id"].to_numpy()
        for tid in np.unique(tr_ids):
            mask = np.flatnonzero(tr_ids == tid)
            if len(mask) > 1:
                steps = haversine_km(
                    lons[mask[:-1]],
                    lats[mask[:-1]],
                    lons[mask[1:]],
                    lats[mask[1:]],
                )
                legs[mask[:-1]] = steps
        frame["leg_distance_km"] = legs

        self.prepared = frame
        self.by_vehicle = {int(tid): group for tid, group in frame.groupby("tr_id", sort=False)}


def schedule_index(schedule: Any) -> ScheduleIndex:
    """Return a :class:`ScheduleIndex` for ``schedule``, building it once.

    The index is memoised on the DataFrame through ``DataFrame.attrs`` so the
    caller does not have to thread it through every call site, and its lifetime
    is exactly the lifetime of the frame.
    """
    if isinstance(schedule, ScheduleIndex):
        return schedule
    cached = schedule.attrs.get(_SCHEDULE_INDEX_KEY) if hasattr(schedule, "attrs") else None
    if isinstance(cached, ScheduleIndex):
        return cached
    built = ScheduleIndex(schedule)
    if hasattr(schedule, "attrs"):
        try:
            schedule.attrs[_SCHEDULE_INDEX_KEY] = built
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            pass
    return built


def empty_schedule() -> pd.DataFrame:
    """Empty schedule with the columns callers index into."""
    return pd.DataFrame(
        columns=[
            "tt_action_item_id", "tr_id", "time_begin", "stop_lon", "stop_lat",
            "stop_idx", "route_stop_count", "planned_gap_prev_s",
            "planned_gap_next_s", "leg_distance_km",
        ]
    )
