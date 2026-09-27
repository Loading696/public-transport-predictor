"""Cascading delay propagation over the planned route network.

Purpose
-------
The ML regressor in :mod:`src.predictor` answers a *single-point* question:
"how late will vehicle ``tr_id`` be at stop ``target_stop_id``?".  Dispatchers
need a *network* question: "if this bus is 4 minutes late, which downstream
stops does it hit, which passengers miss their transfer, and do the other buses
on the same corridor recover or absorb it?".  This module answers that by
projecting the measured delay forward along the planned stop sequence and, in a
second stage, propagating the residual across intersecting routes.

The model
---------
The network is a directed graph whose vertices are **planned stop-visits**
(``tr_id`` x stop) and whose edges are **delay-transfer links**.  Every edge
``e = (u -> v)`` is the same one-parameter operator::

    T_e(D) = max(0, lambda_e * D - b_e) + n_v

with

``lambda_e``  *carry* -- fraction of the upstream delay that survives the edge.
``b_e``       *buffer* -- seconds of slack the vehicle can claw back on that
              edge before the delay is fully absorbed (the fleet's recovery
              capacity).
``n_v``       *local source* -- nuisance delay generated **at** ``v`` itself
              (dwell overrun, run-time degradation).

The solved delay field is the least fixed point of the max-plus system

.. math::

    D[v] = \\max\\Big( I[v],\\; \\max_{e=(u,v)} \\big(\\lambda_e D[u] - b_e + n_v\\big)^+ \\Big)

where ``I[v] >= 0`` is the *injection*: the delay actually measured (or predicted)
at ``v``.  The injection enters as a **floor**, not as an increment, because a
measurement supersedes anything the propagation would have produced.

Intra-route recursion (the one-dimensional backbone)
----------------------------------------------------
Along a route the continuation links form a chain, so the fixed point has the
closed form obtained by running the recursion forward.  With ``R_i`` the planned
run time of segment ``i`` and ``psi in [0, 1)`` the *recoverable share* of that
run time::

    D_{i} = max( I_i,  D_{i-1} - psi * R_i + omega_i )          omega_i = n_i
    D_0   = max( I_0,  0 )

Three consequences drive the design:

1. **Catch-up is automatic.**  The ``max(0, .)`` floor is the plan: a vehicle is
   never rewarded for arriving early, so the ``psi * R_i`` term burns the delay
   down until it reaches zero and then stops.  No special-case "recovery mode".
2. **Absorption / reach.**  Along a route the total recoverable buffer is
   ``B = sum_i psi * R_i``.  A delay ``D`` at the origin is *fully absorbed* at
   the first stop where the cumulative buffer exceeds ``D``; if ``D >= B`` the
   residual ``D - B`` survives to the end of the route.  This single comparison
   is what a dispatcher actually wants to know.
3. **Stability.**  Continuation links have ``lambda = 1`` but a strictly positive
   buffer, so a cycle needs at least one transfer link, and transfer links are
   built with ``lambda <= max_transfer_carry < 1``.  The composite operator of
   any cycle is therefore a contraction, the fixed point is unique, and the
   Gauss-Seidel sweep converges geometrically.  ``max_iterations`` is a safety
   net, not a tuning knob.

Cross-route coupling
--------------------
Routes are matched by **geometric corridor overlap**, not by stop id: in this
dataset ``tt_action_item_id`` is unique per stop-visit, so a shared physical
stop has a different id on every route.  Candidate pairs come from a lat/lon grid
(only the 3x3 neighbourhood of a cell is examined), then from the scheduled
connection test ``0 < plan(b, j) - plan(a, i) <= transfer_window_s``.  The carry
is scaled by the pair's corridor-overlap index, so two routes that merely cross
at one intersection couple weakly while two routes sharing a corridor couple
strongly.

Scaling the coupling to a full network
--------------------------------------
The vertex graph implemented here is the ``O(D)``-sized exact core.  A regional
deployment scales along three axes, all of which reuse the same edge operator:

* **Shared corridors** (implemented).  Contract corridors into corridor
  vertices; the coupling strength is the overlap index.  Reduces the graph to
  ``O(#corridors)``.
* **Time expansion.**  Replace each stop-visit vertex with ``(stop, time-slot)``
  and connect slots within the transfer window.  Captures queues at hubs and
  vehicle bunching; cost grows with the horizon but stays sparse.
* **Hierarchical levels.**  Solve leaf segments exactly, then pass boundary
  delays (delays at segment ends) to a coarser level.  Cost drops from ``O(V)``
  to ``O(leaves)`` while keeping corridor-level interactions.

Causality
---------
The propagation reads only ``time_begin`` (the **plan**) plus the delay already
predicted at ``T``.  ``time_fact_begin`` is never touched, so the module is safe
for the live and the validate contours alike.  The projection is *forward* in
time from ``T`` over a plan that is by definition known at ``T``.

Examples
--------
>>> from src.cascade import CascadeConfig, build_cascade_graph, solve_cascade
>>> graph = build_cascade_graph(schedule_df)            # doctest: +SKIP
>>> result = solve_cascade(graph, {(3, 14): 240.0})     # doctest: +SKIP
>>> result.summary()["absorbed_routes"]                 # doctest: +SKIP
7
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

EARTH_RADIUS_KM = 6371.0088
#: Mean meridional degree length; used to size the corridor-detection grid.
KM_PER_DEG_LAT = 111.32

#: Absolute tolerance (seconds) below which two delays are considered equal.
#: Guards against float noise re-triggering the Gauss-Seidel loop forever.
EPS_S = 1e-9

#: Operational regimes.  Congestion removes recovery capacity, so a single
#: ``regime`` knob flips the whole model between optimistic and pessimistic.
REGIMES: dict[str, dict[str, float]] = {
    # Plenty of slack: buses claw delay back quickly, contagion barely travels.
    "free": {
        "run_recovery": 0.25,
        "dwell_overrun_s": 0.0,
        "transfer_carry": 0.45,
        "transfer_window_s": 600.0,
    },
    # Default.  A few minutes of recoverable running time per segment.
    "normal": {
        "run_recovery": 0.15,
        "dwell_overrun_s": 5.0,
        "transfer_carry": 0.70,
        "transfer_window_s": 900.0,
    },
    # Gridlock: almost no recovery, late buses drag their neighbours.
    "congested": {
        "run_recovery": 0.05,
        "dwell_overrun_s": 20.0,
        "transfer_carry": 0.85,
        "transfer_window_s": 1500.0,
    },
}


def pairwise_haversine_km(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in kilometres between two equally shaped arrays.

    Parameters
    ----------
    lon1, lat1 : numpy.ndarray
        Longitudes / latitudes of the first point set, in degrees.
    lon2, lat2 : numpy.ndarray
        Longitudes / latitudes of the second point set, in degrees.

    Returns
    -------
    numpy.ndarray
        Pairwise distances in kilometres, same shape as the inputs.

    Notes
    -----
    Spherical approximation with the WGS-84 mean radius.  Over the ~30 km extent
    of an urban route the error versus the ellipsoid is well under 0.2 %, which is
    far below the noise of the corridor radius we compare it against.
    """
    lon1 = np.asarray(lon1, dtype=float)
    lat1 = np.asarray(lat1, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlam = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _finite(value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` to a finite float, falling back to ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _iso(moment: pd.Timestamp | None) -> str | None:
    """Format a timestamp as second-precision ISO-8601 (``None`` passes through).

    ETAs carry sub-second precision because they come from a float delay; the UI
    and the API both want whole seconds, and rounding here keeps every producer
    of an ETA consistent.
    """
    if moment is None:
        return None
    try:
        return pd.Timestamp(moment).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return str(moment)


def _clip(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CascadeConfig:
    """Tunable parameters of the cascading-delay model.

    The defaults correspond to the ``"normal"`` regime.  Every field can also be
    overridden individually; :meth:`for_regime` starts from a named profile.

    Attributes
    ----------
    regime : str
        Name of the operational regime used by :meth:`for_regime`.  Kept for
        traceability in the API payload.
    run_recovery : float
        ``psi`` -- share of the planned run time a vehicle may recover, i.e. how
        much faster than plan it can drive while still being "on time".  Bounded
        to ``[0, 0.9]``; the upper bound keeps the recursion strictly stable.
        Physically ``0.10-0.30`` for urban bus traffic: timetable padding plus
        slack in signal timing.
    dwell_overrun_s : float
        ``omega`` -- nuisance dwell overrun added at **every** projected stop.
        Zero when the caller wants a pure "same vehicle, no new incidents" view.
    transfer_carry : float
        ``lambda`` for cross-route edges: the share of a late bus's delay that a
        connecting service inherits.  Clamped to ``[0, 0.95]``; strictly below 1
        is what guarantees the fixed point exists.
    transfer_window_s : float
        Largest scheduled gap ``plan(b, j) - plan(a, i)`` that still counts as a
        practical transfer.  Beyond it the connection is not usable and no
        contagion is modelled.
    shared_stop_bonus : float
        Additive boost to ``transfer_carry`` when the two routes meet at a stop
        that appears in both schedules' *address* strings -- a cheap proxy for a
        genuine interchange rather than a kerbside overlap.
    corridor_radius_km : float
        Two stops closer than this are considered the same place for the purpose
        of building transfer links.
    min_run_s : float
        Floor for a planned run time.  Guards against zero/negative gaps in
        malformed schedule rows, which would otherwise create a negative buffer
        and make the recursion *amplify* delay.
    max_transfer_carry : float
        Hard ceiling applied to every transfer link after the shared-stop bonus,
        independent of ``transfer_carry``.
    max_links_per_pair : int
        Cap on transfer links kept between one ordered pair of routes.  Bounds
        the edge count in dense downtown grids.
    max_candidates_per_pair : int
        Cap on geometric candidate pairs examined per ordered route pair.
    max_iterations : int
        Safety cap on Gauss-Seidel sweeps.  The sweep converges geometrically
        (see the module docstring), so this is normally never reached.
    corridor_sample_stops : int
        Number of stops kept per (grid cell, route) when building the corridor
        index, so a cell shared by a terminus and 40 mid-route stops cannot
        dominate the candidate list.
    """

    regime: str = "normal"
    run_recovery: float = 0.15
    dwell_overrun_s: float = 5.0
    transfer_carry: float = 0.70
    transfer_window_s: float = 900.0
    shared_stop_bonus: float = 0.15
    corridor_radius_km: float = 0.35
    min_run_s: float = 20.0
    max_transfer_carry: float = 0.95
    max_links_per_pair: int = 6
    max_candidates_per_pair: int = 4000
    max_iterations: int = 64
    corridor_sample_stops: int = 6

    def __post_init__(self) -> None:
        # A frozen dataclass still allows object.__setattr__; normalise here so
        # every downstream computation can rely on the invariants.
        object.__setattr__(self, "run_recovery", _clip(_finite(self.run_recovery), 0.0, 0.9))
        object.__setattr__(self, "dwell_overrun_s", max(0.0, _finite(self.dwell_overrun_s)))
        object.__setattr__(self, "transfer_carry", _clip(_finite(self.transfer_carry), 0.0, 0.95))
        object.__setattr__(self, "transfer_window_s", max(0.0, _finite(self.transfer_window_s)))
        object.__setattr__(self, "corridor_radius_km", max(1e-3, _finite(self.corridor_radius_km, 0.35)))
        object.__setattr__(self, "min_run_s", max(1.0, _finite(self.min_run_s, 20.0)))
        object.__setattr__(
            self, "max_transfer_carry", _clip(_finite(self.max_transfer_carry, 0.95), 0.0, 0.95)
        )

    @classmethod
    def for_regime(cls, regime: str, **overrides: Any) -> CascadeConfig:
        """Build a config from a named regime with optional field overrides.

        Parameters
        ----------
        regime : {'free', 'normal', 'congested'}
            Profile name; unknown names fall back to ``"normal"``.
        **overrides
            Any :class:`CascadeConfig` field, applied after the profile.

        Returns
        -------
        CascadeConfig
            Fully normalised configuration.
        """
        profile = REGIMES.get(str(regime).lower(), REGIMES["normal"])
        base = dict(profile)
        base["regime"] = str(regime).lower()
        base.update(overrides)
        return cls(**base)

    def as_dict(self) -> dict[str, Any]:
        """Return the config as a JSON-ready dict (for API payloads)."""
        return {
            "regime": self.regime,
            "run_recovery": self.run_recovery,
            "dwell_overrun_s": self.dwell_overrun_s,
            "transfer_carry": self.transfer_carry,
            "transfer_window_s": self.transfer_window_s,
            "corridor_radius_km": self.corridor_radius_km,
            "max_transfer_carry": self.max_transfer_carry,
        }


# --------------------------------------------------------------------------- #
# Graph data structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RouteStop:
    """One planned stop-visit of one vehicle run -- a graph *vertex*.

    Attributes
    ----------
    vertex : int
        Dense index of this vertex inside :class:`CascadeGraph`.
    tr_id : int
        Vehicle / run identifier (the ``tr_id`` column of the schedule).
    stop_id : int
        ``tt_action_item_id`` of the planned arrival.  Note this id is unique per
        stop-**visit**, not per physical stop, so it cannot be used to detect
        shared interchanges across routes.
    position : int
        0-based position of the vertex within its route, in ascending plan order.
    plan_arrival : pandas.Timestamp
        Planned arrival (``time_begin``).
    plan_departure : pandas.Timestamp
        Planned departure.  Equals ``plan_arrival`` when the schedule carries no
        separate departure column, in which case dwell is folded into the
        inter-stop run time.
    run_planned_s : float
        ``R`` -- planned run time of the **inbound** segment (previous stop ->
        this one).  ``0.0`` for the first vertex of a route.
    buffer_s : float
        ``b = run_recovery * run_planned_s`` -- recoverable seconds available on
        the inbound segment.  This is the fleet's catch-up capacity.
    dwell_planned_s : float
        Planned dwell at this stop, ``plan_departure - plan_arrival``.
    lon, lat : float
        Stop coordinates parsed from the WKT ``geom`` column; ``nan`` if absent.
    address : str | None
        ``building_address`` if present, else ``None``.  Used only as a weak
        interchange hint.
    inbound : int
        Vertex index of the continuation source, or ``-1`` for the first stop.
    """

    vertex: int
    tr_id: int
    stop_id: int
    position: int
    plan_arrival: pd.Timestamp
    plan_departure: pd.Timestamp
    run_planned_s: float
    buffer_s: float
    dwell_planned_s: float
    lon: float
    lat: float
    address: str | None
    inbound: int


@dataclass(frozen=True, slots=True)
class DelayLink:
    """One directed delay-transfer edge -- a graph *edge*.

    Applying the link to an upstream delay ``D`` yields
    ``max(0, carry * D - buffer_s) + local_source``.

    Attributes
    ----------
    src, dst : int
        Source and destination vertex indices.
    kind : {'continuation', 'transfer'}
        ``'continuation'`` is the same vehicle moving to its next stop
        (``carry == 1``); ``'transfer'`` is cross-route coupling.
    carry : float
        ``lambda`` -- share of the upstream delay surviving the edge.
    buffer_s : float
        ``b`` -- seconds the receiving vehicle can absorb before losing the delay.
    local_source : float
        ``n`` -- nuisance delay generated at ``dst`` by traversing this edge.
    src_tr_id, dst_tr_id : int
        Route identifiers of the endpoints, denormalised for reporting.
    plan_gap_s : float
        Scheduled time between the source departure and the destination arrival.
        Meaningful for transfer links; ``run_planned_s`` for continuation.
    distance_km : float
        Geographic distance between the two stops (``nan`` when unknown).
    overlap : float
        Corridor-overlap index in ``[0, 1]`` for the route pair, ``1.0`` for
        continuation links.
    """

    src: int
    dst: int
    kind: str
    carry: float
    buffer_s: float
    local_source: float
    src_tr_id: int
    dst_tr_id: int
    plan_gap_s: float
    distance_km: float
    overlap: float

    def apply(self, delay_s: float) -> float:
        """Apply the edge operator to an upstream delay.

        Parameters
        ----------
        delay_s : float
            Upstream delay in seconds.

        Returns
        -------
        float
            Delay handed to ``dst``: ``max(0, carry * delay_s - buffer_s) +
            local_source``.
        """
        return max(0.0, self.carry * delay_s - self.buffer_s) + self.local_source


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #


def _parse_points(geom: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(lon, lat)`` arrays from a WKT ``geom`` column.

    Delegates to :func:`src.schedule_index.parse_point_wkt` so the cascade reads
    exactly the same coordinates as the feature builder (single source of truth
    for the POINT WKT grammar).  Unparseable entries become ``nan``.

    Imported lazily, and from the shared schedule module rather than from
    ``src.predictor``: the backend runs the cascade, and the backend must not
    pull in the ML module.
    """
    from src.schedule_index import parse_point_wkt

    parsed = [parse_point_wkt(value) for value in geom]
    if not parsed:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    lons = np.asarray([item[0] for item in parsed], dtype=float)
    lats = np.asarray([item[1] for item in parsed], dtype=float)
    return lons, lats


def _pick_departure_column(frame: pd.DataFrame) -> str | None:
    """Return the first column that plausibly holds the planned departure time.

    The hackathon schedule exposes only ``time_begin`` (arrival).  Real dispatch
    extracts usually add a departure column; when one is present the cascade uses
    it so that dwell is not silently folded into the run time.
    """
    for name in ("time_depart_begin", "time_departure", "time_depart", "departure"):
        if name in frame.columns:
            return name
    return None


def build_cascade_graph(
    schedule: pd.DataFrame,
    config: CascadeConfig | None = None,
) -> CascadeGraph:
    """Build the stop-visit graph with continuation and transfer links.

    Parameters
    ----------
    schedule : pandas.DataFrame
        A schedule table.  Required columns: ``tr_id``, ``tt_action_item_id``,
        ``time_begin``.  Optional: a departure column (see
        :func:`_pick_departure_column`), ``geom`` (POINT WKT) and
        ``building_address``.
    config : CascadeConfig, optional
        Model parameters; defaults to the ``"normal"`` regime.

    Returns
    -------
    CascadeGraph
        Immutable graph.  Routes are ordered by ascending ``time_begin`` with
        ``tt_action_item_id`` as a stable tie-break, which is what makes the
        chain recursion well defined even when a vehicle revisits a stop.

    Notes
    -----
    Complexity.  Continuation links are ``O(V)``.  Transfer links use a lat/lon
    grid: stops are bucketed by ``floor(lat / cell)`` and only the 3x3
    neighbourhood of each cell is examined, so candidate generation is
    ``O(V * neighbours)`` rather than ``O(V^2)`` -- important because a regional
    extract can hold hundreds of thousands of stop-visits.
    """
    cfg = config or CascadeConfig()
    if schedule is None or len(schedule) == 0:
        return CascadeGraph((), (), cfg, stats={"stops": 0, "routes": 0, "continuation_links": 0,
                                               "transfer_links": 0, "corridor_pairs": 0})

    frame = schedule.copy()
    frame["tr_id"] = pd.to_numeric(frame["tr_id"], errors="coerce")
    frame = frame.dropna(subset=["tr_id"])
    frame["tr_id"] = frame["tr_id"].astype("int64")
    frame["tt_action_item_id"] = pd.to_numeric(frame["tt_action_item_id"], errors="coerce")
    frame = frame.dropna(subset=["tt_action_item_id"])
    frame["tt_action_item_id"] = frame["tt_action_item_id"].astype("int64")
    frame["time_begin"] = pd.to_datetime(frame["time_begin"], errors="coerce")
    frame = frame.dropna(subset=["time_begin"])
    if frame.empty:
        return CascadeGraph((), (), cfg, stats={"stops": 0, "routes": 0, "continuation_links": 0,
                                               "transfer_links": 0, "corridor_pairs": 0})

    depart_col = _pick_departure_column(frame)
    if depart_col is not None:
        frame["_departure"] = pd.to_datetime(frame[depart_col], errors="coerce")
    else:
        # No departure column: the vehicle is treated as departing as it arrives,
        # so dwell is implicit in the inter-stop interval.
        frame["_departure"] = frame["time_begin"]
    frame["_departure"] = frame["_departure"].fillna(frame["time_begin"])
    if "geom" in frame.columns:
        lons, lats = _parse_points(frame["geom"])
    else:
        lons = np.full(len(frame), np.nan)
        lats = np.full(len(frame), np.nan)
    frame["_lon"] = lons
    frame["_lat"] = lats
    frame["_address"] = frame["building_address"] if "building_address" in frame.columns else None
    frame = frame.sort_values(["tr_id", "time_begin", "tt_action_item_id"], kind="stable")

    stops: list[RouteStop] = []
    continuation: list[DelayLink] = []
    route_vertices: dict[int, np.ndarray] = {}
    by_stop: dict[tuple[int, int], int] = {}
    vertex = 0
    for tr_id, group in frame.groupby("tr_id", sort=True):
        arrivals = group["time_begin"].tolist()
        departures = group["_departure"].tolist()
        stop_ids = group["tt_action_item_id"].tolist()
        lons_v = group["_lon"].to_numpy(dtype=float)
        lats_v = group["_lat"].to_numpy(dtype=float)
        addresses = group["_address"].tolist() if frame["_address"] is not None else [None] * len(group)
        indices: list[int] = []
        for position in range(len(group)):
            arrival = pd.Timestamp(arrivals[position])
            departure = pd.Timestamp(departures[position])
            if position == 0:
                run_planned = 0.0
                inbound = -1
            else:
                prev_departure = pd.Timestamp(departures[position - 1])
                run_planned = max(cfg.min_run_s, (arrival - prev_departure).total_seconds())
                inbound = indices[position - 1]
            stop = RouteStop(
                vertex=vertex,
                tr_id=int(tr_id),
                stop_id=int(stop_ids[position]),
                position=position,
                plan_arrival=arrival,
                plan_departure=departure,
                run_planned_s=float(run_planned),
                buffer_s=float(run_planned * cfg.run_recovery),
                dwell_planned_s=max(0.0, (departure - arrival).total_seconds()),
                lon=float(lons_v[position]),
                lat=float(lats_v[position]),
                address=None if addresses[position] is None else str(addresses[position]),
                inbound=inbound,
            )
            stops.append(stop)
            by_stop[(int(tr_id), int(stop_ids[position]))] = vertex
            indices.append(vertex)
            if inbound >= 0:
                continuation.append(
                    DelayLink(
                        src=inbound,
                        dst=vertex,
                        kind="continuation",
                        carry=1.0,
                        buffer_s=stop.buffer_s,
                        local_source=cfg.dwell_overrun_s,
                        src_tr_id=int(tr_id),
                        dst_tr_id=int(tr_id),
                        plan_gap_s=stop.run_planned_s,
                        distance_km=0.0,
                        overlap=1.0,
                    )
                )
            vertex += 1
        route_vertices[int(tr_id)] = np.asarray(indices, dtype=np.int64)

    transfers, corridor_pairs = _build_transfer_links(stops, route_vertices, cfg)

    stats = {
        "stops": len(stops),
        "routes": len(route_vertices),
        "continuation_links": len(continuation),
        "transfer_links": len(transfers),
        "corridor_pairs": corridor_pairs,
        "vertices_without_geometry": int(
            sum(1 for stop in stops if not (math.isfinite(stop.lon) and math.isfinite(stop.lat)))
        ),
    }
    return CascadeGraph(
        tuple(stops),
        tuple(continuation) + tuple(transfers),
        cfg,
        route_vertices=route_vertices,
        by_stop=by_stop,
        stats=stats,
    )


def _build_transfer_links(
    stops: Sequence[RouteStop],
    route_vertices: Mapping[int, np.ndarray],
    cfg: CascadeConfig,
) -> tuple[list[DelayLink], int]:
    """Create cross-route transfer links from geometric corridor overlap.

    The routine has four stages:

    1. **Grid bucketing.**  Every geocoded stop is placed in a lat/lon cell of
       side ``corridor_radius_km``.  Stops per (cell, route) are capped at
       ``corridor_sample_stops`` so a terminus cannot monopolise a cell.
    2. **Neighbourhood gather.**  For each cell of each route, the 3x3
       neighbourhood is collected *once* and grouped by route.  This is the step
       that keeps the routine near-linear: without it, every stop would rescan
       every neighbouring cell for every route.
    3. **Vectorised filtering.**  For one (route_a, route_b, cell) block the
       haversine distances and the scheduled connection test
       ``0 < plan(b, j) - plan(a, i) <= transfer_window_s`` are evaluated as a
       single numpy block, then masked.  A bus arriving an hour before the
       connecting service cannot hold it up, hence the lower bound on the gap.
    4. **Selection.**  Candidates of an ordered route pair are de-duplicated,
       ranked by (shared address, smallest distance) and truncated to
       ``max_links_per_pair``.  The survivors' corridor-overlap index rescales
       their carry, so routes sharing a corridor couple far harder than routes
       that merely cross at one intersection.

    Parameters
    ----------
    stops : sequence of RouteStop
        All vertices of the graph.
    route_vertices : mapping
        ``tr_id -> ordered vertex indices``.
    cfg : CascadeConfig
        Model parameters.

    Returns
    -------
    (list of DelayLink, int)
        The transfer links and the number of intersecting route pairs detected.

    Notes
    -----
    Cost is ``O(sum over cells of |neighbourhood|)`` dictionary work plus one
    small numpy block per (cell, route pair) -- not ``O(V^2)``.  The remaining
    risk is a pair of routes that occupy exactly the same cells; the block then
    spans a whole route pair at once (``n_a x n_b`` booleans), which is still
    vectorised and, at 689 stops per route, a few hundred kilobytes.
    """
    cell_deg = max(1e-4, cfg.corridor_radius_km / KM_PER_DEG_LAT)
    size = len(stops)
    lons = np.asarray([stop.lon for stop in stops], dtype=float)
    lats = np.asarray([stop.lat for stop in stops], dtype=float)
    # Plan departures as int64 nanoseconds so connection gaps are a vector
    # subtraction instead of a Python loop over Timestamps.
    departures = np.asarray(
        [stop.plan_departure.value for stop in stops], dtype=np.int64
    )
    addresses = [
        " ".join(stop.address.lower().split()) if stop.address else "" for stop in stops
    ]
    geocoded = np.isfinite(lons) & np.isfinite(lats)
    window_ns = int(cfg.transfer_window_s * 1e9)

    # 1. grid bucketing, per route, plus a cell -> routes index so the
    # neighbourhood gather below is a single dict lookup per cell.
    cells_by_route: dict[int, dict[tuple[int, int], list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cell_index: dict[tuple[int, int], dict[int, list[int]]] = defaultdict(dict)
    for vertex in range(size):
        if not geocoded[vertex]:
            continue
        cell = (int(math.floor(lats[vertex] / cell_deg)), int(math.floor(lons[vertex] / cell_deg)))
        route = stops[vertex].tr_id
        bucket = cells_by_route[route][cell]
        if len(bucket) < cfg.corridor_sample_stops:
            bucket.append(vertex)
            cell_index[cell].setdefault(route, []).append(vertex)

    # 2 + 3 + 4: neighbourhood gather, vectorised filter, per-pair selection.
    # best[(route_a, route_b)][(src, dst)] = (shared_address, distance_km, gap_s)
    best: dict[tuple[int, int], dict[tuple[int, int], tuple[bool, float, float]]] = defaultdict(dict)
    for route_a, cells_a in cells_by_route.items():
        for cell, verts_a in cells_a.items():
            neighbours: dict[int, list[int]] = defaultdict(list)
            for dlat in (-1, 0, 1):
                for dlon in (-1, 0, 1):
                    target = (cell[0] + dlat, cell[1] + dlon)
                    for route_b, verts_b in cell_index.get(target, {}).items():
                        if route_b != route_a:
                            neighbours[route_b].extend(verts_b)
            if not neighbours:
                continue
            arr_a = np.asarray(verts_a, dtype=np.int64)
            for route_b, verts_b in neighbours.items():
                arr_b = np.asarray(sorted(set(verts_b)), dtype=np.int64)
                # Explicit [None, :] / [:, None] so that single-element blocks
                # still broadcast to a 2-D matrix instead of staying 1-D.
                distance = pairwise_haversine_km(
                    lons[arr_a][:, None], lats[arr_a][:, None],
                    lons[arr_b][None, :], lats[arr_b][None, :],
                )
                gap_dep = departures[arr_b][None, :] - departures[arr_a][:, None]
                mask = (distance <= cfg.corridor_radius_km) & (gap_dep > 0) & (gap_dep <= window_ns)
                if not mask.any():
                    continue
                bucket = best[(route_a, route_b)]
                for i, j in zip(*np.nonzero(mask)):
                    if len(bucket) >= cfg.max_candidates_per_pair:
                        # Two identical corridors would otherwise materialise an
                        # n_a x n_b dict.  The cap bounds it; the ranking below
                        # then keeps the most relevant subset.
                        break
                    src = int(arr_a[i])
                    dst = int(arr_b[j])
                    shared = bool(addresses[src]) and addresses[src] == addresses[dst]
                    key = (src, dst)
                    previous = bucket.get(key)
                    score = (shared, -float(distance[i, j]), float(gap_dep[i, j]) / 1e9)
                    if previous is None or score > previous:
                        bucket[key] = (shared, float(distance[i, j]), float(gap_dep[i, j]) / 1e9)

    links: list[DelayLink] = []
    for (route_a, route_b), bucket in sorted(best.items()):
        if not bucket:
            continue
        ranked = sorted(bucket.items(), key=lambda item: (item[1][0], item[1][1]))[
            : cfg.max_links_per_pair
        ]
        overlap = _overlap_index(route_vertices[route_a], route_vertices[route_b], stops)
        for (src_vertex, dst_vertex), (shared, distance, gap_s) in ranked:
            dst_stop = stops[dst_vertex]
            carry = cfg.transfer_carry * (0.5 + 0.5 * overlap)
            if shared:
                carry += cfg.shared_stop_bonus
            carry = min(cfg.max_transfer_carry, _clip(carry, 0.0, cfg.max_transfer_carry))
            if carry <= 0.0:
                continue
            # The connecting service absorbs delay only up to its own catch-up
            # capacity on the inbound segment, and never beyond the slack the
            # timetable itself leaves in the connection.
            buffer_s = max(0.0, min(dst_stop.buffer_s, gap_s * cfg.run_recovery))
            links.append(
                DelayLink(
                    src=src_vertex,
                    dst=dst_vertex,
                    kind="transfer",
                    carry=carry,
                    buffer_s=buffer_s,
                    local_source=0.0,
                    src_tr_id=route_a,
                    dst_tr_id=route_b,
                    plan_gap_s=gap_s,
                    distance_km=distance,
                    overlap=overlap,
                )
            )
    return links, len(best)


def _overlap_index(verts_a: np.ndarray, verts_b: np.ndarray, stops: Sequence[RouteStop]) -> float:
    """Corridor-overlap index in ``[0, 1]`` for a pair of routes.

    The index is the Jaccard similarity of the two routes' occupied ``0.1 deg``
    grid cells.  ``1.0`` means the two routes are geographically interleaved
    (shared corridor), ``0.0`` means they only graze each other.  It rescales the
    carry of transfer links so that a bus sharing a corridor with a late bus drags
    its neighbours much harder than one that merely crosses at an intersection.
    """
    if len(verts_a) == 0 or len(verts_b) == 0:
        return 0.0

    def cells(verts: np.ndarray) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for vertex in verts:
            stop = stops[int(vertex)]
            if math.isfinite(stop.lon) and math.isfinite(stop.lat):
                out.add((int(round(stop.lat * 10.0)), int(round(stop.lon * 10.0))))
        return out

    left = cells(verts_a)
    right = cells(verts_b)
    if not left or not right:
        return 0.0
    union = len(left | right)
    return float(len(left & right) / union) if union else 0.0


# --------------------------------------------------------------------------- #
# Graph container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CascadeGraph:
    """Immutable stop-visit graph plus the indices needed by the solver.

    Attributes
    ----------
    stops : tuple of RouteStop
        Vertices, grouped by route and ordered by plan time within a route.
    links : tuple of DelayLink
        All edges: continuation links first, then transfer links.
    config : CascadeConfig
        Parameters the graph was built with.
    route_vertices : mapping
        ``tr_id -> numpy.ndarray`` of vertex indices in route order.
    by_stop : mapping
        ``(tr_id, stop_id) -> vertex index`` for O(1) injection lookup.
    stats : dict
        Build-time counters (vertex/edge counts, corridor pairs, ...).
    out_links, in_links : mapping
        Adjacency indexes ``vertex -> list[DelayLink]``, filled in ``__post_init__``.
    transfer_links : tuple of DelayLink
        Subset of :attr:`links` with ``kind == 'transfer'``; the solver relaxes
        exactly this set, so it is materialised once.
    """

    stops: tuple[RouteStop, ...]
    links: tuple[DelayLink, ...]
    config: CascadeConfig
    route_vertices: Mapping[int, np.ndarray] = field(default_factory=dict)
    by_stop: Mapping[tuple[int, int], int] = field(default_factory=dict)
    stats: Mapping[str, Any] = field(default_factory=dict)
    out_links: Mapping[int, tuple[DelayLink, ...]] = field(default_factory=dict)
    in_links: Mapping[int, tuple[DelayLink, ...]] = field(default_factory=dict)
    transfer_links: tuple[DelayLink, ...] = ()

    def __post_init__(self) -> None:
        outgoing: dict[int, list[DelayLink]] = defaultdict(list)
        incoming: dict[int, list[DelayLink]] = defaultdict(list)
        transfers: list[DelayLink] = []
        for link in self.links:
            outgoing[link.src].append(link)
            incoming[link.dst].append(link)
            if link.kind == "transfer":
                transfers.append(link)
        object.__setattr__(self, "out_links", {key: tuple(value) for key, value in outgoing.items()})
        object.__setattr__(self, "in_links", {key: tuple(value) for key, value in incoming.items()})
        object.__setattr__(self, "transfer_links", tuple(transfers))

    # -- lookups ----------------------------------------------------------- #

    def vertex_for(self, tr_id: int, stop_id: int) -> int:
        """Return the vertex index for ``(tr_id, stop_id)``, or ``-1`` if absent.

        Tolerant of unhashable / non-numeric input: a malformed ``tr_id`` from a
        live NDTP unit must never raise inside the request path.
        """
        try:
            return int(self.by_stop.get((int(tr_id), int(stop_id)), -1))
        except (TypeError, ValueError):
            return -1

    def has_route(self, tr_id: int) -> bool:
        """Return ``True`` when ``tr_id`` exists in the graph."""
        try:
            return int(tr_id) in self.route_vertices
        except (TypeError, ValueError):
            return False

    def stop(self, vertex: int) -> RouteStop | None:
        """Return the :class:`RouteStop` at ``vertex``, or ``None`` if out of range."""
        if 0 <= int(vertex) < len(self.stops):
            return self.stops[int(vertex)]
        return None

    def network(self) -> dict[str, Any]:
        """Return a JSON-ready description of the graph structure."""
        return {
            **dict(self.stats),
            "config": self.config.as_dict(),
            "injected_vertices_capable": len(self.by_stop),
        }


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #


def propagate_chain(
    delay_in: float,
    buffers: np.ndarray,
    sources: np.ndarray,
    injections: np.ndarray | None = None,
) -> np.ndarray:
    """Run the exact one-dimensional recursion along a route.

    Implements, for ``i = 1 .. n-1``::

        D[i] = max( I[i],  D[i-1] - buffers[i] + sources[i] )
        D[0] = max( I[0],  max(0, delay_in) )

    Parameters
    ----------
    delay_in : float
        Delay already carried into the first vertex (seconds).
    buffers : numpy.ndarray
        ``b`` per vertex: recoverable seconds on the **inbound** segment.
        ``buffers[0]`` is ignored.
    sources : numpy.ndarray
        ``n`` per vertex: nuisance delay generated at the vertex.
    injections : numpy.ndarray, optional
        ``I`` per vertex: measured/predicted delay acting as a lower bound.

    Returns
    -------
    numpy.ndarray
        Delay field ``D`` for the whole chain.

    Notes
    -----
    This is a forward pass, so it is exact and ``O(n)`` -- no fixed-point
    iteration is needed inside a route.  The recursion is monotone in
    ``delay_in``, in ``sources`` and in ``injections``, and non-increasing in
    ``buffers``; those three monotonicities are what make the graph-level
    Gauss-Seidel sweep converge.

    The ``max(0, .)`` is applied once at the start: ``buffers`` is non-negative
    and ``sources`` non-negative, so a delay that has been fully absorbed can
    never be resurrected by a later stop.
    """
    buffers = np.asarray(buffers, dtype=float)
    sources = np.asarray(sources, dtype=float)
    count = buffers.shape[0]
    out = np.zeros(count, dtype=float)
    if count == 0:
        return out
    current = max(0.0, _finite(delay_in))
    inj = np.zeros(count, dtype=float) if injections is None else np.asarray(injections, dtype=float)
    for index in range(count):
        if index > 0:
            current = current - buffers[index] + sources[index]
        if current < 0.0:
            current = 0.0
        measured = inj[index]
        if measured > current:
            current = measured
        out[index] = current
    return out


def _sweep(
    graph: CascadeGraph,
    delay: np.ndarray,
    injections: np.ndarray,
    overruns: np.ndarray,
) -> int:
    """Perform one Gauss-Seidel sweep: route chains first, then transfer links.

    Parameters
    ----------
    graph : CascadeGraph
        Graph providing the chains and the transfer edges.
    delay : numpy.ndarray
        Delay field, updated in place.
    injections : numpy.ndarray
        Measured delays acting as per-vertex floors.
    overruns : numpy.ndarray
        Per-vertex nuisance source overriding the config default (``nan`` = use
        the edge's ``local_source``).

    Returns
    -------
    int
        Number of vertices whose delay actually increased -- the sweep's
        convergence signal.
    """
    changed = 0
    # Stage 1: exact chain propagation within every route.  Chains are
    # independent, so any order is correct.
    for vertices in graph.route_vertices.values():
        if len(vertices) < 2:
            continue
        buffers = np.asarray([graph.stops[int(v)].buffer_s for v in vertices], dtype=float)
        sources = np.asarray(
            [_vertex_source(graph, int(v), overruns) for v in vertices], dtype=float
        )
        chain = propagate_chain(
            delay_in=delay[vertices[0]],
            buffers=buffers,
            sources=sources,
            injections=injections[vertices],
        )
        for position, vertex in enumerate(vertices):
            new_value = chain[position]
            if new_value > delay[vertex] + EPS_S:
                delay[vertex] = new_value
                changed += 1
    # Stage 2: cross-route relaxation.  Each transfer link can raise its
    # destination, which the next sweep propagates downstream along the chain.
    for link in graph.transfer_links:
        upstream = delay[link.src]
        if upstream <= EPS_S:
            continue
        candidate = link.carry * upstream - link.buffer_s
        if candidate > delay[link.dst] + EPS_S:
            delay[link.dst] = candidate
            changed += 1
    # A transfer link raises a vertex but must not lift the measured floor.
    np.maximum(delay, injections, out=delay)
    return changed


def _vertex_source(graph: CascadeGraph, vertex: int, overruns: np.ndarray) -> float:
    """Nuisance delay source for a vertex: explicit override or config default."""
    override = float(overruns[vertex])
    if math.isfinite(override):
        return max(0.0, override)
    inbound = graph.stops[vertex].inbound
    if inbound < 0:
        return 0.0
    for link in graph.in_links.get(vertex, ()):  # pragma: no branch - tiny loop
        if link.kind == "continuation":
            return max(0.0, link.local_source)
    return 0.0


def _resolve_vertex(graph: CascadeGraph, key: Any) -> int:
    """Resolve an injection/override key to a vertex index, or ``-1``.

    Accepts a bare vertex index or a ``(tr_id, stop_id)`` pair.  Anything else --
    a string, a float, a tuple of the wrong arity -- yields ``-1`` so that a
    malformed caller is skipped instead of raising inside the request path.
    """
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        return int(key)
    if isinstance(key, tuple) and len(key) == 2:
        return graph.vertex_for(key[0], key[1])
    return -1


def solve_cascade(
    graph: CascadeGraph,
    injections: Mapping[Any, float] | None = None,
    overruns: Mapping[Any, float] | None = None,
    config: CascadeConfig | None = None,
) -> CascadeResult:
    """Solve the cascading-delay field over the whole graph.

    Parameters
    ----------
    graph : CascadeGraph
        Graph built by :func:`build_cascade_graph`.
    injections : mapping, optional
        Measured delays.  Keys are either ``(tr_id, stop_id)`` pairs or vertex
        indices; anything else is ignored.  Values are seconds; negative values
        are clamped to 0.  This is where the ML predictions from
        :mod:`src.runtime` enter the model.
    overruns : mapping, optional
        Per-vertex nuisance sources (dwell overruns) in seconds, same key
        conventions.  Defaults to ``config.dwell_overrun_s`` at every stop.
    config : CascadeConfig, optional
        Overrides ``graph.config`` for this solve only.  Note that the
        recoverable buffer ``b`` is baked into the graph at build time, so a
        different ``run_recovery`` here has no effect; build a new graph instead.

    Returns
    -------
    CascadeResult
        Delay field, ETAs, contagion split and absorption diagnostics.

    Notes
    -----
    Convergence.  Continuation edges are a chain per route (DAG) and are solved
    exactly; transfer edges have ``carry < 1`` and positive buffers, so the
    composite operator around any cycle is a contraction.  The Gauss-Seidel
    sweep therefore converges geometrically with ratio bounded by
    ``max_transfer_carry``; ``config.max_iterations`` is only a safety net.
    """
    cfg = config or graph.config
    size = len(graph.stops)
    injected = np.zeros(size, dtype=float)
    for key, value in (injections or {}).items():
        vertex = _resolve_vertex(graph, key)
        if 0 <= vertex < size:
            injected[vertex] = max(injected[vertex], max(0.0, _finite(value)))
    override = np.full(size, np.nan, dtype=float)
    for key, value in (overruns or {}).items():
        vertex = _resolve_vertex(graph, key)
        if 0 <= vertex < size:
            override[vertex] = max(0.0, _finite(value))

    delay = injected.copy()
    iterations = 0
    if size:
        for iterations in range(1, max(1, int(cfg.max_iterations)) + 1):
            if _sweep(graph, delay, injected, override) == 0:
                break
    # Reference solution without cross-route edges.  Differencing it against the
    # full field separates "my own delay propagated along my route" from "another
    # vehicle dragged me in" -- the two need very different dispatch reactions.
    own = _chain_only_solution(graph, injected, override)
    return CascadeResult(
        graph=graph,
        config=cfg,
        delay=delay,
        injected=injected,
        overruns=override,
        iterations=iterations,
        own_delay=own,
    )


def _chain_only_solution(
    graph: CascadeGraph,
    injections: np.ndarray,
    overruns: np.ndarray,
) -> np.ndarray:
    """Solve the graph ignoring transfer links (pure intra-route propagation).

    Used as the reference against which cross-route contagion is measured.
    """
    size = len(graph.stops)
    own = np.zeros(size, dtype=float)
    for vertices in graph.route_vertices.values():
        if len(vertices) == 0:
            continue
        buffers = np.asarray([graph.stops[int(v)].buffer_s for v in vertices], dtype=float)
        sources = np.asarray(
            [_vertex_source(graph, int(v), overruns) for v in vertices], dtype=float
        )
        own[vertices] = propagate_chain(
            delay_in=0.0, buffers=buffers, sources=sources, injections=injections[vertices]
        )
    return own


@dataclass(frozen=True, slots=True)
class CascadeResult:
    """Outcome of a :func:`solve_cascade` call.

    Attributes
    ----------
    graph : CascadeGraph
        The graph that was solved.
    config : CascadeConfig
        Parameters used.
    delay : numpy.ndarray
        ``D[v]`` -- total predicted delay in seconds at every vertex.
    injected : numpy.ndarray
        ``I[v]`` -- the measured/predicted delay supplied by the caller.
    overruns : numpy.ndarray
        Per-vertex nuisance source actually used (``nan`` = config default).
    iterations : int
        Gauss-Seidel sweeps performed.
    own_delay : numpy.ndarray
        ``D_own[v]`` -- the delay the vehicle would carry on its own, solved with
        the transfer links switched off.  ``delay - own_delay`` is exactly the
        part contributed by *other* vehicles.
    """

    graph: CascadeGraph
    config: CascadeConfig
    delay: np.ndarray
    injected: np.ndarray
    overruns: np.ndarray
    iterations: int
    own_delay: np.ndarray

    # -- derived fields ---------------------------------------------------- #

    @property
    def contagion(self) -> np.ndarray:
        """Delay contributed by *other* vehicles, ``max(0, D - D_own)``.

        Deliberately **not** ``D - I``: a bus that is 5 minutes late on its own
        and still 5 minutes late three stops later has suffered no contagion.
        Differencing against the transfer-free solution is what makes
        :meth:`contaminated_routes` mean "was dragged in by someone else".
        """
        return np.maximum(0.0, self.delay - self.own_delay)

    def route_delay(self, tr_id: int) -> np.ndarray:
        """Delay field of one route, in route order."""
        vertices = self.graph.route_vertices.get(int(tr_id))
        if vertices is None:
            return np.empty(0, dtype=float)
        return self.delay[vertices]

    def route_projection(
        self, tr_id: int, horizon: int = 10, from_injection: bool = True
    ) -> list[dict[str, Any]]:
        """Per-stop ETA projection for one route.

        Parameters
        ----------
        tr_id : int
            Route identifier.
        horizon : int
            Maximum number of stops to return.
        from_injection : bool
            Start at the first stop that carries a measured/predicted delay
            (default) rather than at the route origin.  For a dispatcher the
            interesting window starts where the bus is late, not at the depot.

        Returns
        -------
        list of dict
            One entry per stop: ``tr_id``, ``stop_id``, ``position``,
            ``plan_time`` / ``eta`` (ISO-8601, second precision), ``delay_s``,
            ``injected_s``, ``own_delay_s``, ``contagion_s`` (delay from other
            vehicles), ``recovered_s`` (buffer burned on the inbound segment) and
            ``address``.
        """
        vertices = self.graph.route_vertices.get(int(tr_id))
        if vertices is None:
            return []
        start = 0
        if from_injection:
            injected_positions = np.flatnonzero(self.injected[vertices] > EPS_S)
            if injected_positions.size:
                start = int(injected_positions[0])
        rows: list[dict[str, Any]] = []
        for vertex in vertices[start : start + max(0, int(horizon))]:
            stop = self.graph.stops[int(vertex)]
            delay = float(self.delay[int(vertex)])
            eta = stop.plan_arrival + pd.Timedelta(seconds=delay)
            rows.append(
                {
                    "tr_id": stop.tr_id,
                    "stop_id": stop.stop_id,
                    "position": stop.position,
                    "address": stop.address,
                    "plan_time": _iso(stop.plan_arrival),
                    "eta": _iso(eta),
                    "delay_s": round(delay, 3),
                    "injected_s": round(float(self.injected[int(vertex)]), 3),
                    "own_delay_s": round(float(self.own_delay[int(vertex)]), 3),
                    "contagion_s": round(
                        max(0.0, delay - float(self.own_delay[int(vertex)])), 3
                    ),
                    "recovered_s": round(stop.buffer_s, 3),
                }
            )
        return rows

    def absorption(self, tr_id: int) -> dict[str, Any]:
        """Absorption diagnostics for one route.

        Walks the route forward from the first injected stop, accumulating the
        recoverable buffer ``B += psi * R`` and stopping at the first vertex whose
        delay has reached zero -- the *absorbing stop*.  Then:

        * ``absorbed`` -- a zero was reached (or the route ends on time): the
          fleet claws the delay back and passengers downstream are unaffected;
        * ``residual_at_end_s`` -- otherwise, the delay that survives the whole
          route and lands on the last stop.

        Comparing the injected delay against the total buffer is a single
        comparison that a dispatcher can act on, which is why it is surfaced
        separately from the per-stop projection.

        Returns
        -------
        dict
            ``tr_id``, ``origin_stop_id``, ``delay_s``, ``recoverable_s``,
            ``absorbed`` (bool), ``absorbing_stop_id`` (``None`` if the delay
            never reaches zero), ``residual_at_end_s``, ``max_delay_s`` and
            ``max_delay_stop_id``.
        """
        vertices = self.graph.route_vertices.get(int(tr_id))
        payload: dict[str, Any] = {
            "tr_id": int(tr_id),
            "origin_stop_id": None,
            "delay_s": 0.0,
            "recoverable_s": 0.0,
            "absorbed": True,
            "absorbing_stop_id": None,
            "residual_at_end_s": 0.0,
            "max_delay_s": 0.0,
            "max_delay_stop_id": None,
        }
        if vertices is None or len(vertices) == 0:
            return payload
        injected_positions = np.flatnonzero(self.injected[vertices] > EPS_S)
        if injected_positions.size == 0:
            return payload
        start = int(injected_positions[0])
        origin = self.graph.stops[int(vertices[start])]
        buffer_total = 0.0
        absorbing: int | None = None
        for position in range(start + 1, len(vertices)):
            vertex = int(vertices[position])
            buffer_total += self.graph.stops[vertex].buffer_s
            if absorbing is None and self.delay[vertex] <= EPS_S:
                absorbing = vertex
        tail = float(self.delay[vertices[-1]])
        peak_vertex = int(vertices[int(np.argmax(self.delay[vertices[start:]])) + start])
        payload.update(
            {
                "origin_stop_id": origin.stop_id,
                "delay_s": round(float(self.injected[origin.vertex]), 3),
                "recoverable_s": round(float(buffer_total), 3),
                "absorbed": bool(absorbing is not None or tail <= EPS_S),
                "absorbing_stop_id": (
                    None if absorbing is None else self.graph.stops[absorbing].stop_id
                ),
                "residual_at_end_s": round(max(0.0, tail), 3),
                "max_delay_s": round(float(self.delay[peak_vertex]), 3),
                "max_delay_stop_id": self.graph.stops[peak_vertex].stop_id,
            }
        )
        return payload

    def contaminated_routes(self) -> list[int]:
        """Routes dragged off their own trajectory by *other* vehicles.

        A route qualifies when any of its stops carries a
        :attr:`contagion` strictly above zero, i.e. ``D`` exceeds what the
        vehicle would have achieved on its own with the same injections.
        """
        return sorted(
            int(tr_id)
            for tr_id, vertices in self.graph.route_vertices.items()
            if np.any(self.delay[vertices] - self.own_delay[vertices] > EPS_S)
        )

    def summary(self) -> dict[str, Any]:
        """Network-level JSON-ready summary of the cascade.

        Returns
        -------
        dict
            Graph stats, solver iterations, the number of routes with a measured
            injection, the list of routes contaminated by upstream delays, the
            share of vehicles that fully absorb their delay, and the per-route
            absorption payload.
        """
        injected_routes = [
            int(tr_id)
            for tr_id, vertices in self.graph.route_vertices.items()
            if np.any(self.injected[vertices] > EPS_S)
        ]
        absorption = [self.absorption(tr_id) for tr_id in injected_routes]
        absorbed = [row for row in absorption if row["absorbed"]]
        return {
            **dict(self.graph.stats),
            "iterations": int(self.iterations),
            "config": self.config.as_dict(),
            "injected_routes": injected_routes,
            "contaminated_routes": self.contaminated_routes(),
            "absorbed_routes": len(absorbed),
            "absorbed_ratio": round(len(absorbed) / len(absorption), 4) if absorption else None,
            "max_delay_s": round(float(np.max(self.delay)) if self.delay.size else 0.0, 3),
            "total_contagion_s": round(float(np.sum(self.contagion)), 3),
            "absorption": absorption,
        }


# --------------------------------------------------------------------------- #
# Single-route projection + engine
# --------------------------------------------------------------------------- #


def project_route_eta(
    graph: CascadeGraph,
    tr_id: int,
    stop_id: int,
    delay_s: float,
    horizon: int = 8,
    dwell_overrun_s: float | None = None,
) -> list[dict[str, Any]]:
    """Project one vehicle's delay forward along its own route, without a solve.

    This is the cheap path used by ``POST /predict``: a single vehicle cannot be
    contaminated by anything the caller has not told us about, so the exact chain
    recursion is sufficient and ``O(horizon)``.  Use :func:`solve_cascade` when
    the fleet's mutual coupling matters.

    Parameters
    ----------
    graph : CascadeGraph
        Graph holding the route.
    tr_id : int
        Vehicle to project.
    stop_id : int
        Stop the measured delay belongs to.  Projection starts at this stop.
    delay_s : float
        Delay at ``stop_id`` in seconds.  Negative values are clamped to 0.
    horizon : int
        Number of stops to emit, starting at ``stop_id``.
    dwell_overrun_s : float, optional
        Per-stop nuisance dwell added from ``stop_id`` onwards.  Defaults to
        ``config.dwell_overrun_s``.

    Returns
    -------
    list of dict
        Same shape as :meth:`CascadeResult.route_projection`.  Empty when the
        vehicle or the stop is unknown, or the delay is not positive -- this
        function never raises, because it sits in the request path.
    """
    vertices = graph.route_vertices.get(int(tr_id)) if _is_int(tr_id) else None
    if vertices is None:
        return []
    origin = graph.vertex_for(tr_id, stop_id)
    if origin < 0 or origin not in set(int(v) for v in vertices):
        return []
    delay = max(0.0, _finite(delay_s))
    if delay <= 0.0:
        return []
    order = list(int(v) for v in vertices)
    begin = order.index(origin)
    tail = order[begin : begin + max(1, int(horizon))]
    overrun = graph.config.dwell_overrun_s if dwell_overrun_s is None else max(0.0, _finite(dwell_overrun_s))
    rows: list[dict[str, Any]] = []
    current = delay
    for position, vertex in enumerate(tail):
        stop = graph.stops[vertex]
        if position > 0:
            current = max(0.0, current - stop.buffer_s + overrun)
        eta = stop.plan_arrival + pd.Timedelta(seconds=current)
        rows.append(
            {
                "tr_id": stop.tr_id,
                "stop_id": stop.stop_id,
                "position": stop.position,
                "address": stop.address,
                "plan_time": _iso(stop.plan_arrival),
                "eta": _iso(eta),
                "delay_s": round(current, 3),
                "recovered_s": round(stop.buffer_s, 3),
                "is_target": vertex == origin,
            }
        )
    return rows


def route_transfer_matrix(result: CascadeResult, tr_id: int) -> np.ndarray:
    """Elasticity of downstream ETAs w.r.t. an injection at each route stop.

    The recursion is piecewise linear, so its Jacobian is available in closed
    form.  The map at vertex ``j`` is
    ``D[j] = max(I[j], max(0, D[j-1] - b[j] + n[j]))`` and its *right* derivative
    w.r.t. ``D[j-1]`` is

    .. code-block:: text

        slope[j] = 1  if  D[j-1] - b[j] + n[j] >= max(0, I[j])   (delay-carrying)
                 = 0  otherwise                                    (masked/absorbed)

    so ``M[i, j] = dD[j] / dI[i] = prod_{k=i+1..j} slope[k]``.  The right
    derivative is the operationally meaningful one: at a kink the question is
    "what happens if the vehicle gets *more* late", and a vehicle on the plan
    with ``b = 0`` does carry an infinitesimal delay forward.

    Since ``slope`` is 0/1 valued, the product over ``(i, j]`` is 1 exactly when
    no zero sits in that window, i.e. when ``i >= lastzero(j)`` with
    ``lastzero(j) = max{k <= j : slope[k] = 0}``:

    .. code-block:: text

        M[i, j] = 1  if  i >= lastzero(j)     (then upper-triangularised)
                 = 0  otherwise

    The matrix is therefore 0/1 valued and upper-triangular.  Its rows read
    directly: an injection at stop ``i`` is fully felt until the first stop whose
    arrival is either masked by a larger measured delay or pinned to the plan,
    and is invisible everywhere after that.  That is the formal statement of
    "the fleet forgets it".

    Parameters
    ----------
    result : CascadeResult
        Solved cascade.  Only the operating point (injections, sources, the
        resulting delay field) is used.
    tr_id : int
        Route to analyse.

    Returns
    -------
    numpy.ndarray
        ``(n, n)`` matrix ``M[i, j] = dD[j] / dI[i]`` over the route's stops, or
        a ``(0, 0)`` array when the route is unknown.
    """
    vertices = result.graph.route_vertices.get(int(tr_id))
    if vertices is None:
        return np.zeros((0, 0), dtype=float)
    count = len(vertices)
    if count == 0:
        return np.zeros((0, 0), dtype=float)
    # lastzero[j] = index of the most recent delay-cancelling segment at or
    # before j; an injection older than that cannot influence D[j].
    last_zero = np.zeros(count, dtype=np.int64)
    for j in range(1, count):
        stop = result.graph.stops[int(vertices[j])]
        previous = float(result.delay[int(vertices[j - 1])])
        floor = max(0.0, float(result.injected[stop.vertex]))
        source = _vertex_source(result.graph, stop.vertex, result.overruns)
        last_zero[j] = j if (previous - stop.buffer_s + source) < floor - EPS_S else last_zero[j - 1]
    index = np.arange(count)
    matrix = (index[:, None] >= last_zero[None, :]).astype(float)
    return np.triu(matrix)


def _is_int(value: Any) -> bool:
    """Return ``True`` when ``value`` can be losslessly read as an int."""
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


class CascadeEngine:
    """Stateful convenience wrapper: one cached graph, many cheap projections.

    Building the graph parses the whole schedule and runs the corridor search, so
    it must happen once per process.  :class:`CascadeEngine` owns that cost and
    exposes the two operations the application actually performs:

    * :meth:`solve` -- fleet-level cascade for the simulator / ``/stream/status``;
    * :meth:`project` -- single-vehicle ETA projection for ``/predict`` records.

    Attributes
    ----------
    graph : CascadeGraph
        The cached graph (``None`` until a schedule is supplied).
    config : CascadeConfig
        Parameters used for both operations.
    """

    __slots__ = ("_graph", "_config")

    def __init__(self, config: CascadeConfig | None = None) -> None:
        self._config = config or CascadeConfig()
        self._graph: CascadeGraph | None = None

    @classmethod
    def from_schedule(
        cls,
        schedule: pd.DataFrame,
        config: CascadeConfig | None = None,
    ) -> CascadeEngine:
        """Build an engine with a pre-computed graph."""
        engine = cls(config)
        engine._graph = build_cascade_graph(schedule, engine._config)
        return engine

    @property
    def graph(self) -> CascadeGraph:
        """Return the cached graph; an empty graph when no schedule was given."""
        return self._graph if self._graph is not None else CascadeGraph((), (), self._config)

    @property
    def config(self) -> CascadeConfig:
        """Return the engine configuration."""
        return self._config

    def attach_schedule(self, schedule: pd.DataFrame) -> None:
        """(Re)build the graph from ``schedule``.  Idempotent for equal input."""
        self._graph = build_cascade_graph(schedule, self._config)

    def solve(
        self,
        injections: Mapping[Any, float] | None = None,
        overruns: Mapping[Any, float] | None = None,
    ) -> CascadeResult:
        """Solve the network cascade for the given injections."""
        return solve_cascade(self.graph, injections, overruns, self._config)

    def project(
        self,
        tr_id: Any,
        stop_id: Any,
        delay_s: Any,
        horizon: int = 8,
        dwell_overrun_s: float | None = None,
    ) -> list[dict[str, Any]]:
        """Project one vehicle's delay forward; never raises."""
        if self._graph is None:
            return []
        return project_route_eta(
            self._graph, tr_id, stop_id, delay_s, horizon=horizon, dwell_overrun_s=dwell_overrun_s
        )
