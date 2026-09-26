"""Tests for the cascading-delay model in :mod:`src.cascade` (synthetic, no data).

Checks the mathematical contract of the propagation, not the code layout:

  1. the plan floor -- a delay never turns into phantom early arrivals;
  2. catch-up -- with ``psi = 0`` the delay is carried verbatim to the last stop;
  3. absorption -- with ``psi > 0`` the delay burns down by ``psi * R`` per stop
     and dies exactly where the closed form says it does;
  4. nuisance sources add linearly when there is no recovery capacity;
  5. monotone + deterministic -- a bigger injection never yields a smaller
     delay, and solving twice yields the same field;
  6. cross-route contagion -- an intersecting route inherits part of the delay
     and is reported in ``contaminated_routes``;
  7. a measured injection acts as a floor, so it wins over anything propagated;
  8. the fast single-route projection agrees with the full graph solve;
  9. the elasticity matrix is upper-triangular and decays after absorption;
 10. total functions -- empty schedules, unknown ids and garbage never raise.

Note on configuration: the recoverable buffer ``b = psi * R`` is baked into the
graph when it is built, so every solve reuses ``graph.config``.  Each scenario
therefore gets its own graph.

Usage:
    py ml/test_cascade.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cascade import (  # noqa: E402
    CascadeConfig,
    CascadeEngine,
    build_cascade_graph,
    project_route_eta,
    route_transfer_matrix,
    solve_cascade,
)

STEP_S = 300.0  # planned inter-stop run time in the mock timetable
STOPS = 10
PSI = 0.15
BUFFER = PSI * STEP_S  # 45 s of catch-up per segment
BASE = pd.Timestamp("2026-01-06 08:00:00")


def mock_schedule(stops: int = STOPS) -> pd.DataFrame:
    """Build a two-route mock timetable sharing one physical corridor.

    Route 1 departs at ``08:00:00`` with a 300 s headway; route 2 is offset by
    120 s and displaced by ~0.26 km, so its stops fall inside the corridor radius
    while the scheduled connection gap (120 s) stays inside the transfer window.
    Every inter-stop run is exactly ``STEP_S`` seconds, which makes the expected
    delay field computable by hand.
    """
    rows: list[dict[str, object]] = []
    layout = {
        1: {"offset_s": 0, "lat": 55.750, "lon": 37.600},
        2: {"offset_s": 120, "lat": 55.752, "lon": 37.602},
    }
    for route, spec in layout.items():
        for index in range(stops):
            rows.append(
                {
                    "tr_id": route,
                    "tt_action_item_id": route * 100 + index + 1,
                    "time_begin": (
                        BASE + pd.Timedelta(seconds=spec["offset_s"] + STEP_S * index)
                    ).isoformat(),
                    "geom": f"POINT ({spec['lon'] + 0.005 * index:.6f} {spec['lat']:.6f})",
                    "building_address": f"route-{route}-stop-{index}",
                }
            )
    return pd.DataFrame(rows)


def make_graph(schedule: pd.DataFrame, *, psi: float = 0.0, dwell: float = 0.0, carry: float = 0.7):
    """Build a graph with an explicit recovery / nuisance / coupling profile.

    ``carry=0`` yields a purely one-dimensional graph, which is what isolates the
    chain recursion from the cross-route relaxation in the assertions below.
    """
    config = CascadeConfig(run_recovery=psi, dwell_overrun_s=dwell, transfer_carry=carry)
    return build_cascade_graph(schedule, config)


def main() -> None:
    schedule = mock_schedule()
    routes = (1, 2)
    carry_graph = make_graph(schedule)
    plain_graph = make_graph(schedule, carry=0.0)
    psi_graph = make_graph(schedule, psi=PSI)
    plain_psi_graph = make_graph(schedule, psi=PSI, carry=0.0)

    # ---- graph shape ----------------------------------------------------- #
    graph = carry_graph
    assert graph.stats["stops"] == 2 * STOPS, graph.stats
    assert graph.stats["routes"] == 2, graph.stats
    assert graph.stats["continuation_links"] == 2 * (STOPS - 1), graph.stats
    assert graph.stats["transfer_links"] > 0, "corridor overlap produced no transfer links"
    # Ordered pairs: the corridor is detected in both directions.
    assert graph.stats["corridor_pairs"] == 2, graph.stats
    assert plain_graph.stats["transfer_links"] == 0, plain_graph.stats
    for route in routes:
        vertices = graph.route_vertices[route]
        assert len(vertices) == STOPS, (route, len(vertices))
        first = graph.stops[int(vertices[0])]
        assert first.buffer_s == 0.0 and first.inbound == -1, first
        for position in range(1, STOPS):
            stop = graph.stops[int(vertices[position])]
            assert stop.run_planned_s == STEP_S, (route, position, stop.run_planned_s)
            assert stop.position == position, stop
    assert np.isclose(psi_graph.stops[1].buffer_s, BUFFER), psi_graph.stops[1]
    assert all(
        link.carry < 1.0 for link in psi_graph.transfer_links
    ), "transfer carry must stay contractive"
    assert all(
        math_isclose(link.carry, 0.7 * (0.5 + 0.5 * link.overlap))
        for link in psi_graph.transfer_links
    ), "carry must follow the corridor-overlap index"

    # ---- 1. plan floor + 2. no recovery ---------------------------------- #
    carried = solve_cascade(plain_graph, {(1, 103): 300.0})
    chain = carried.route_delay(1)
    assert np.all(chain >= -1e-12), chain  # never a phantom early arrival
    assert np.allclose(chain[2:], 300.0), chain  # delay carried verbatim
    assert carried.route_delay(2).max() == 0.0, "no transfer links -> no contagion"

    # ---- 3. absorption with catch-up ------------------------------------- #
    result = solve_cascade(plain_psi_graph, {(1, 103): 300.0})
    chain = result.route_delay(1)
    expected = [300.0 - BUFFER * k for k in range(STOPS - 3)]
    expected.append(0.0)  # the delay dies instead of going negative
    assert np.allclose(chain[2:], expected), (chain[2:], expected)
    absorption = result.absorption(1)
    assert absorption["delay_s"] == 300.0, absorption
    assert absorption["absorbed"] is True, absorption
    assert absorption["absorbing_stop_id"] == 110, absorption
    assert np.isclose(absorption["recoverable_s"], BUFFER * (STOPS - 3)), absorption
    assert absorption["residual_at_end_s"] == 0.0, absorption
    # A delay larger than the total buffer survives to the end of the route.
    total_buffer = BUFFER * (STOPS - 3)
    big = solve_cascade(plain_psi_graph, {(1, 103): total_buffer + 100.0})
    big_absorption = big.absorption(1)
    assert big_absorption["absorbed"] is False, big_absorption
    assert big_absorption["absorbing_stop_id"] is None, big_absorption
    assert np.isclose(big_absorption["residual_at_end_s"], 100.0), big_absorption
    # Zero recovery capacity: no delay is ever absorbed.
    huge = solve_cascade(plain_graph, {(1, 103): 10_000.0})
    assert np.isclose(huge.route_delay(1)[-1], 10_000.0), huge.route_delay(1)

    # ---- 4. nuisance sources --------------------------------------------- #
    dwell_graph = make_graph(schedule, dwell=20.0, carry=0.0)
    chain = solve_cascade(dwell_graph, {(1, 101): 0.0}).route_delay(1)
    assert np.allclose(chain, [20.0 * k for k in range(STOPS)]), chain
    # With an injection and no recovery capacity the two simply add up.
    mixed = solve_cascade(dwell_graph, {(1, 103): 100.0}).route_delay(1)
    assert np.allclose(mixed[2:], 100.0 + 20.0 * np.arange(0, STOPS - 2)), mixed
    # A dwell overrun keeps regenerating delay: with no recovery capacity the
    # delay grows linearly and never converges.  `absorption` needs a measured
    # injection, so this is asserted on the delay field itself.
    growing = solve_cascade(dwell_graph, {(1, 101): 0.0}).route_delay(1)
    assert growing[-1] == 20.0 * (STOPS - 1), growing
    # Same overrun, but the vehicle has catch-up capacity: growth is sub-linear
    # and the end-of-route delay is materially lower.
    damped_graph = make_graph(schedule, psi=0.5, dwell=20.0, carry=0.0)
    damped = solve_cascade(damped_graph, {(1, 101): 0.0}).route_delay(1)
    assert damped[-1] < growing[-1], (damped, growing)
    assert np.all(np.diff(damped) <= 20.0), damped
    # A persistent overrun with recovery converges to the fixed point
    # n / (1 - psi) = 20 / 0.5 = 40 s instead of diverging.
    assert damped[-1] <= 40.0, damped

    # ---- 5. monotonicity + determinism ----------------------------------- #
    small = solve_cascade(plain_psi_graph, {(1, 103): 60.0}).route_delay(1)
    large = solve_cascade(plain_psi_graph, {(1, 103): 600.0}).route_delay(1)
    assert np.all(large >= small - 1e-9), "monotonicity in the injection"
    again = solve_cascade(plain_psi_graph, {(1, 103): 300.0})
    assert np.allclose(result.delay, again.delay), "solver is not deterministic"
    # The fixed point is reached well before the iteration cap.
    assert result.iterations <= 3, result.iterations
    spread_once = solve_cascade(psi_graph, {(1, 103): 300.0})
    spread_twice = solve_cascade(psi_graph, {(1, 103): 300.0})
    assert np.allclose(spread_once.delay, spread_twice.delay), "cross-route determinism"
    assert spread_once.iterations < psi_graph.config.max_iterations, spread_once.iterations

    # ---- 6. cross-route contagion ---------------------------------------- #
    victim = spread_once.route_delay(2)
    donor = spread_once.route_delay(1)
    assert victim[0] == 0.0, victim  # nothing before the corridor meeting point
    assert victim[3] > 0.0, f"no contagion onto route 2: {victim}"
    assert np.any(spread_once.contagion > 0.0), spread_once.contagion
    assert spread_once.contaminated_routes() == [2], spread_once.contaminated_routes()
    # Contagion is strictly weaker than the donor's delay: carry < 1, and the
    # receiving vehicle has a buffer of its own to burn.
    assert victim[3] < donor[3], (donor[3], victim[3])
    assert np.all(victim >= 0.0), victim
    # A zero carry disables cross-route coupling entirely.
    assert solve_cascade(plain_psi_graph, {(1, 103): 300.0}).contagion.max() == 0.0
    # The transfer link obeys the documented operator (indices are global), and
    # the link fed by the late part of route 1 is the one that bites.
    active = max(
        psi_graph.transfer_links,
        key=lambda item: item.apply(float(spread_once.delay[item.src])),
    )
    assert active.apply(float(spread_once.delay[active.src])) > 0.0, active
    assert np.isclose(
        spread_once.delay[active.dst],
        max(
            0.0,
            active.carry * spread_once.delay[active.src] - active.buffer_s,
        )
        + active.local_source,
    ), (active, spread_once.delay[active.dst])
    assert spread_once.contagion[active.dst] > 0.0, "transfer target carries no contagion"

    # ---- 7. injection wins as a floor ------------------------------------ #
    floored = solve_cascade(psi_graph, {(1, 103): 300.0, (2, 204): 900.0})
    assert np.isclose(floored.route_delay(2)[3], 900.0), floored.route_delay(2)
    assert floored.route_delay(2)[3] >= victim[3], "measured injection is not a floor"
    assert floored.absorption(2)["delay_s"] == 900.0, floored.absorption(2)
    # Coupling is symmetric in time: the late route 2 pushes back onto the
    # downstream stops of route 1 through the reverse-direction transfer link.
    assert floored.contaminated_routes() == [1, 2], floored.contaminated_routes()

    # ---- 8. fast projection == graph solve -------------------------------- #
    projection = project_route_eta(plain_psi_graph, 1, 103, 300.0, horizon=STOPS)
    assert len(projection) == STOPS - 2, projection
    assert np.allclose(
        [row["delay_s"] for row in projection], result.route_delay(1)[2:], atol=1e-3
    ), projection
    assert projection[0]["is_target"] is True, projection[0]
    for row in projection:
        # Absorbed stops land exactly on the plan: the vehicle is never early.
        assert row["eta"] >= row["plan_time"], "ETA must never precede the plan"
        assert (row["eta"] > row["plan_time"]) == (row["delay_s"] > 0.0), row
    # The summary projection starts at the injection, not at the depot.
    summary_projection = result.route_projection(1, horizon=4)
    assert len(summary_projection) == 4, summary_projection
    assert summary_projection[0]["stop_id"] == 103, summary_projection[0]
    assert summary_projection[0]["injected_s"] == 300.0, summary_projection[0]
    assert result.route_projection(1, horizon=4, from_injection=False)[0]["position"] == 0
    # The projection agrees with the solve even when transfer links exist,
    # because a single vehicle cannot be contaminated by its own future.
    spread_projection = project_route_eta(psi_graph, 1, 103, 300.0, horizon=STOPS)
    assert np.allclose(
        [row["delay_s"] for row in spread_projection], donor[2:], atol=1e-3
    ), spread_projection

    # ---- 9. elasticity ---------------------------------------------------- #
    matrix = route_transfer_matrix(result, 1)
    assert matrix.shape == (STOPS, STOPS), matrix.shape
    assert np.allclose(matrix, np.triu(matrix)), "must be upper-triangular"
    assert np.allclose(np.diag(matrix), 1.0), "diagonal must be 1"
    assert set(np.unique(matrix)) <= {0.0, 1.0}, matrix
    # Sensitivity dies where the delay is absorbed: the row of the injection stop
    # stays 1 up to the last stop that still carries delay and 0 afterwards.
    carrying = int(np.flatnonzero(chain[2:] > 0.0)[-1]) + 2
    assert np.allclose(matrix[2, 2:carrying], 1.0), matrix[2]
    assert np.allclose(matrix[2, carrying + 1 :], 0.0), matrix[2]
    # Rows starting at or after the absorbing stop are the identity: nothing
    # upstream of them can move them.  This is "the fleet forgets it".
    assert np.allclose(matrix[carrying + 1 :, :], np.eye(STOPS)[carrying + 1 :, :]), matrix
    # A tiny injection that dies immediately makes the whole route insensitive.
    tiny = solve_cascade(plain_psi_graph, {(1, 101): 1.0})
    assert np.allclose(route_transfer_matrix(tiny, 1), np.eye(STOPS)), route_transfer_matrix(tiny, 1)
    # No recovery capacity: the injection row stays fully sensitive end to end,
    # i.e. the delay is remembered at every downstream stop.
    unit = route_transfer_matrix(carried, 1)
    assert np.allclose(unit[2, 2:], 1.0), unit
    assert np.allclose(route_transfer_matrix(result, 999), np.zeros((0, 0))), "unknown route"

    # ---- 10. totality ------------------------------------------------------ #
    empty = build_cascade_graph(pd.DataFrame(), CascadeConfig())
    assert empty.stats["stops"] == 0, empty.stats
    assert solve_cascade(empty, {(1, 1): 100.0}).summary()["max_delay_s"] == 0.0
    assert project_route_eta(empty, 1, 1, 100.0) == []
    assert solve_cascade(empty, {"garbage": float("nan")}).delay.size == 0
    for bad in (999, None, "x", float("nan")):
        assert project_route_eta(plain_graph, bad, 103, 100.0) == [], bad
        assert project_route_eta(plain_graph, 1, bad, 100.0) == [], bad
    assert solve_cascade(plain_graph, {}).delay.max() == 0.0, "no injections -> no delay"
    assert solve_cascade(plain_graph, {(1, 103): -50.0}).delay.max() == 0.0, "negative clamp"
    assert solve_cascade(plain_graph, {(1, 999_999): 100.0}).delay.max() == 0.0, "unknown stop"
    # A schedule without geometry still yields a usable one-dimensional graph.
    no_geom = schedule.drop(columns=["geom"])
    blind = build_cascade_graph(no_geom, CascadeConfig())
    assert blind.stats["vertices_without_geometry"] == 2 * STOPS, blind.stats
    assert blind.stats["transfer_links"] == 0, "no geometry -> no corridors"
    assert solve_cascade(blind, {(1, 103): 300.0}).route_delay(1)[2] == 300.0

    # ---- engine wrapper ---------------------------------------------------- #
    engine = CascadeEngine.from_schedule(schedule, CascadeConfig(run_recovery=PSI))
    assert engine.graph.stats["stops"] == 2 * STOPS, engine.graph.stats
    assert engine.project(1, 103, 300.0, horizon=3)[0]["delay_s"] == 300.0
    assert engine.project("oops", 103, 300.0) == []
    assert CascadeEngine().project(1, 103, 300.0) == []  # no schedule attached yet
    engine.attach_schedule(schedule)
    assert len(engine.solve({(1, 103): 300.0}).route_delay(1)) == STOPS
    summary = engine.solve({(1, 103): 300.0}).summary()
    for key in ("iterations", "contaminated_routes", "absorbed_ratio", "absorption"):
        assert key in summary, (key, summary)
    assert summary["injected_routes"] == [1], summary
    assert summary["total_contagion_s"] > 0.0, summary

    # ---- regime presets ---------------------------------------------------- #
    assert CascadeConfig.for_regime("congested").run_recovery < PSI, "regimes must differ"
    assert CascadeConfig.for_regime("normal").run_recovery == PSI
    assert CascadeConfig.for_regime("bogus").run_recovery == CascadeConfig.for_regime("normal").run_recovery
    assert CascadeConfig(run_recovery=5.0).run_recovery == 0.9, "config must clamp"
    assert CascadeConfig(transfer_carry=1.0).transfer_carry == 0.95, "carry must clamp"

    print("ALL CASCADE TESTS PASSED")


def math_isclose(value: float, target: float) -> bool:
    """Local ``math.isclose`` wrapper (kept explicit for readability)."""
    return abs(value - target) <= 1e-9


if __name__ == "__main__":
    main()
