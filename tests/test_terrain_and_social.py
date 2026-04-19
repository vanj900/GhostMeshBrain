"""Tests for terrain types, WorldEventSystem, seasons, and social actions.

Coverage
--------
Terrain
  - FOREST and ROCKY cells appear in a freshly generated grid
  - Moving into FOREST costs energy (delta_energy < 0)
  - Moving into ROCKY costs heat (delta_heat > 0)
  - GATHER on FOREST yields positive energy
  - GATHER on ROCKY yields positive integrity
  - Both FOREST and ROCKY respawn after gathering
  - FOREST and ROCKY are in _GATHERABLE
  - available_actions() includes GATHER when standing on FOREST or ROCKY

Seasons
  - current_season starts at "spring" after reset
  - Season advances after season_length ticks
  - Summer boosts FOOD energy yield (multiplier > 1)
  - Winter reduces FOOD energy yield (multiplier < 1)
  - season_length=0 disables seasons (always spring)
  - Respawn delay is longer in winter than summer

WorldEventSystem
  - tick() returns a WorldEvent
  - storm_prob=1.0 always fires a storm hit
  - storm_prob=0.0 never fires a storm hit
  - predator_prob=1.0 immediately starts a predator event
  - predator event expires after its duration
  - drought_prob=1.0 immediately starts a drought
  - drought expires after its duration
  - Seeded RNG produces reproducible events
  - label field reflects active events
  - predator_threat=0.0 when no predator is active

WorldObservation new fields
  - predator_threat defaults to 0.0
  - active_world_event defaults to ""
  - get_observation() passes predator_threat and active_world_event through

Social actions
  - WorldAction enum contains SIGNAL, COOPERATE, BETRAY, OBSERVE_OTHER
  - SIGNAL is cheaper than BROADCAST in energy cost

MultiAgentRunner social features
  - ReputationSystem starts all agents at 0.5
  - cooperated() raises score, betrayed() lowers score, clamped to [0,1]
  - AgentResult has final_reputation and starting_mask fields
  - Agents receive different starting_mask values
  - run() completes without error with social actions enabled
  - AgentResult fields signals_sent / cooperations / betrayals / observations_made exist
"""

from __future__ import annotations

import os
import tempfile

import pytest

from thermodynamic_agency.world.grid_world import (
    GridWorld,
    WorldAction,
    WorldObservation,
    CellType,
    WorldEventSystem,
    WorldEvent,
    _GATHERABLE,
    _GATHER_EFFECTS,
    _ENTRY_EFFECTS,
    _SEASON_FORAGE_MULTIPLIER,
    _SEASON_RESPAWN_MULTIPLIER,
)
from thermodynamic_agency.world.multi_agent_runner import (
    MultiAgentRunner,
    ReputationSystem,
    AgentResult,
    SIGNAL_ENERGY_COST,
    BROADCAST_ENERGY_COST,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _world(seed: int = 1, **kwargs) -> GridWorld:
    return GridWorld(seed=seed, **kwargs)


def _place_on(world: GridWorld, cell: str) -> bool:
    """Move agent onto a cell of the given type. Returns True if found."""
    for y in range(world.height):
        for x in range(world.width):
            if world.cell_at((x, y)) == cell:
                world._agent_pos = (x, y)
                return True
    return False


def _find_adjacent(world: GridWorld, target_cell: str):
    """Return (agent_pos, action) to move onto a target_cell, or None."""
    from thermodynamic_agency.world.grid_world import _MOVEMENT_DELTA
    action_map = {v: WorldAction(k) for k, v in _MOVEMENT_DELTA.items()}
    for y in range(1, world.height - 1):
        for x in range(1, world.width - 1):
            if world.cell_at((x, y)) == target_cell:
                for (dx, dy), action in [
                    ((0, -1), WorldAction.NORTH),
                    ((0, 1), WorldAction.SOUTH),
                    ((1, 0), WorldAction.EAST),
                    ((-1, 0), WorldAction.WEST),
                ]:
                    nx, ny = x - dx, y - dy   # agent pos that leads into (x,y)
                    if (
                        0 <= nx < world.width
                        and 0 <= ny < world.height
                        and world.cell_at((nx, ny)) not in (CellType.WALL.value, target_cell)
                    ):
                        return (nx, ny), action
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Terrain — FOREST
# ─────────────────────────────────────────────────────────────────────────────

class TestForestTerrain:
    def test_forest_in_gatherable(self):
        assert CellType.FOREST.value in _GATHERABLE

    def test_forest_gather_effects_defined(self):
        assert "delta_energy" in _GATHER_EFFECTS.get(CellType.FOREST.value, {})
        assert _GATHER_EFFECTS[CellType.FOREST.value]["delta_energy"] > 0

    def test_forest_entry_effect_costs_energy(self):
        effect = _ENTRY_EFFECTS.get(CellType.FOREST.value, {})
        assert "delta_energy" in effect
        assert effect["delta_energy"] < 0

    def test_forest_cells_appear_in_grid(self):
        """At least one FOREST cell should be generated in a fresh grid."""
        world = _world(seed=1)
        found = any(
            world.cell_at((x, y)) == CellType.FOREST.value
            for y in range(world.height)
            for x in range(world.width)
        )
        assert found, "No FOREST cell found in grid"

    def test_entering_forest_costs_energy(self):
        world = _world(seed=5)
        result = _find_adjacent(world, CellType.FOREST.value)
        if result is None:
            pytest.skip("No adjacent-entry forest cell found")
        (nx, ny), action = result
        world._agent_pos = (nx, ny)
        step = world.step(action)
        if step.new_position != (nx, ny):  # movement succeeded
            assert step.metabolic_delta.get("delta_energy", 0.0) < 0

    def test_gather_forest_yields_energy(self):
        world = _world(seed=3)
        if not _place_on(world, CellType.FOREST.value):
            pytest.skip("No FOREST cell found")
        result = world.step(WorldAction.GATHER)
        assert result.gathered
        assert result.metabolic_delta.get("delta_energy", 0.0) > 0

    def test_forest_respawns_after_gather(self):
        world = _world(seed=3)
        if not _place_on(world, CellType.FOREST.value):
            pytest.skip("No FOREST cell found")
        pos = world.agent_position
        world.step(WorldAction.GATHER)
        assert world.cell_at(pos) == CellType.EMPTY.value
        for _ in range(40):
            world.step(WorldAction.WAIT)
        assert world.cell_at(pos) == CellType.FOREST.value

    def test_available_actions_includes_gather_on_forest(self):
        world = _world(seed=3)
        if not _place_on(world, CellType.FOREST.value):
            pytest.skip("No FOREST cell found")
        assert WorldAction.GATHER in world.available_actions()


# ─────────────────────────────────────────────────────────────────────────────
# Terrain — ROCKY
# ─────────────────────────────────────────────────────────────────────────────

class TestRockyTerrain:
    def test_rocky_in_gatherable(self):
        assert CellType.ROCKY.value in _GATHERABLE

    def test_rocky_gather_effects_defined(self):
        assert "delta_integrity" in _GATHER_EFFECTS.get(CellType.ROCKY.value, {})
        assert _GATHER_EFFECTS[CellType.ROCKY.value]["delta_integrity"] > 0

    def test_rocky_entry_effect_costs_heat(self):
        effect = _ENTRY_EFFECTS.get(CellType.ROCKY.value, {})
        assert "delta_heat" in effect
        assert effect["delta_heat"] > 0

    def test_rocky_cells_appear_in_grid(self):
        found = any(
            _world(seed=s).cell_at((x, y)) == CellType.ROCKY.value
            for s in range(1, 10)
            for y in range(10)
            for x in range(10)
        )
        assert found, "No ROCKY cell found across 9 seeds"

    def test_entering_rocky_costs_heat(self):
        # Find a seed where ROCKY has a navigable adjacent cell
        for seed in range(1, 20):
            world = _world(seed=seed)
            result = _find_adjacent(world, CellType.ROCKY.value)
            if result is None:
                continue
            (nx, ny), action = result
            world._agent_pos = (nx, ny)
            step = world.step(action)
            if step.new_position != (nx, ny):
                assert step.metabolic_delta.get("delta_heat", 0.0) > 0
                return
        pytest.skip("Could not find navigable ROCKY cell in seeds 1–19")

    def test_gather_rocky_yields_integrity(self):
        for seed in range(1, 20):
            world = _world(seed=seed)
            if _place_on(world, CellType.ROCKY.value):
                result = world.step(WorldAction.GATHER)
                assert result.gathered
                assert result.metabolic_delta.get("delta_integrity", 0.0) > 0
                return
        pytest.skip("No ROCKY cell found in seeds 1–19")

    def test_rocky_respawns_after_gather(self):
        for seed in range(1, 20):
            world = _world(seed=seed)
            if _place_on(world, CellType.ROCKY.value):
                pos = world.agent_position
                world.step(WorldAction.GATHER)
                assert world.cell_at(pos) == CellType.EMPTY.value
                for _ in range(45):
                    world.step(WorldAction.WAIT)
                assert world.cell_at(pos) == CellType.ROCKY.value
                return
        pytest.skip("No ROCKY cell found")


# ─────────────────────────────────────────────────────────────────────────────
# Seasons
# ─────────────────────────────────────────────────────────────────────────────

class TestSeasons:
    def test_initial_season_is_spring(self):
        world = _world(seed=1)
        assert world.current_season == "spring"

    def test_season_advances_after_season_length(self):
        world = GridWorld(seed=1, season_length=10)
        for _ in range(10):
            world.step(WorldAction.WAIT)
        assert world.current_season == "summer"

    def test_full_cycle_returns_to_spring(self):
        world = GridWorld(seed=1, season_length=5)
        for _ in range(20):
            world.step(WorldAction.WAIT)
        assert world.current_season == "spring"

    def test_summer_boosts_food_yield(self):
        """Summer multiplier > 1 so food gather gives more energy in summer."""
        assert _SEASON_FORAGE_MULTIPLIER["summer"] > 1.0

    def test_winter_reduces_food_yield(self):
        assert _SEASON_FORAGE_MULTIPLIER["winter"] < 1.0

    def test_summer_gather_more_energy_than_winter(self):
        """Summer multiplier produces more delta_energy than winter for FOOD."""
        world = _world(seed=10)
        # Force summer
        world._season_index = 1
        assert world.current_season == "summer"
        summer_effects = world._apply_season_to_gather(
            CellType.FOOD.value, {"delta_energy": 15.0}
        )
        # Force winter
        world._season_index = 3
        assert world.current_season == "winter"
        winter_effects = world._apply_season_to_gather(
            CellType.FOOD.value, {"delta_energy": 15.0}
        )
        assert summer_effects["delta_energy"] > winter_effects["delta_energy"]

    def test_season_length_zero_disables_seasons(self):
        world = GridWorld(seed=1, season_length=0)
        for _ in range(100):
            world.step(WorldAction.WAIT)
        assert world.current_season == "spring"

    def test_winter_respawn_slower_than_summer(self):
        assert _SEASON_RESPAWN_MULTIPLIER["winter"] > _SEASON_RESPAWN_MULTIPLIER["summer"]

    def test_reset_resets_season_to_spring(self):
        world = GridWorld(seed=1, season_length=5)
        for _ in range(15):
            world.step(WorldAction.WAIT)
        world.reset()
        assert world.current_season == "spring"


# ─────────────────────────────────────────────────────────────────────────────
# WorldEventSystem
# ─────────────────────────────────────────────────────────────────────────────

class TestWorldEventSystem:
    def test_tick_returns_world_event(self):
        es = WorldEventSystem(seed=0)
        event = es.tick(0)
        assert isinstance(event, WorldEvent)

    def test_storm_prob_1_always_fires(self):
        es = WorldEventSystem(storm_prob=1.0, predator_prob=0.0, drought_prob=0.0, seed=0)
        for tick in range(20):
            event = es.tick(tick)
            assert event.storm_hit

    def test_storm_prob_0_never_fires(self):
        es = WorldEventSystem(storm_prob=0.0, predator_prob=0.0, drought_prob=0.0, seed=0)
        for tick in range(100):
            event = es.tick(tick)
            assert not event.storm_hit

    def test_predator_prob_1_activates_immediately(self):
        es = WorldEventSystem(storm_prob=0.0, predator_prob=1.0, drought_prob=0.0, seed=0)
        event = es.tick(0)
        assert es.predator_active
        assert event.predator_threat > 0.0

    def test_predator_event_expires(self):
        from thermodynamic_agency.world.grid_world import _PREDATOR_MAX_DURATION
        es = WorldEventSystem(storm_prob=0.0, predator_prob=1.0, drought_prob=0.0, seed=0)
        # Force predator on
        es._predator_active = True
        es._predator_ticks_remaining = 3
        for _ in range(3):
            es.tick(0)
        assert not es.predator_active

    def test_drought_prob_1_activates_immediately(self):
        es = WorldEventSystem(storm_prob=0.0, predator_prob=0.0, drought_prob=1.0, seed=0)
        es.tick(0)
        assert es.drought_active

    def test_drought_expires(self):
        es = WorldEventSystem(storm_prob=0.0, predator_prob=0.0, drought_prob=0.0, seed=0)
        es._drought_active = True
        es._drought_ticks_remaining = 2
        es.tick(0)
        es.tick(1)
        assert not es.drought_active

    def test_seeded_rng_reproducible(self):
        es1 = WorldEventSystem(storm_prob=0.1, predator_prob=0.05, seed=77)
        es2 = WorldEventSystem(storm_prob=0.1, predator_prob=0.05, seed=77)
        for t in range(50):
            e1 = es1.tick(t)
            e2 = es2.tick(t)
            assert e1.storm_hit == e2.storm_hit
            assert e1.predator_threat == e2.predator_threat
            assert e1.drought_active == e2.drought_active

    def test_label_reflects_active_events(self):
        es = WorldEventSystem(storm_prob=1.0, predator_prob=0.0, drought_prob=0.0, seed=0)
        event = es.tick(0)
        assert "storm" in event.label

    def test_no_predator_threat_when_inactive(self):
        es = WorldEventSystem(storm_prob=0.0, predator_prob=0.0, drought_prob=0.0, seed=0)
        event = es.tick(0)
        assert event.predator_threat == 0.0

    def test_world_event_drought_active_field(self):
        es = WorldEventSystem(storm_prob=0.0, predator_prob=0.0, drought_prob=1.0, seed=0)
        event = es.tick(0)
        assert event.drought_active


# ─────────────────────────────────────────────────────────────────────────────
# WorldObservation new fields
# ─────────────────────────────────────────────────────────────────────────────

class TestObservationNewFields:
    def test_default_predator_threat_is_zero(self):
        world = _world(seed=1)
        obs = world.get_observation()
        assert obs.predator_threat == 0.0

    def test_default_active_event_is_empty_string(self):
        world = _world(seed=1)
        obs = world.get_observation()
        assert obs.active_world_event == ""

    def test_get_observation_passes_predator_threat(self):
        world = _world(seed=1)
        obs = world.get_observation(predator_threat=0.75)
        assert obs.predator_threat == 0.75

    def test_get_observation_passes_active_event(self):
        world = _world(seed=1)
        obs = world.get_observation(active_world_event="storm")
        assert obs.active_world_event == "storm"

    def test_step_observation_has_default_predator_threat(self):
        world = _world(seed=1)
        result = world.step(WorldAction.WAIT)
        assert result.observation is not None
        assert result.observation.predator_threat == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# WorldAction enum — social actions
# ─────────────────────────────────────────────────────────────────────────────

class TestSocialWorldActions:
    def test_signal_in_world_action(self):
        assert WorldAction.SIGNAL.value == "signal"

    def test_cooperate_in_world_action(self):
        assert WorldAction.COOPERATE.value == "cooperate"

    def test_betray_in_world_action(self):
        assert WorldAction.BETRAY.value == "betray"

    def test_observe_other_in_world_action(self):
        assert WorldAction.OBSERVE_OTHER.value == "observe_other"

    def test_signal_cheaper_than_broadcast(self):
        assert SIGNAL_ENERGY_COST < BROADCAST_ENERGY_COST


# ─────────────────────────────────────────────────────────────────────────────
# ReputationSystem
# ─────────────────────────────────────────────────────────────────────────────

class TestReputationSystem:
    def test_initial_scores_neutral(self):
        rep = ReputationSystem(n_agents=3)
        for i in range(3):
            assert rep.score(i) == pytest.approx(0.5)

    def test_cooperated_raises_score(self):
        rep = ReputationSystem(n_agents=2)
        before = rep.score(0)
        rep.cooperated(0)
        assert rep.score(0) > before

    def test_betrayed_lowers_score(self):
        rep = ReputationSystem(n_agents=2)
        before = rep.score(1)
        rep.betrayed(1)
        assert rep.score(1) < before

    def test_score_clamped_above_one(self):
        rep = ReputationSystem(n_agents=1)
        for _ in range(20):
            rep.cooperated(0)
        assert rep.score(0) <= 1.0

    def test_score_clamped_below_zero(self):
        rep = ReputationSystem(n_agents=1)
        for _ in range(20):
            rep.betrayed(0)
        assert rep.score(0) >= 0.0

    def test_all_scores_returns_dict(self):
        rep = ReputationSystem(n_agents=3)
        scores = rep.all_scores()
        assert isinstance(scores, dict)
        assert set(scores.keys()) == {0, 1, 2}


# ─────────────────────────────────────────────────────────────────────────────
# MultiAgentRunner — social features
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiAgentRunnerSocial:
    def _run_small(self, n_agents: int = 3, ticks: int = 30, seed: int = 42) -> list[AgentResult]:
        os.environ["GHOST_HUD"] = "0"
        os.environ["GHOST_PULSE"] = "0"
        runner = MultiAgentRunner(
            n_agents=n_agents,
            seed=seed,
            world_width=10,
            world_height=10,
            storm_prob=0.05,
            predator_prob=0.05,
            drought_prob=0.05,
        )
        return runner.run(max_ticks=ticks)

    def test_run_completes_without_error(self):
        results = self._run_small()
        assert len(results) == 3

    def test_agent_result_has_social_fields(self):
        results = self._run_small()
        for res in results:
            assert hasattr(res, "signals_sent")
            assert hasattr(res, "cooperations")
            assert hasattr(res, "betrayals")
            assert hasattr(res, "observations_made")
            assert hasattr(res, "final_reputation")
            assert hasattr(res, "starting_mask")

    def test_diverse_starting_masks(self):
        """Agents should be spawned with different starting masks."""
        results = self._run_small(n_agents=5)
        masks = [r.starting_mask for r in results]
        # With 5 agents, we expect at least 2 distinct masks
        assert len(set(masks)) >= 2

    def test_starting_mask_values_are_valid(self):
        from thermodynamic_agency.world.multi_agent_runner import _STARTING_MASKS
        results = self._run_small(n_agents=5)
        for res in results:
            assert res.starting_mask in _STARTING_MASKS

    def test_final_reputation_in_range(self):
        results = self._run_small()
        for res in results:
            assert 0.0 <= res.final_reputation <= 1.0

    def test_social_action_counters_non_negative(self):
        results = self._run_small(ticks=50)
        for res in results:
            assert res.signals_sent >= 0
            assert res.cooperations >= 0
            assert res.betrayals >= 0
            assert res.observations_made >= 0
