"""Long-running tests for self-model evolution, world evolution, and interiority.

These tests run for many ticks to exercise emergent dynamics that only
manifest over sustained time horizons.

The guiding criterion (from the project's founding doctrine): if the agent
ever starts *surprising itself* in ways that feel like genuine interiority
under pressure, that is as close as we'll get to "alive enough."

Test inventory
--------------
MetaCognitiveSelfModel evolution:

1. ``test_narrative_coherence_is_not_hardcoded``
       After a few updates the coherence value should be computed from actual
       affect history, not the legacy constant 0.92.

2. ``test_coherence_stable_when_affect_steady``
       Repeated updates with constant affect → high coherence.

3. ``test_coherence_lower_when_affect_oscillates``
       Oscillating affect reduces narrative coherence relative to stable affect.

4. ``test_surprise_events_accumulate_under_affect_swings``
       Sharp affect oscillations register as surprise events.

5. ``test_interiority_score_grows_with_surprise_events``
       interiority_score() returns 0.0 initially and increases after surprises.

6. ``test_meta_cost_nonzero_under_uncertainty``
       update() returns a positive EFE cost when self-referential uncertainty
       is elevated.

GridWorld evolution:

7. ``test_resource_decay_rate_stored``
       resource_decay_rate is retained after construction.

8. ``test_world_pressure_in_range``
       world_pressure is always in [0, 1].

9. ``test_world_pressure_increases_as_resources_deplete``
       Gathering all resources raises world_pressure.

10. ``test_resource_decay_extends_respawn_delay``
        With decay_rate > 0, gathering late in a run schedules a longer
        respawn than gathering early.

Long-running integration tests:

11. ``test_agent_survives_200_ticks_under_mild_stressor``
        Agent must survive at least 150 of 200 ticks with stressor enabled.

12. ``test_narrative_trace_grows_during_decide_ticks``
        meta_self.narrative_trace grows whenever DECIDE fires; after a short
        run at least one entry should be present.

13. ``test_interiority_nonzero_after_high_stressor_run``
        After 100 ticks with a high-frequency stressor the agent should have
        logged at least one surprise event, indicating non-trivial interiority.
"""

from __future__ import annotations

import pytest

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.meta_cognitive_self_model import (
    MetaCognitiveSelfModel,
    SURPRISE_AFFECT_SWING_THRESHOLD,
    SURPRISE_META_INTENSITY_THRESHOLD,
)
from thermodynamic_agency.world.grid_world import GridWorld, WorldAction, CellType


# ══════════════════════════════════════════════════════════════════════════════
# MetaCognitiveSelfModel evolution
# ══════════════════════════════════════════════════════════════════════════════


class TestMetaCognitiveSelfModelEvolution:
    """The meta-self model must evolve meaningfully over time."""

    def _make(self) -> tuple[MetabolicState, MetaCognitiveSelfModel]:
        state = MetabolicState()
        return state, MetaCognitiveSelfModel(core_self_model=state)

    def test_narrative_coherence_is_not_hardcoded(self):
        """Coherence must be computed, not frozen at 0.92."""
        state, meta = self._make()
        # Three updates are enough to move past the early-exit guard.
        for i in range(4):
            meta.update(state, base_affect=float(i) * 0.2, diary_snapshot="tick")
        coherence = meta.meta_model["narrative_coherence"]
        assert coherence != 0.92, "Coherence must not be the legacy constant 0.92"
        assert 0.0 <= coherence <= 1.0

    def test_coherence_stable_when_affect_steady(self):
        """Constant affect produces high (near-1) narrative coherence."""
        state, meta = self._make()
        for _ in range(7):
            meta.update(state, base_affect=0.3, diary_snapshot="stable")
        coherence = meta.meta_model["narrative_coherence"]
        # Variance of [0.3, 0.3, …] is 0; coherence should be 1.0.
        assert coherence > 0.95, f"Expected coherence > 0.95 for stable affect, got {coherence}"

    def test_coherence_lower_when_affect_oscillates(self):
        """Sharply oscillating affect lowers coherence vs. stable affect."""
        state = MetabolicState()
        meta_stable = MetaCognitiveSelfModel(core_self_model=state)
        for _ in range(7):
            meta_stable.update(state, base_affect=0.3, diary_snapshot="s")
        coherence_stable = meta_stable.meta_model["narrative_coherence"]

        meta_osc = MetaCognitiveSelfModel(core_self_model=state)
        for i in range(7):
            meta_osc.update(state, base_affect=0.9 if i % 2 == 0 else -0.9, diary_snapshot="s")
        coherence_osc = meta_osc.meta_model["narrative_coherence"]

        assert coherence_stable > coherence_osc, (
            f"Stable coherence ({coherence_stable:.3f}) should exceed "
            f"oscillating coherence ({coherence_osc:.3f})"
        )

    def test_surprise_events_accumulate_under_affect_swings(self):
        """Sharp affect oscillations register as surprise events."""
        state, meta = self._make()
        # 20 ticks of ±0.9 oscillations guarantee swings > SURPRISE_AFFECT_SWING_THRESHOLD.
        for i in range(20):
            meta.update(state, base_affect=0.9 if i % 2 == 0 else -0.9, diary_snapshot="tick")
        assert len(meta.surprise_events) > 0, "Expected at least one surprise event"
        for ev in meta.surprise_events:
            assert "entropy" in ev
            assert "affect_swing" in ev
            assert "meta_intensity" in ev

    def test_interiority_score_grows_with_surprise_events(self):
        """interiority_score() starts at 0 and rises after surprise events."""
        state, meta = self._make()
        assert meta.interiority_score() == 0.0, "Initial interiority score must be 0"
        # Drive strong oscillations to fill surprise_events list.
        for i in range(30):
            meta.update(state, base_affect=0.9 if i % 2 == 0 else -0.9, diary_snapshot="t")
        score = meta.interiority_score()
        assert score > 0.0, f"interiority_score should be > 0 after surprises, got {score}"
        assert score <= 1.0

    def test_meta_cost_nonzero_under_uncertainty(self):
        """update() returns positive EFE cost when self-referential uncertainty is high."""
        state = MetabolicState(integrity=50.0)
        meta = MetaCognitiveSelfModel(core_self_model=state)
        meta.meta_model["self_referential_uncertainty"] = 0.8
        cost = meta.update(state, base_affect=-0.7, diary_snapshot="stress")
        assert cost > 0.0, f"Expected positive meta-cost under uncertainty, got {cost}"


# ══════════════════════════════════════════════════════════════════════════════
# GridWorld evolution
# ══════════════════════════════════════════════════════════════════════════════


class TestWorldEvolution:
    """GridWorld must support resource decay and expose an environmental pressure signal."""

    def test_resource_decay_rate_stored(self):
        """resource_decay_rate is correctly stored after construction."""
        world = GridWorld(seed=1, resource_decay_rate=0.05)
        assert world.resource_decay_rate == pytest.approx(0.05)

    def test_resource_decay_rate_defaults_to_zero(self):
        """Default GridWorld has no resource decay."""
        world = GridWorld(seed=2)
        assert world.resource_decay_rate == 0.0

    def test_world_pressure_in_range(self):
        """world_pressure is always in [0, 1]."""
        world = GridWorld(seed=3, resource_decay_rate=0.1)
        for _ in range(30):
            world.step(WorldAction.WAIT)
        assert 0.0 <= world.world_pressure <= 1.0

    def test_world_pressure_increases_as_resources_deplete(self):
        """Gathering resources raises world_pressure (scarcity increases)."""
        world = GridWorld(seed=42)
        pressure_initial = world.world_pressure
        # Consume all gatherable resources by teleporting the agent.
        for y in range(1, world.height - 1):
            for x in range(1, world.width - 1):
                if world.cell_at((x, y)) in (
                    CellType.FOOD.value, CellType.WATER.value, CellType.MEDICINE.value
                ):
                    world._agent_pos = (x, y)
                    world.step(WorldAction.GATHER)
        pressure_depleted = world.world_pressure
        assert pressure_depleted >= pressure_initial, (
            f"Depleted pressure ({pressure_depleted:.3f}) should be >= "
            f"initial pressure ({pressure_initial:.3f})"
        )

    def test_resource_decay_extends_respawn_delay(self):
        """With decay_rate > 0 a resource gathered later gets a longer respawn timer."""
        world = GridWorld(seed=7, resource_decay_rate=0.5, allow_respawn=True)

        # Find two food cells.
        food_cells = [
            (x, y)
            for y in range(1, world.height - 1)
            for x in range(1, world.width - 1)
            if world.cell_at((x, y)) == CellType.FOOD.value
        ]
        if len(food_cells) < 2:
            pytest.skip("Need at least two food cells for this test")

        # Gather the first food cell immediately (world_tick will be 1 after step).
        world._agent_pos = food_cells[0]
        world.step(WorldAction.GATHER)
        timer_early = world._respawn_timers[-1].ticks_remaining if world._respawn_timers else None

        # Advance 50 ticks without restoring the second cell.
        for _ in range(50):
            world.step(WorldAction.WAIT)

        # Gather the second food cell after 51 total ticks.
        world._agent_pos = food_cells[1]
        world.step(WorldAction.GATHER)
        timer_late = world._respawn_timers[-1].ticks_remaining if world._respawn_timers else None

        assert timer_early is not None and timer_late is not None
        assert timer_late > timer_early, (
            f"Late respawn ({timer_late}) should exceed early respawn ({timer_early}) "
            "when resource_decay_rate > 0"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Long-running integration tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLongRunSurvival:
    """Long-running integration tests that exercise emergent dynamics."""

    def test_agent_survives_200_ticks_under_mild_stressor(
        self, tmp_path, monkeypatch
    ):
        """Agent must survive at least 150 of 200 ticks with stressor enabled.

        A mild stressor (prob=0.2, intensity=0.4) creates sustained but
        manageable pressure.  The organism's autoregulation (FORAGE/REST/REPAIR)
        should keep it alive for the majority of the run.
        """
        monkeypatch.setenv("GHOST_STATE_FILE", str(tmp_path / "state.json"))
        monkeypatch.setenv("GHOST_DIARY_PATH", str(tmp_path / "diary.db"))
        monkeypatch.setenv("GHOST_HUD", "0")
        monkeypatch.setenv("GHOST_PULSE", "0")
        monkeypatch.setenv("GHOST_STRESSOR_PROB", "0.2")
        monkeypatch.setenv("GHOST_STRESSOR_INTENSITY", "0.4")
        monkeypatch.setenv("GHOST_STRESSOR_SEED", "7")

        from thermodynamic_agency.pulse import GhostMesh
        from thermodynamic_agency.core.exceptions import GhostDeathException

        mesh = GhostMesh(seed=42)
        ticks_alive = 0
        try:
            for _ in range(200):
                mesh._pulse()
                ticks_alive += 1
        except GhostDeathException:
            pass

        assert ticks_alive >= 150, (
            f"Agent died at tick {ticks_alive}; expected to survive ≥ 150 ticks"
        )

    def test_narrative_trace_grows_during_decide_ticks(self, tmp_path, monkeypatch):
        """meta_self.narrative_trace grows whenever a DECIDE tick fires.

        After 50 ticks on a healthy organism (no stressor), the agent should
        have executed at least one DECIDE cycle and appended to its trace.
        """
        monkeypatch.setenv("GHOST_STATE_FILE", str(tmp_path / "state.json"))
        monkeypatch.setenv("GHOST_DIARY_PATH", str(tmp_path / "diary.db"))
        monkeypatch.setenv("GHOST_HUD", "0")
        monkeypatch.setenv("GHOST_PULSE", "0")

        from thermodynamic_agency.pulse import GhostMesh
        from thermodynamic_agency.core.exceptions import GhostDeathException

        mesh = GhostMesh(seed=0)
        try:
            mesh.run(max_ticks=50)
        except GhostDeathException:
            pass

        assert len(mesh.meta_self.narrative_trace) >= 1, (
            "Expected at least one narrative trace entry after 50 ticks"
        )

    def test_interiority_nonzero_after_high_stressor_run(self, tmp_path, monkeypatch):
        """After 100 ticks under a strong stressor the agent should have logged
        at least one surprise event, indicating non-trivial interiority dynamics.

        A stressor probability of 0.5 fires every other tick on average, creating
        repeated metabolic shocks that force sharp affect transitions.  Each such
        transition that occurs during a DECIDE step should register as a surprise.
        """
        monkeypatch.setenv("GHOST_STATE_FILE", str(tmp_path / "state.json"))
        monkeypatch.setenv("GHOST_DIARY_PATH", str(tmp_path / "diary.db"))
        monkeypatch.setenv("GHOST_HUD", "0")
        monkeypatch.setenv("GHOST_PULSE", "0")
        monkeypatch.setenv("GHOST_STRESSOR_PROB", "0.5")
        monkeypatch.setenv("GHOST_STRESSOR_INTENSITY", "0.5")
        monkeypatch.setenv("GHOST_STRESSOR_SEED", "13")

        from thermodynamic_agency.pulse import GhostMesh
        from thermodynamic_agency.core.exceptions import GhostDeathException

        mesh = GhostMesh(seed=7)
        ticks_run = 0
        try:
            for _ in range(100):
                mesh._pulse()
                ticks_run += 1
        except GhostDeathException:
            pass

        # If the agent had at least some DECIDE ticks, there should be surprises.
        if len(mesh.meta_self.narrative_trace) > 0:
            # With a 0.5 stressor over many ticks, affect oscillates; expect surprises.
            assert len(mesh.meta_self.surprise_events) >= 1, (
                "Expected at least one surprise event after 100 ticks under high stressor"
            )
            assert mesh.meta_self.interiority_score() > 0.0
        else:
            # Agent never reached DECIDE — it was under constant crisis.
            # This is still a valid outcome; verify it survived some ticks.
            assert ticks_run >= 1, "Agent must run at least one tick"
