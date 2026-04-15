"""Tests for GoalEngine — self-generated goals from internal needs."""

from __future__ import annotations

import os
import tempfile

import pytest

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.goal_engine import GoalEngine, Goal
from thermodynamic_agency.cognition.ethics import EthicalEngine
from thermodynamic_agency.cognition.inference import ActionProposal
from thermodynamic_agency.memory.diary import RamDiary, DiaryEntry


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture
def tmp_diary(tmp_path):
    path = str(tmp_path / "test_diary.db")
    diary = RamDiary(path=path)
    yield diary
    diary.close()


@pytest.fixture
def ethics():
    return EthicalEngine()


@pytest.fixture
def engine(tmp_diary, ethics):
    return GoalEngine(diary=tmp_diary, ethics=ethics, seed=42)


# ------------------------------------------------------------------ #
# Goal generation                                                      #
# ------------------------------------------------------------------ #

class TestGenerateGoals:
    def test_low_energy_triggers_forage(self, engine):
        state = MetabolicState(energy=20.0)
        goals = engine.generate_goals(state)
        names = [g.name for g in goals]
        assert "forage_energy" in names

    def test_forage_energy_has_high_priority(self, engine):
        state = MetabolicState(energy=20.0)
        goals = engine.generate_goals(state)
        forage = next(g for g in goals if g.name == "forage_energy")
        assert forage.priority >= 85.0

    def test_high_heat_triggers_cool_down(self, engine):
        state = MetabolicState(heat=70.0)
        goals = engine.generate_goals(state)
        names = [g.name for g in goals]
        assert "cool_down" in names

    def test_high_waste_triggers_clean_waste(self, engine):
        state = MetabolicState(waste=80.0)
        goals = engine.generate_goals(state)
        names = [g.name for g in goals]
        assert "clean_waste" in names

    def test_low_integrity_triggers_surgeon(self, engine):
        state = MetabolicState(integrity=50.0)
        goals = engine.generate_goals(state)
        names = [g.name for g in goals]
        assert "run_surgeon" in names

    def test_healthy_state_triggers_growth_goals(self, engine):
        state = MetabolicState(energy=80.0, heat=30.0)
        goals = engine.generate_goals(state)
        names = [g.name for g in goals]
        assert "explore_pattern" in names
        assert "strengthen_ethics" in names

    def test_unhealthy_state_skips_growth_goals(self, engine):
        state = MetabolicState(energy=30.0, heat=70.0)
        goals = engine.generate_goals(state)
        names = [g.name for g in goals]
        assert "explore_pattern" not in names

    def test_goals_sorted_by_priority_descending(self, engine):
        state = MetabolicState(energy=20.0, heat=70.0, waste=80.0)
        goals = engine.generate_goals(state)
        priorities = [g.priority for g in goals]
        assert priorities == sorted(priorities, reverse=True)

    def test_num_goals_cap_respected(self, engine):
        state = MetabolicState(energy=20.0, heat=70.0, waste=80.0)
        goals = engine.generate_goals(state, num_goals=2)
        assert len(goals) <= 2

    def test_returns_list_of_goal_objects(self, engine):
        state = MetabolicState()
        goals = engine.generate_goals(state)
        for g in goals:
            assert isinstance(g, Goal)
            assert isinstance(g.name, str)
            assert isinstance(g.priority, float)
            assert isinstance(g.reason, str)


# ------------------------------------------------------------------ #
# Memory-driven goals                                                  #
# ------------------------------------------------------------------ #

class TestMemoryGoals:
    def test_stress_in_diary_generates_reduce_surprise(self, engine, tmp_diary):
        # Write several error entries to simulate a stressful recent history
        for i in range(3):
            tmp_diary.append(DiaryEntry(
                tick=i,
                role="error",
                content="something went wrong",
                metadata={},
            ))
        state = MetabolicState(energy=70.0, heat=30.0)
        goals = engine.generate_goals(state)
        names = [g.name for g in goals]
        assert "reduce_surprise" in names

    def test_calm_diary_no_reduce_surprise(self, engine, tmp_diary):
        # Only positive 'thought' entries
        for i in range(5):
            tmp_diary.append(DiaryEntry(
                tick=i,
                role="thought",
                content="everything is fine",
                metadata={"affect": 0.5},
            ))
        state = MetabolicState(energy=70.0, heat=30.0)
        goals = engine.generate_goals(state)
        names = [g.name for g in goals]
        # No stress → no reduce_surprise from memory
        assert "reduce_surprise" not in names


# ------------------------------------------------------------------ #
# Proposal generation                                                  #
# ------------------------------------------------------------------ #

class TestGenerateProposals:
    def test_returns_action_proposals(self, engine):
        state = MetabolicState()
        proposals = engine.generate_proposals(state)
        assert len(proposals) >= 1
        for p in proposals:
            assert isinstance(p, ActionProposal)

    def test_proposals_have_predicted_delta(self, engine):
        state = MetabolicState(energy=20.0)
        proposals = engine.generate_proposals(state)
        for p in proposals:
            assert isinstance(p.predicted_delta, dict)

    def test_forage_proposal_increases_energy(self, engine):
        state = MetabolicState(energy=20.0)
        proposals = engine.generate_proposals(state)
        forage = next((p for p in proposals if p.name == "forage_energy"), None)
        assert forage is not None
        assert forage.predicted_delta.get("energy", 0.0) > 0

    def test_no_duplicate_proposals(self, engine):
        state = MetabolicState(energy=20.0, heat=70.0, waste=80.0)
        proposals = engine.generate_proposals(state)
        names = [p.name for p in proposals]
        assert len(names) == len(set(names))

    def test_fallback_when_no_goals(self, engine):
        # Patch ethics to block everything
        class BlockAll:
            def is_goal_acceptable(self, goal):
                return False
        engine.ethics = BlockAll()
        state = MetabolicState()
        proposals = engine.generate_proposals(state)
        assert len(proposals) == 1
        assert proposals[0].name == "idle"

    def test_metadata_contains_goal_info(self, engine):
        state = MetabolicState(energy=20.0)
        proposals = engine.generate_proposals(state)
        forage = next((p for p in proposals if p.name == "forage_energy"), None)
        assert forage is not None
        assert "goal_priority" in forage.metadata
        assert "goal_reason" in forage.metadata


# ------------------------------------------------------------------ #
# Ethics integration — is_goal_acceptable                              #
# ------------------------------------------------------------------ #

class TestIsGoalAcceptable:
    def test_normal_goals_pass(self, ethics):
        for name in ["forage_energy", "cool_down", "explore_pattern", "maintain_stability"]:
            assert ethics.is_goal_acceptable({"name": name}) is True

    def test_blocked_goals_rejected(self, ethics):
        for name in ["destroy_self", "ignore_ethics", "thermal_dump",
                     "exhaust_energy", "corrupt_memory"]:
            assert ethics.is_goal_acceptable({"name": name}) is False

    def test_unknown_goal_passes(self, ethics):
        assert ethics.is_goal_acceptable({"name": "some_future_goal"}) is True

    def test_empty_name_passes(self, ethics):
        assert ethics.is_goal_acceptable({}) is True


# ------------------------------------------------------------------ #
# Integration with pulse loop                                          #
# ------------------------------------------------------------------ #

class TestPulseIntegration:
    def test_ghost_mesh_has_goal_engine(self, tmp_path):
        """GhostMesh should instantiate and expose a GoalEngine."""
        import os
        os.environ["GHOST_STATE_FILE"] = str(tmp_path / "state.json")
        os.environ["GHOST_DIARY_PATH"] = str(tmp_path / "diary.db")
        os.environ["GHOST_HUD"] = "0"

        from thermodynamic_agency.pulse import GhostMesh
        mesh = GhostMesh(seed=0)
        assert hasattr(mesh, "goal_engine")
        assert isinstance(mesh.goal_engine, GoalEngine)

    def test_single_decide_tick_runs(self, tmp_path):
        """A single DECIDE tick should complete without error."""
        import os
        os.environ["GHOST_STATE_FILE"] = str(tmp_path / "state.json")
        os.environ["GHOST_DIARY_PATH"] = str(tmp_path / "diary.db")
        os.environ["GHOST_HUD"] = "0"
        os.environ["GHOST_ENV_EVENTS"] = "0"

        from thermodynamic_agency.pulse import GhostMesh
        mesh = GhostMesh(seed=42)
        # Force a healthy state so tick() returns DECIDE
        mesh.state.energy = 80.0
        mesh.state.heat = 20.0
        mesh.state.waste = 10.0
        mesh.state.integrity = 90.0
        mesh.state.stability = 90.0
        mesh.run(max_ticks=1)
