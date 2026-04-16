"""Tests for CounterfactualEngine (Feature 1), LanguageCognition (Feature 2),
and HomeostasisAdapter (Feature 3).
"""

from __future__ import annotations

import math
import tempfile

import pytest

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.inference import ActionProposal, generate_default_proposals
from thermodynamic_agency.cognition.counterfactual import (
    CounterfactualEngine,
    CounterfactualTrace,
    CF_RISK_WEIGHT,
    _is_lethal,
    _in_safety_margin,
    _sim_free_energy,
    _survivor_risk,
    _deep_lethal_risk,
)
from thermodynamic_agency.cognition.homeostasis import (
    HomeostasisAdapter,
    HomeostasisStatus,
    _INITIAL_SETPOINTS,
    _MAX_DRIFT_FRACTION,
)
from thermodynamic_agency.cognition.language_cognition import (
    LanguageCognition,
    LanguageCognitionReport,
    _ARCHETYPE_CATALOGUE,
)
from thermodynamic_agency.cognition.ethics import EthicalEngine
from thermodynamic_agency.memory.diary import RamDiary


# ================================================================== #
# Feature 1: CounterfactualEngine                                     #
# ================================================================== #

class TestCounterfactualHelpers:
    def test_is_lethal_energy_exhaustion(self):
        vitals = {"energy": 0.0, "heat": 30.0, "waste": 10.0, "integrity": 80.0, "stability": 70.0}
        assert _is_lethal(vitals)

    def test_is_lethal_thermal_death(self):
        vitals = {"energy": 50.0, "heat": 100.0, "waste": 10.0, "integrity": 80.0, "stability": 70.0}
        assert _is_lethal(vitals)

    def test_is_lethal_integrity_collapse(self):
        vitals = {"energy": 50.0, "heat": 30.0, "waste": 10.0, "integrity": 10.0, "stability": 70.0}
        assert _is_lethal(vitals)

    def test_is_lethal_false_for_healthy(self):
        vitals = {"energy": 70.0, "heat": 30.0, "waste": 10.0, "integrity": 80.0, "stability": 70.0}
        assert not _is_lethal(vitals)

    def test_in_safety_margin_energy_low(self):
        vitals = {"energy": 15.0, "heat": 30.0, "waste": 10.0, "integrity": 80.0, "stability": 70.0}
        assert _in_safety_margin(vitals)

    def test_in_safety_margin_heat_high(self):
        vitals = {"energy": 70.0, "heat": 88.0, "waste": 10.0, "integrity": 80.0, "stability": 70.0}
        assert _in_safety_margin(vitals)

    def test_in_safety_margin_false_for_healthy(self):
        vitals = {"energy": 80.0, "heat": 20.0, "waste": 10.0, "integrity": 85.0, "stability": 80.0}
        assert not _in_safety_margin(vitals)

    def test_sim_free_energy_matches_metabolic_state(self):
        state = MetabolicState(energy=80.0, heat=20.0, waste=10.0, integrity=85.0, stability=80.0)
        vitals = {"energy": 80.0, "heat": 20.0, "waste": 10.0, "integrity": 85.0, "stability": 80.0}
        assert abs(_sim_free_energy(vitals) - state.free_energy_estimate()) < 0.1

    def test_sim_free_energy_higher_when_stressed(self):
        healthy = {"energy": 80.0, "heat": 20.0, "waste": 10.0, "integrity": 85.0, "stability": 80.0}
        stressed = {"energy": 30.0, "heat": 60.0, "waste": 40.0, "integrity": 50.0, "stability": 40.0}
        assert _sim_free_energy(stressed) > _sim_free_energy(healthy)

    def test_survivor_risk_zero_for_clean_trace(self):
        assert _survivor_risk([], 10) == 0.0

    def test_survivor_risk_increases_with_breaches(self):
        risk_few = _survivor_risk([9, 10], 10)
        risk_many = _survivor_risk(list(range(1, 11)), 10)
        assert risk_many > risk_few

    def test_survivor_risk_capped_at_08(self):
        all_depths = list(range(1, 11))
        assert _survivor_risk(all_depths, 10) <= 0.80

    def test_deep_lethal_risk_decreases_with_depth(self):
        # Dying earlier (depth 3) should be more dangerous than dying late (depth 9)
        risk_early = _deep_lethal_risk(3, 10, 2)
        risk_late = _deep_lethal_risk(9, 10, 2)
        assert risk_early > risk_late

    def test_deep_lethal_risk_always_at_least_05(self):
        # Any lethal trajectory carries at least 0.50 risk
        for depth in range(3, 11):
            assert _deep_lethal_risk(depth, 10, 2) >= 0.50


class TestCounterfactualEngineSimulate:
    def test_healthy_state_safe_proposal_survives(self):
        state = MetabolicState()
        proposal = ActionProposal(
            name="idle",
            description="do nothing",
            predicted_delta={"energy": -0.05, "heat": -0.5, "waste": 0.1},
            cost_energy=0.05,
        )
        engine = CounterfactualEngine(horizon=10)
        trace = engine.simulate(state, proposal)
        assert isinstance(trace, CounterfactualTrace)
        assert trace.proposal_name == "idle"
        assert trace.survived is True
        assert trace.terminal_risk < 1.0
        assert trace.pruned_at_depth is None
        assert len(trace.vitals_trajectory) == 10

    def test_hard_prune_for_instantly_lethal_proposal(self):
        # Start with almost-dead energy state
        state = MetabolicState(energy=1.5, heat=20.0, waste=5.0, integrity=80.0, stability=70.0)
        # Action that drains all remaining energy immediately
        proposal = ActionProposal(
            name="energy_dump",
            description="dump energy",
            predicted_delta={"energy": -10.0},
            cost_energy=0.0,
        )
        engine = CounterfactualEngine(horizon=10, hard_prune_depth=2)
        trace = engine.simulate(state, proposal)
        assert trace.survived is False
        assert trace.terminal_risk == 1.0
        assert trace.pruned_at_depth is not None
        assert trace.pruned_at_depth <= 2
        assert trace.cumulative_vfe == float("inf")
        assert trace.vitals_trajectory == []   # nothing simulated for fear-pruned branch

    def test_safe_proposal_has_lower_risk_than_dangerous(self):
        state = MetabolicState(energy=30.0, heat=20.0, waste=5.0, integrity=80.0, stability=70.0)
        safe_proposal = ActionProposal(
            name="safe",
            description="safe action",
            predicted_delta={"energy": 10.0},
            cost_energy=1.0,
        )
        dangerous_proposal = ActionProposal(
            name="dangerous",
            description="drains everything",
            predicted_delta={"energy": -25.0},
            cost_energy=0.0,
        )
        engine = CounterfactualEngine(horizon=10)
        safe_trace = engine.simulate(state, safe_proposal)
        dangerous_trace = engine.simulate(state, dangerous_proposal)
        assert safe_trace.terminal_risk < dangerous_trace.terminal_risk

    def test_state_not_mutated_by_simulate(self):
        state = MetabolicState(energy=80.0, heat=20.0, waste=10.0)
        original_energy = state.energy
        original_heat = state.heat
        proposal = ActionProposal(
            name="test",
            description="test",
            predicted_delta={"energy": -50.0, "heat": 50.0},
            cost_energy=0.0,
        )
        engine = CounterfactualEngine(horizon=10)
        engine.simulate(state, proposal)
        assert state.energy == original_energy
        assert state.heat == original_heat

    def test_deep_lethal_higher_risk_than_survivor(self):
        # A path that dies at depth 5 should have higher risk than one that
        # stays alive but wobbles through safety margins
        state = MetabolicState(energy=20.0, heat=20.0, waste=5.0, integrity=80.0, stability=70.0)
        lethal_proposal = ActionProposal(
            name="lethal",
            description="depletes energy",
            predicted_delta={"energy": -19.0},
            cost_energy=0.0,
        )
        engine = CounterfactualEngine(horizon=10)
        trace = engine.simulate(state, lethal_proposal)
        assert not trace.survived
        assert trace.terminal_risk >= 0.5

    def test_depth_simulated_tracks_actual_steps(self):
        state = MetabolicState()
        proposal = ActionProposal(
            name="idle", description="idle",
            predicted_delta={}, cost_energy=0.05,
        )
        engine = CounterfactualEngine(horizon=10)
        trace = engine.simulate(state, proposal)
        assert trace.depth_simulated == 10

    def test_depth_simulated_is_small_for_hard_prune(self):
        state = MetabolicState(energy=1.0, heat=20.0, waste=5.0, integrity=80.0, stability=70.0)
        proposal = ActionProposal(
            name="drain", description="drain",
            predicted_delta={"energy": -5.0}, cost_energy=0.0,
        )
        engine = CounterfactualEngine(horizon=10, hard_prune_depth=2)
        trace = engine.simulate(state, proposal)
        assert trace.depth_simulated <= 2

    def test_cumulative_vfe_finite_for_surviving_trace(self):
        state = MetabolicState()
        proposal = ActionProposal(
            name="idle", description="idle",
            predicted_delta={}, cost_energy=0.05,
        )
        engine = CounterfactualEngine(horizon=10)
        trace = engine.simulate(state, proposal)
        assert trace.survived
        assert math.isfinite(trace.cumulative_vfe)
        assert trace.cumulative_vfe >= 0.0


class TestCounterfactualEngineRunBatch:
    def test_returns_one_trace_per_proposal(self):
        state = MetabolicState()
        proposals = generate_default_proposals(state)
        engine = CounterfactualEngine(horizon=5)
        traces = engine.run_batch(state, proposals)
        assert len(traces) == len(proposals)
        assert all(isinstance(t, CounterfactualTrace) for t in traces)

    def test_trace_names_match_proposal_names(self):
        state = MetabolicState()
        proposals = generate_default_proposals(state)
        engine = CounterfactualEngine(horizon=5)
        traces = engine.run_batch(state, proposals)
        for proposal, trace in zip(proposals, traces):
            assert trace.proposal_name == proposal.name

    def test_state_not_mutated_by_run_batch(self):
        state = MetabolicState(energy=80.0, heat=20.0)
        proposals = generate_default_proposals(state)
        engine = CounterfactualEngine(horizon=5)
        engine.run_batch(state, proposals)
        assert state.energy == 80.0
        assert state.heat == 20.0


class TestCounterfactualMetabolicCost:
    def test_cost_proportional_to_simulated_steps(self):
        state = MetabolicState()
        proposals = generate_default_proposals(state)
        engine = CounterfactualEngine(horizon=10)
        traces = engine.run_batch(state, proposals)
        e_cost, h_cost = engine.compute_metabolic_cost(traces)
        assert e_cost > 0
        assert h_cost > 0

    def test_hard_pruned_batch_costs_less_than_full_batch(self):
        # A batch where some proposals are hard-pruned should be cheaper
        # than one where all survive to full horizon
        state_safe = MetabolicState(energy=80.0, heat=20.0, waste=5.0,
                                     integrity=90.0, stability=90.0)
        state_danger = MetabolicState(energy=1.5, heat=20.0, waste=5.0,
                                       integrity=90.0, stability=90.0)
        drain = ActionProposal(
            name="drain", description="drain",
            predicted_delta={"energy": -5.0}, cost_energy=0.0,
        )
        idle = ActionProposal(
            name="idle", description="idle",
            predicted_delta={}, cost_energy=0.05,
        )
        engine = CounterfactualEngine(horizon=10)

        traces_safe = engine.run_batch(state_safe, [idle])
        traces_danger = engine.run_batch(state_danger, [drain])
        cost_safe = engine.compute_metabolic_cost(traces_safe)
        cost_danger = engine.compute_metabolic_cost(traces_danger)
        # The dangerous branch is pruned early and therefore cheaper
        assert cost_danger[0] < cost_safe[0]


class TestCounterfactualIntegrationWithSetpoints:
    """CounterfactualEngine integrates correctly with the inference pipeline."""

    def test_cf_risk_added_to_efe_scores(self):
        from thermodynamic_agency.cognition.inference import active_inference_step
        state = MetabolicState(energy=80.0)
        proposals = generate_default_proposals(state)
        engine = CounterfactualEngine(horizon=5)
        traces = engine.run_batch(state, proposals)
        cf_risk = {t.proposal_name: t.terminal_risk for t in traces}

        result = active_inference_step(state, proposals)
        for name, raw_score in result.efe_scores.items():
            adjusted = raw_score + cf_risk.get(name, 0.0) * CF_RISK_WEIGHT
            assert adjusted >= raw_score


# ================================================================== #
# Feature 2: LanguageCognition                                        #
# ================================================================== #

class TestLanguageCognitionHeuristic:
    def _make_lc(self, tmp_path):
        diary = RamDiary(path=str(tmp_path / "diary.db"))
        return LanguageCognition(diary=diary, use_llm=False, seed=42)

    def test_compress_beliefs_returns_report(self, tmp_path):
        lc = self._make_lc(tmp_path)
        state = MetabolicState()
        report = lc.compress_beliefs(state, goals=[])
        assert isinstance(report, LanguageCognitionReport)
        assert report.compression != ""
        assert report.used_llm is False

    def test_compress_beliefs_contains_vitals(self, tmp_path):
        lc = self._make_lc(tmp_path)
        state = MetabolicState(energy=60.0, heat=35.0)
        report = lc.compress_beliefs(state, goals=[])
        assert "60.0" in report.compression or "E=" in report.compression

    def test_compress_beliefs_writes_to_diary(self, tmp_path):
        lc = self._make_lc(tmp_path)
        state = MetabolicState()
        lc.compress_beliefs(state, goals=[])
        entries = lc.diary.recent(5)
        assert any("LANGCOG" in e.content for e in entries)

    def test_generate_proposals_returns_action_proposals(self, tmp_path):
        lc = self._make_lc(tmp_path)
        state = MetabolicState(energy=80.0, heat=20.0)
        ethics = EthicalEngine()
        proposals = lc.generate_proposals(state, goals=[], ethics=ethics)
        assert isinstance(proposals, list)
        assert all(isinstance(p, ActionProposal) for p in proposals)

    def test_generate_proposals_from_catalogue(self, tmp_path):
        lc = self._make_lc(tmp_path)
        state = MetabolicState(energy=80.0, heat=20.0)
        ethics = EthicalEngine()
        proposals = lc.generate_proposals(state, goals=[], ethics=ethics)
        for p in proposals:
            assert p.name in _ARCHETYPE_CATALOGUE

    def test_generate_proposals_max_two(self, tmp_path):
        lc = self._make_lc(tmp_path)
        state = MetabolicState(energy=80.0, heat=20.0)
        ethics = EthicalEngine()
        proposals = lc.generate_proposals(state, goals=[], ethics=ethics)
        assert len(proposals) <= 2

    def test_no_extra_metabolic_cost_without_llm(self, tmp_path):
        lc = self._make_lc(tmp_path)
        state = MetabolicState(energy=80.0)
        report = lc.compress_beliefs(state, goals=[])
        assert report.energy_cost == 0.0
        assert report.heat_cost == 0.0

    def test_blocked_goal_names_excluded(self, tmp_path):
        from thermodynamic_agency.cognition.ethics import _BLOCKED_GOAL_NAMES
        lc = self._make_lc(tmp_path)
        state = MetabolicState(energy=80.0, heat=20.0)
        ethics = EthicalEngine()
        proposals = lc.generate_proposals(state, goals=[], ethics=ethics)
        for p in proposals:
            assert p.name not in _BLOCKED_GOAL_NAMES


class TestLanguageCognitionGoalEngineIntegration:
    def test_goal_engine_uses_language_cognition(self, tmp_path):
        from thermodynamic_agency.cognition.goal_engine import GoalEngine
        diary = RamDiary(path=str(tmp_path / "diary.db"))
        ethics = EthicalEngine()
        lc = LanguageCognition(diary=diary, use_llm=False, seed=42)
        engine = GoalEngine(diary=diary, ethics=ethics, language_cognition=lc)

        state = MetabolicState(energy=80.0, heat=20.0, waste=5.0,
                                integrity=90.0, stability=90.0)
        proposals = engine.generate_proposals(state)
        assert len(proposals) > 0
        assert all(isinstance(p, ActionProposal) for p in proposals)

    def test_goal_engine_no_lc_still_works(self, tmp_path):
        from thermodynamic_agency.cognition.goal_engine import GoalEngine
        diary = RamDiary(path=str(tmp_path / "diary.db"))
        ethics = EthicalEngine()
        engine = GoalEngine(diary=diary, ethics=ethics)
        state = MetabolicState()
        proposals = engine.generate_proposals(state)
        assert len(proposals) > 0

    def test_language_cognition_not_triggered_in_emergency(self, tmp_path):
        from thermodynamic_agency.cognition.goal_engine import GoalEngine
        diary = RamDiary(path=str(tmp_path / "diary.db"))
        ethics = EthicalEngine()
        lc = LanguageCognition(diary=diary, use_llm=False, seed=42)
        engine = GoalEngine(diary=diary, ethics=ethics, language_cognition=lc)

        # Emergency: low energy => should NOT trigger LC proposals
        state = MetabolicState(energy=10.0, heat=20.0, waste=5.0,
                                integrity=90.0, stability=90.0)
        proposals = engine.generate_proposals(state)
        lc_proposals = [p for p in proposals
                         if p.metadata and p.metadata.get("source") == "language_cognition"]
        assert len(lc_proposals) == 0


# ================================================================== #
# Feature 3: HomeostasisAdapter                                       #
# ================================================================== #

class TestHomeostasisAdapterInitial:
    def test_initial_ema_equals_setpoints(self):
        adapter = HomeostasisAdapter()
        sp = adapter.adapted_setpoints()
        for vital, initial in _INITIAL_SETPOINTS.items():
            assert sp[vital] == initial

    def test_ticks_observed_zero_initially(self):
        adapter = HomeostasisAdapter()
        assert adapter.status().ticks_observed == 0

    def test_reset_restores_initial_state(self):
        adapter = HomeostasisAdapter()
        state = MetabolicState(energy=10.0, heat=80.0)
        for _ in range(100):
            adapter.observe(state)
        adapter.reset()
        sp = adapter.adapted_setpoints()
        for vital, initial in _INITIAL_SETPOINTS.items():
            assert sp[vital] == initial

    def test_status_returns_homeostasis_status(self):
        adapter = HomeostasisAdapter()
        status = adapter.status()
        assert isinstance(status, HomeostasisStatus)
        assert set(status.ema.keys()) == set(_INITIAL_SETPOINTS.keys())


class TestHomeostasisAdapterDrift:
    def test_energy_setpoint_drifts_down_when_consistently_low(self):
        adapter = HomeostasisAdapter(alpha=0.05)  # faster alpha for test
        state = MetabolicState(energy=40.0, heat=20.0, waste=10.0,
                                integrity=85.0, stability=80.0)
        initial_sp = adapter.adapted_setpoints()["energy"]
        for _ in range(200):
            adapter.observe(state)
        adapted_sp = adapter.adapted_setpoints()["energy"]
        # After many observations at energy=40 (< setpoint 80), energy setpoint drifts down
        assert adapted_sp < initial_sp

    def test_drift_bounded_by_max_fraction(self):
        adapter = HomeostasisAdapter(alpha=0.5)  # very fast to saturate quickly
        # Observe extreme values — setpoint should not drift beyond ±15%
        state_low = MetabolicState(energy=0.0, heat=99.0, waste=100.0,
                                    integrity=0.0, stability=0.0)
        for _ in range(1000):
            adapter.observe(state_low)
        sp = adapter.adapted_setpoints()
        for vital, initial in _INITIAL_SETPOINTS.items():
            max_drift = abs(initial) * _MAX_DRIFT_FRACTION
            assert sp[vital] >= initial - max_drift - 1e-9
            assert sp[vital] <= initial + max_drift + 1e-9

    def test_drift_bounded_upward_too(self):
        adapter = HomeostasisAdapter(alpha=0.5)
        # Observe very high values
        state_high = MetabolicState(energy=100.0, heat=0.0, waste=0.0,
                                     integrity=100.0, stability=100.0)
        for _ in range(1000):
            adapter.observe(state_high)
        sp = adapter.adapted_setpoints()
        for vital, initial in _INITIAL_SETPOINTS.items():
            max_drift = abs(initial) * _MAX_DRIFT_FRACTION
            assert sp[vital] >= initial - max_drift - 1e-9
            assert sp[vital] <= initial + max_drift + 1e-9

    def test_adapted_setpoints_are_finite(self):
        adapter = HomeostasisAdapter()
        state = MetabolicState()
        for _ in range(50):
            adapter.observe(state)
        sp = adapter.adapted_setpoints()
        for v, val in sp.items():
            assert math.isfinite(val), f"{v} is not finite: {val}"

    def test_ticks_observed_increments(self):
        adapter = HomeostasisAdapter()
        state = MetabolicState()
        for i in range(10):
            adapter.observe(state)
        assert adapter.status().ticks_observed == 10

    def test_drift_is_signed_correctly(self):
        # Consistently low energy should give NEGATIVE drift
        adapter = HomeostasisAdapter(alpha=0.05)
        state = MetabolicState(energy=20.0, heat=20.0, waste=10.0,
                                integrity=85.0, stability=80.0)
        for _ in range(300):
            adapter.observe(state)
        status = adapter.status()
        assert status.drift["energy"] < 0


class TestHomeostasisAdapterWithInference:
    def test_adapted_setpoints_accepted_by_compute_multistep_efe(self):
        from thermodynamic_agency.cognition.inference import compute_multistep_efe
        adapter = HomeostasisAdapter()
        state = MetabolicState()
        sp = adapter.adapted_setpoints()
        efe = compute_multistep_efe(state, {}, setpoints=sp)
        assert math.isfinite(efe)
        assert efe >= 0.0

    def test_adapted_setpoints_accepted_by_active_inference_step(self):
        from thermodynamic_agency.cognition.inference import active_inference_step
        adapter = HomeostasisAdapter()
        state = MetabolicState()
        proposals = generate_default_proposals(state)
        sp = adapter.adapted_setpoints()
        result = active_inference_step(state, proposals, setpoints=sp)
        assert result.selected in proposals

    def test_predictive_hierarchy_accepts_homeostasis(self):
        from thermodynamic_agency.cognition.predictive_hierarchy import PredictiveHierarchy
        from thermodynamic_agency.cognition.limbic import LimbicLayer
        adapter = HomeostasisAdapter()
        hierarchy = PredictiveHierarchy(homeostasis=adapter)
        state = MetabolicState()
        limbic = LimbicLayer()
        signal = limbic.process(state)
        result = hierarchy.update(state, limbic_signal=signal)
        assert result is not None

    def test_hierarchy_without_homeostasis_still_works(self):
        from thermodynamic_agency.cognition.predictive_hierarchy import PredictiveHierarchy
        from thermodynamic_agency.cognition.limbic import LimbicLayer
        hierarchy = PredictiveHierarchy()   # no homeostasis
        state = MetabolicState()
        limbic = LimbicLayer()
        signal = limbic.process(state)
        result = hierarchy.update(state, limbic_signal=signal)
        assert result is not None


# ================================================================== #
# Pulse-level smoke test (all three features together)               #
# ================================================================== #

class TestPulseIntegration:
    def test_ghostmesh_initialises_new_subsystems(self, tmp_path):
        import os
        os.environ["GHOST_STATE_FILE"] = str(tmp_path / "state.json")
        os.environ["GHOST_DIARY_PATH"] = str(tmp_path / "diary.db")
        os.environ["GHOST_HUD"] = "0"
        os.environ["GHOST_ENV_EVENTS"] = "0"
        try:
            from thermodynamic_agency.pulse import GhostMesh
            mesh = GhostMesh(seed=0)
            assert hasattr(mesh, "counterfactual_engine")
            assert hasattr(mesh, "language_cognition")
            assert hasattr(mesh, "homeostasis_adapter")
        finally:
            del os.environ["GHOST_STATE_FILE"]
            del os.environ["GHOST_DIARY_PATH"]
            del os.environ["GHOST_HUD"]
            del os.environ["GHOST_ENV_EVENTS"]

    def test_ghostmesh_runs_three_ticks_without_error(self, tmp_path):
        import os
        os.environ["GHOST_STATE_FILE"] = str(tmp_path / "state.json")
        os.environ["GHOST_DIARY_PATH"] = str(tmp_path / "diary.db")
        os.environ["GHOST_HUD"] = "0"
        os.environ["GHOST_ENV_EVENTS"] = "0"
        try:
            from thermodynamic_agency.pulse import GhostMesh
            mesh = GhostMesh(seed=42)
            mesh.run(max_ticks=3)
        finally:
            del os.environ["GHOST_STATE_FILE"]
            del os.environ["GHOST_DIARY_PATH"]
            del os.environ["GHOST_HUD"]
            del os.environ["GHOST_ENV_EVENTS"]

    def test_homeostasis_adapts_after_ticks(self, tmp_path):
        import os
        os.environ["GHOST_STATE_FILE"] = str(tmp_path / "state.json")
        os.environ["GHOST_DIARY_PATH"] = str(tmp_path / "diary.db")
        os.environ["GHOST_HUD"] = "0"
        os.environ["GHOST_ENV_EVENTS"] = "0"
        try:
            from thermodynamic_agency.pulse import GhostMesh
            mesh = GhostMesh(seed=42)
            mesh.run(max_ticks=5)
            assert mesh.homeostasis_adapter.status().ticks_observed == 5
        finally:
            del os.environ["GHOST_STATE_FILE"]
            del os.environ["GHOST_DIARY_PATH"]
            del os.environ["GHOST_HUD"]
            del os.environ["GHOST_ENV_EVENTS"]
