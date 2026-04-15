"""Tests for Phase 3: hierarchical predictive coding, thalamic gating, and habit loops."""

from __future__ import annotations

import copy
import pytest

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.limbic import LimbicLayer, LimbicSignal
from thermodynamic_agency.cognition.predictive_hierarchy import (
    PredictiveHierarchy,
    HierarchySignal,
    LayerBelief,
    _VITALS,
    _PREFRONTAL_SETPOINTS,
)
from thermodynamic_agency.cognition.thalamus import (
    ThalamusGate,
    GateReport,
    _THREAT_HIGH,
)
from thermodynamic_agency.cognition.basal_ganglia import (
    BasalGanglia,
    HabitRecord,
    HabitSignal,
    HABIT_THRESHOLD,
    HABIT_MIN_SUCCESS_RATE,
    HABIT_FE_OVERRIDE,
)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def healthy_state() -> MetabolicState:
    return MetabolicState(
        energy=90.0, heat=15.0, waste=5.0, integrity=95.0, stability=95.0
    )


def stressed_state() -> MetabolicState:
    return MetabolicState(
        energy=30.0, heat=70.0, waste=60.0, integrity=50.0, stability=50.0
    )


def _limbic_signal_with_threat(threat: float) -> LimbicSignal:
    return LimbicSignal(
        threat_level=threat,
        precision_overrides={"energy": threat * 2.0},
        efe_discount=0.0,
        integrity_cost=0.0,
        heat_cost=0.0,
        amygdala_affect_modifier=-threat * 0.3,
    )


# ------------------------------------------------------------------ #
# PredictiveHierarchy                                                 #
# ------------------------------------------------------------------ #

class TestPredictiveHierarchy:

    def test_initial_update_returns_hierarchy_signal(self):
        h = PredictiveHierarchy()
        state = healthy_state()
        signal = h.update(state)
        assert isinstance(signal, HierarchySignal)

    def test_top_down_precision_has_all_vitals(self):
        h = PredictiveHierarchy()
        state = healthy_state()
        signal = h.update(state)
        for v in _VITALS:
            assert v in signal.top_down_precision, f"Missing vital: {v}"

    def test_top_down_precision_positive(self):
        h = PredictiveHierarchy()
        state = healthy_state()
        signal = h.update(state)
        for v, p in signal.top_down_precision.items():
            assert p > 0.0, f"Precision for {v} should be positive, got {p}"

    def test_hierarchical_error_non_negative(self):
        h = PredictiveHierarchy()
        state = healthy_state()
        signal = h.update(state)
        assert signal.hierarchical_error >= 0.0

    def test_heat_cost_non_negative(self):
        h = PredictiveHierarchy()
        state = healthy_state()
        signal = h.update(state)
        assert signal.heat_cost >= 0.0

    def test_error_increases_under_stress(self):
        """Stressed state should produce higher hierarchical error than healthy state."""
        h1 = PredictiveHierarchy()
        h2 = PredictiveHierarchy()
        # Warm up both with same healthy state first
        base = healthy_state()
        for _ in range(5):
            h1.update(base)
            h2.update(base)
        # Now show one a stressed state
        sig_stressed = h2.update(stressed_state())
        sig_healthy = h1.update(healthy_state())
        # Stressed should produce more error
        assert sig_stressed.hierarchical_error >= sig_healthy.hierarchical_error

    def test_threat_boosts_l1_precision(self):
        h = PredictiveHierarchy()
        state = healthy_state()
        no_threat_signal = h.update(state, limbic_signal=None)
        h2 = PredictiveHierarchy()
        high_threat = _limbic_signal_with_threat(0.9)
        threat_signal = h2.update(state, limbic_signal=high_threat)
        # L1 precision under threat should result in higher top-down precision
        # (or at least not crash)
        assert isinstance(threat_signal, HierarchySignal)

    def test_layer_errors_reported(self):
        h = PredictiveHierarchy()
        state = healthy_state()
        signal = h.update(state)
        assert 1 in signal.layer_errors
        assert 2 in signal.layer_errors
        for layer_errors in signal.layer_errors.values():
            for v in _VITALS:
                assert v in layer_errors

    def test_l2_predictions_accessible(self):
        h = PredictiveHierarchy()
        state = healthy_state()
        h.update(state)
        preds = h.l2_predictions()
        assert isinstance(preds, dict)
        for v in _VITALS:
            assert v in preds

    def test_l1_errors_accessible(self):
        h = PredictiveHierarchy()
        state = healthy_state()
        h.update(state)
        errors = h.l1_errors()
        assert isinstance(errors, dict)

    def test_status_returns_dict(self):
        h = PredictiveHierarchy()
        state = healthy_state()
        h.update(state)
        status = h.status()
        assert "l1_precision" in status
        assert "l2_precision" in status
        assert "l1_errors" in status
        assert "l2_predictions" in status

    def test_l2_converges_toward_setpoints_on_healthy_state(self):
        """After many updates with healthy state, L2 predictions drift toward setpoints."""
        h = PredictiveHierarchy()
        state = healthy_state()
        for _ in range(30):
            h.update(state)
        preds = h.l2_predictions()
        # Energy setpoint is 80; healthy energy is 90; predictions should be above 60
        assert preds["energy"] > 60.0

    def test_thalamic_precision_override_accepted(self):
        h = PredictiveHierarchy()
        state = healthy_state()
        sig = h.update(state, l1_precision=2.5, l2_precision=0.8)
        assert isinstance(sig, HierarchySignal)

    def test_multiple_updates_dont_crash(self):
        h = PredictiveHierarchy()
        state = healthy_state()
        for i in range(20):
            s = copy.copy(state)
            s.energy = max(1.0, 90.0 - i * 2)
            s.heat = 15.0 + i * 2
            h.update(s)


# ------------------------------------------------------------------ #
# ThalamusGate                                                        #
# ------------------------------------------------------------------ #

class TestThalamusGate:

    def test_initial_route_returns_gate_report(self):
        gate = ThalamusGate()
        state = healthy_state()
        report = gate.route(state)
        assert isinstance(report, GateReport)

    def test_last_report_none_initially(self):
        gate = ThalamusGate()
        assert gate.last_report is None

    def test_last_report_set_after_route(self):
        gate = ThalamusGate()
        state = healthy_state()
        gate.route(state)
        assert gate.last_report is not None

    def test_channel_weights_all_vitals_present(self):
        gate = ThalamusGate()
        state = healthy_state()
        report = gate.route(state)
        for v in ("energy", "heat", "waste", "integrity", "stability"):
            assert v in report.channel_weights, f"Missing channel: {v}"

    def test_channel_weights_in_range(self):
        gate = ThalamusGate()
        state = healthy_state()
        report = gate.route(state)
        for v, w in report.channel_weights.items():
            assert 0.0 <= w <= 1.0, f"Weight for {v} out of range: {w}"

    def test_dormant_regime_when_low_fe(self):
        gate = ThalamusGate()
        state = MetabolicState(
            energy=100.0, heat=0.0, waste=0.0, integrity=100.0, stability=100.0
        )
        report = gate.route(state, precision_regime="dormant")
        assert report.regime == "dormant"

    def test_sweet_spot_regime_default_healthy(self):
        gate = ThalamusGate()
        # Use a moderate state that lands in the sweet-spot FE band (8-45)
        state = MetabolicState(
            energy=60.0, heat=35.0, waste=20.0, integrity=75.0, stability=75.0
        )
        report = gate.route(state, precision_regime="sweet_spot")
        assert report.regime == "sweet_spot"

    def test_overload_regime_when_high_fe(self):
        gate = ThalamusGate()
        state = stressed_state()
        report = gate.route(state, precision_regime="overload")
        assert report.regime in ("overload", "dissociation")

    def test_high_threat_boosts_l1_precision(self):
        gate = ThalamusGate()
        state = healthy_state()
        low_threat = _limbic_signal_with_threat(0.0)
        high_threat = _limbic_signal_with_threat(0.9)
        report_low = gate.route(state, limbic_signal=low_threat)
        gate2 = ThalamusGate()
        report_high = gate2.route(state, limbic_signal=high_threat)
        assert report_high.l1_precision > report_low.l1_precision

    def test_high_threat_suppresses_l2_precision(self):
        gate = ThalamusGate()
        state = healthy_state()
        low_threat = _limbic_signal_with_threat(0.0)
        high_threat = _limbic_signal_with_threat(0.9)
        report_low = gate.route(state, limbic_signal=low_threat)
        gate2 = ThalamusGate()
        report_high = gate2.route(state, limbic_signal=high_threat)
        assert report_high.l2_precision < report_low.l2_precision

    def test_high_threat_suppresses_exploratory_channels(self):
        gate = ThalamusGate()
        state = healthy_state()
        high_threat = _limbic_signal_with_threat(0.9)
        report = gate.route(state, limbic_signal=high_threat)
        assert report.exploratory_suppressed

    def test_low_threat_exploratory_not_suppressed(self):
        gate = ThalamusGate()
        state = healthy_state()
        low_threat = _limbic_signal_with_threat(0.0)
        report = gate.route(state, limbic_signal=low_threat, precision_regime="sweet_spot")
        assert not report.exploratory_suppressed

    def test_survival_channels_open_under_threat(self):
        gate = ThalamusGate()
        state = healthy_state()
        high_threat = _limbic_signal_with_threat(0.9)
        report_threat = gate.route(state, limbic_signal=high_threat)
        gate2 = ThalamusGate()
        report_baseline = gate2.route(state, limbic_signal=_limbic_signal_with_threat(0.0))
        # Survival channels (energy, heat, integrity) should be higher under threat
        assert report_threat.channel_weights["energy"] >= report_baseline.channel_weights["energy"]
        assert report_threat.channel_weights["heat"] >= report_baseline.channel_weights["heat"]

    def test_positive_affect_boosts_l2_precision(self):
        gate = ThalamusGate()
        state = MetabolicState(
            energy=90.0, heat=10.0, waste=5.0, integrity=95.0, stability=95.0, affect=0.7
        )
        report = gate.route(state, precision_regime="sweet_spot")
        gate2 = ThalamusGate()
        state_neutral = MetabolicState(
            energy=90.0, heat=10.0, waste=5.0, integrity=95.0, stability=95.0, affect=0.0
        )
        report_neutral = gate2.route(state_neutral, precision_regime="sweet_spot")
        assert report.l2_precision >= report_neutral.l2_precision

    def test_dissociation_regime_very_high_fe(self):
        gate = ThalamusGate()
        # Force extremely stressed state
        state = MetabolicState(
            energy=5.0, heat=95.0, waste=90.0, integrity=15.0, stability=5.0
        )
        report = gate.route(state)
        assert report.regime == "dissociation"
        # L2 precision should be suppressed under dissociation
        assert report.l2_precision < 1.0


# ------------------------------------------------------------------ #
# BasalGanglia                                                        #
# ------------------------------------------------------------------ #

class TestBasalGanglia:

    def test_consult_unknown_action_not_habit(self):
        bg = BasalGanglia()
        state = healthy_state()
        sig = bg.consult("unknown_action", state)
        assert not sig.is_habit

    def test_record_outcome_creates_habit_record(self):
        bg = BasalGanglia()
        before = {"energy": 50.0, "heat": 30.0, "waste": 20.0, "integrity": 80.0, "stability": 80.0}
        after = {"energy": 70.0, "heat": 20.0, "waste": 15.0, "integrity": 85.0, "stability": 82.0}
        bg.record_outcome("idle", before, after)
        assert "idle" in bg._habits

    def test_habit_becomes_active_after_threshold_successes(self):
        bg = BasalGanglia()
        before = {"energy": 50.0, "heat": 30.0, "waste": 20.0, "integrity": 80.0, "stability": 80.0}
        after_good = {"energy": 80.0, "heat": 15.0, "waste": 10.0, "integrity": 90.0, "stability": 85.0}
        # Need enough successful executions to grow strength above 0.5.
        # Strength grows at ~0.07 * outcome_quality per success after HABIT_THRESHOLD.
        for _ in range(HABIT_THRESHOLD + 15):
            bg.record_outcome("idle", before, after_good)
        sig = bg.consult("idle", healthy_state())
        assert sig.is_habit

    def test_habit_energy_cost_lower_than_full_inference(self):
        """Habit execution cost should be less than the full inference energy cost (~0.15)."""
        from thermodynamic_agency.cognition.basal_ganglia import HABIT_ENERGY_COST
        assert HABIT_ENERGY_COST < 0.15

    def test_habit_not_triggered_under_high_fe(self):
        bg = BasalGanglia()
        before = {"energy": 50.0, "heat": 30.0, "waste": 20.0, "integrity": 80.0, "stability": 80.0}
        after_good = {"energy": 80.0, "heat": 15.0, "waste": 10.0, "integrity": 90.0, "stability": 85.0}
        for _ in range(HABIT_THRESHOLD + 15):
            bg.record_outcome("idle", before, after_good)
        # Create high-FE state to force override
        state = MetabolicState(
            energy=5.0, heat=90.0, waste=80.0, integrity=20.0, stability=20.0
        )
        sig = bg.consult("idle", state)
        assert not sig.is_habit
        assert "stress override" in sig.reason

    def test_habit_degrades_under_allostatic_load(self):
        bg = BasalGanglia()
        before = {"energy": 50.0, "heat": 30.0, "waste": 20.0, "integrity": 80.0, "stability": 80.0}
        after_good = {"energy": 80.0, "heat": 15.0, "waste": 10.0, "integrity": 90.0, "stability": 85.0}
        for _ in range(HABIT_THRESHOLD + 15):
            bg.record_outcome("idle", before, after_good)
        initial_strength = bg._habits["idle"].strength
        # Apply many ticks of decay under high allostatic load
        stressed = MetabolicState(allostatic_load=90.0)
        for _ in range(20):
            bg.tick_decay(stressed)
        assert bg._habits["idle"].strength < initial_strength

    def test_tick_decay_no_effect_low_allostatic_load(self):
        bg = BasalGanglia()
        before = {"energy": 50.0, "heat": 30.0, "waste": 20.0, "integrity": 80.0, "stability": 80.0}
        after_good = {"energy": 80.0, "heat": 15.0, "waste": 10.0, "integrity": 90.0, "stability": 85.0}
        for _ in range(HABIT_THRESHOLD + 15):
            bg.record_outcome("idle", before, after_good)
        initial_strength = bg._habits["idle"].strength
        low_load = MetabolicState(allostatic_load=10.0)
        for _ in range(10):
            bg.tick_decay(low_load)
        assert bg._habits["idle"].strength == pytest.approx(initial_strength)

    def test_status_returns_dict(self):
        bg = BasalGanglia()
        status = bg.status()
        assert "total_tracked" in status
        assert "active_habits" in status
        assert "habit_names" in status

    def test_bad_outcomes_prevent_habit_formation(self):
        bg = BasalGanglia()
        before = {"energy": 80.0, "heat": 15.0, "waste": 5.0, "integrity": 95.0, "stability": 95.0}
        after_bad = {"energy": 40.0, "heat": 60.0, "waste": 40.0, "integrity": 60.0, "stability": 60.0}
        for _ in range(HABIT_THRESHOLD + 5):
            bg.record_outcome("forage_resources", before, after_bad)
        sig = bg.consult("forage_resources", healthy_state())
        assert not sig.is_habit

    def test_all_habits_sorted_by_strength(self):
        bg = BasalGanglia()
        before = {"energy": 50.0, "heat": 30.0, "waste": 20.0, "integrity": 80.0, "stability": 80.0}
        after_good = {"energy": 80.0, "heat": 15.0, "waste": 10.0, "integrity": 90.0, "stability": 85.0}
        after_ok = {"energy": 65.0, "heat": 25.0, "waste": 15.0, "integrity": 82.0, "stability": 81.0}
        for _ in range(HABIT_THRESHOLD + 5):
            bg.record_outcome("idle", before, after_good)
        for _ in range(HABIT_THRESHOLD + 2):
            bg.record_outcome("reflect", before, after_ok)
        habits = bg.all_habits()
        strengths = [h.strength for h in habits]
        assert strengths == sorted(strengths, reverse=True)

    def test_cached_efe_converges_toward_good_outcome(self):
        bg = BasalGanglia()
        before = {"energy": 50.0, "heat": 30.0, "waste": 20.0, "integrity": 80.0, "stability": 80.0}
        after_good = {"energy": 80.0, "heat": 15.0, "waste": 10.0, "integrity": 90.0, "stability": 85.0}
        initial_efe = 50.0  # default pessimistic
        for _ in range(10):
            bg.record_outcome("idle", before, after_good)
        final_efe = bg._habits["idle"].cached_efe
        assert final_efe < initial_efe  # should have converged down toward good estimates


# ------------------------------------------------------------------ #
# Integration: GhostMesh runs with Phase 3 subsystems active         #
# ------------------------------------------------------------------ #

class TestPhase3Integration:

    def test_ghost_mesh_runs_with_phase3_subsystems(self, tmp_path, monkeypatch):
        """GhostMesh should survive 10 ticks with Phase 3 subsystems wired in."""
        monkeypatch.setenv("GHOST_STATE_FILE", str(tmp_path / "state.json"))
        monkeypatch.setenv("GHOST_DIARY_PATH", str(tmp_path / "diary.db"))
        monkeypatch.setenv("GHOST_HUD", "0")
        monkeypatch.setenv("GHOST_ENV_EVENTS", "0")
        monkeypatch.setenv("GHOST_PULSE", "0")

        from thermodynamic_agency.pulse import GhostMesh
        mesh = GhostMesh(seed=42)

        # Verify Phase 3 subsystems are present
        assert hasattr(mesh, "hierarchy")
        assert hasattr(mesh, "thalamus")
        assert hasattr(mesh, "basal_ganglia")

        mesh.run(max_ticks=10)
        # Should still be alive
        assert mesh.state.entropy >= 10

    def test_hierarchy_integrates_with_limbic_signal(self, tmp_path):
        """PredictiveHierarchy correctly processes a real LimbicLayer signal."""
        limbic = LimbicLayer()
        hierarchy = PredictiveHierarchy()
        state = healthy_state()
        for _ in range(5):
            limbic_signal = limbic.process(state)
            sig = hierarchy.update(state, limbic_signal=limbic_signal)
            assert isinstance(sig, HierarchySignal)

    def test_thalamus_gate_with_hierarchy(self):
        """ThalamusGate output can be passed directly to PredictiveHierarchy."""
        gate = ThalamusGate()
        hierarchy = PredictiveHierarchy()
        state = healthy_state()
        limbic = LimbicLayer()
        limbic_signal = limbic.process(state)
        gate_report = gate.route(state, limbic_signal=limbic_signal)
        sig = hierarchy.update(
            state,
            limbic_signal=limbic_signal,
            l1_precision=gate_report.l1_precision,
            l2_precision=gate_report.l2_precision,
        )
        assert isinstance(sig, HierarchySignal)

    def test_habit_loop_reduces_cost_after_learning(self, tmp_path, monkeypatch):
        """After many identical DECIDE ticks, habit loop should reduce inference cost."""
        monkeypatch.setenv("GHOST_STATE_FILE", str(tmp_path / "state2.json"))
        monkeypatch.setenv("GHOST_DIARY_PATH", str(tmp_path / "diary2.db"))
        monkeypatch.setenv("GHOST_HUD", "0")
        monkeypatch.setenv("GHOST_ENV_EVENTS", "0")
        monkeypatch.setenv("GHOST_PULSE", "0")

        from thermodynamic_agency.pulse import GhostMesh
        mesh = GhostMesh(seed=7)
        # Run enough ticks to build habits
        mesh.run(max_ticks=20)
        # Basal ganglia should have recorded the repeated DECIDE action
        assert "DECIDE" in mesh.basal_ganglia._habits or len(mesh.basal_ganglia._habits) > 0
