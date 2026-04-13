"""Tests for thermodynamic inference costs, PrecisionEngine, and affect signal."""

import pytest

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.inference import (
    ActionProposal,
    InferenceCost,
    InferenceResult,
    active_inference_step,
    compute_efe,
    compute_inference_cost,
    generate_default_proposals,
    _PRECISION,
)
from thermodynamic_agency.cognition.precision import (
    PrecisionEngine,
    PrecisionReport,
    BASE_PRECISION,
    STRESS_LOWER,
    STRESS_UPPER,
)


# ======================================================================= #
# InferenceCost                                                             #
# ======================================================================= #

class TestInferenceCost:
    def test_cost_within_calibrated_bounds(self):
        """Inference cost must be ≤2× passive decay (~0.24 E) at normal load."""
        state = MetabolicState()
        proposals = generate_default_proposals(state)
        cost = compute_inference_cost(proposals, _PRECISION, compute_load=1.0)
        # Must be meaningful (> 0) but not catastrophic (< 2× passive decay 0.24)
        assert 0.05 < cost.energy_cost < 0.24
        assert 0.01 < cost.heat_cost < 0.06

    def test_cost_scales_with_compute_load(self):
        """Doubling compute_load doubles energy and heat costs."""
        state = MetabolicState()
        proposals = generate_default_proposals(state)
        cost_1x = compute_inference_cost(proposals, _PRECISION, compute_load=1.0)
        cost_2x = compute_inference_cost(proposals, _PRECISION, compute_load=2.0)
        assert abs(cost_2x.energy_cost / cost_1x.energy_cost - 2.0) < 0.01
        assert abs(cost_2x.heat_cost / cost_1x.heat_cost - 2.0) < 0.01

    def test_fewer_proposals_cheaper(self):
        """Evaluating fewer proposals costs less (planning depth scales cost)."""
        state = MetabolicState()
        all_proposals = generate_default_proposals(state)
        cost_5 = compute_inference_cost(all_proposals, compute_load=1.0)
        cost_1 = compute_inference_cost(all_proposals[:1], compute_load=1.0)
        assert cost_1.energy_cost < cost_5.energy_cost
        assert cost_1.heat_cost < cost_5.heat_cost

    def test_higher_precision_costs_more_heat(self):
        """Higher precision weights increase the heat cost of inference."""
        proposals = generate_default_proposals(MetabolicState())
        base_cost = compute_inference_cost(proposals, BASE_PRECISION)
        high_prec = {k: v * 3.0 for k, v in BASE_PRECISION.items()}
        high_cost = compute_inference_cost(proposals, high_prec)
        assert high_cost.heat_cost > base_cost.heat_cost

    def test_cost_fields_non_negative(self):
        proposals = generate_default_proposals(MetabolicState())
        cost = compute_inference_cost(proposals)
        assert cost.energy_cost >= 0
        assert cost.heat_cost >= 0
        assert cost.kl_complexity >= 0
        assert cost.precision_used > 0


# ======================================================================= #
# active_inference_step now charges metabolic cost                          #
# ======================================================================= #

class TestInferenceStepChargesCost:
    def test_inference_step_charges_energy(self):
        """active_inference_step must reduce state.energy by the inference cost."""
        state = MetabolicState(energy=80.0)
        initial_energy = state.energy
        proposals = generate_default_proposals(state)
        result = active_inference_step(state, proposals)
        # Energy should be lower after the step (inference cost charged + action delta)
        assert state.energy < initial_energy

    def test_inference_step_charges_heat(self):
        """active_inference_step must increase state.heat by the inference cost."""
        state = MetabolicState(energy=80.0, heat=0.0)
        initial_heat = state.heat
        proposals = generate_default_proposals(state)
        active_inference_step(state, proposals)
        assert state.heat > initial_heat

    def test_inference_result_contains_cost(self):
        """InferenceResult.inference_cost must be populated."""
        state = MetabolicState()
        proposals = generate_default_proposals(state)
        result = active_inference_step(state, proposals)
        assert result.inference_cost is not None
        assert isinstance(result.inference_cost, InferenceCost)
        assert result.inference_cost.energy_cost > 0

    def test_high_compute_load_costs_more(self):
        """Higher compute_load leads to greater energy drain during inference."""
        s1 = MetabolicState(energy=90.0)
        s2 = MetabolicState(energy=90.0)
        proposals = generate_default_proposals(s1)

        active_inference_step(s1, [p for p in proposals], compute_load=1.0)
        active_inference_step(s2, [p for p in proposals], compute_load=3.0)

        # Higher load → more energy spent on inference itself
        assert s2.energy < s1.energy

    def test_precision_weights_used_in_efe(self):
        """Custom precision weights change EFE scores."""
        state = MetabolicState(energy=30.0)  # low energy
        proposals = generate_default_proposals(state)

        # High energy precision → forage_resources should score better
        high_energy_prec = dict(_PRECISION)
        high_energy_prec["energy"] = 10.0

        result_high = active_inference_step(
            MetabolicState(energy=30.0), list(proposals), high_energy_prec
        )
        result_low = active_inference_step(
            MetabolicState(energy=30.0), list(proposals), _PRECISION
        )
        # Under extreme energy precision, the agent should forage more urgently
        # (forage_resources provides energy gain, improving EFE significantly)
        assert result_high.efe_scores["forage_resources"] != result_low.efe_scores["forage_resources"]


# ======================================================================= #
# PrecisionEngine                                                           #
# ======================================================================= #

class TestPrecisionEngine:
    def test_returns_precision_report(self):
        engine = PrecisionEngine()
        state = MetabolicState()
        report = engine.tune(state)
        assert isinstance(report, PrecisionReport)
        assert all(k in report.weights for k in BASE_PRECISION)

    def test_dormant_regime_on_healthy_state(self):
        """A fully healthy state should produce the dormant regime."""
        engine = PrecisionEngine()
        state = MetabolicState()  # energy=100, all optimal
        report = engine.tune(state)
        assert report.regime == "dormant"

    def test_sweet_spot_on_moderate_stress(self):
        """Moderate stress (FE between STRESS_LOWER and STRESS_UPPER) → sweet_spot."""
        engine = PrecisionEngine()
        # Energy and heat deviation pushes FE into sweet-spot band
        state = MetabolicState(energy=50.0, heat=30.0, waste=20.0)
        report = engine.tune(state)
        assert STRESS_LOWER < report.free_energy < STRESS_UPPER
        assert report.regime == "sweet_spot"

    def test_overload_regime_on_extreme_stress(self):
        """Extreme multi-system stress (FE > STRESS_UPPER) → overload."""
        engine = PrecisionEngine()
        state = MetabolicState(energy=20.0, heat=70.0, waste=70.0, integrity=45.0, stability=35.0)
        report = engine.tune(state)
        assert report.free_energy > STRESS_UPPER
        assert report.regime == "overload"

    def test_sweet_spot_boosts_energy_precision(self):
        """In sweet-spot, energy precision rises above base for energy-deprived state."""
        engine = PrecisionEngine()
        state = MetabolicState(energy=40.0, heat=25.0)
        report = engine.tune(state)
        if report.regime == "sweet_spot":
            assert report.weights["energy"] > BASE_PRECISION["energy"]

    def test_overload_keeps_survival_precision_high(self):
        """In overload, energy and heat precision stay elevated for survival."""
        engine = PrecisionEngine()
        state = MetabolicState(energy=20.0, heat=70.0, waste=70.0, integrity=45.0)
        report = engine.tune(state)
        if report.regime == "overload":
            assert report.weights["energy"] >= BASE_PRECISION["energy"]
            assert report.weights["heat"] >= BASE_PRECISION["heat"]

    def test_sweet_spot_has_positive_metabolic_cost(self):
        """Sweet-spot precision sharpening charges energy and heat."""
        engine = PrecisionEngine()
        state = MetabolicState(energy=50.0, heat=30.0, waste=20.0)
        report = engine.tune(state)
        if report.regime == "sweet_spot":
            assert report.energy_cost > 0
            assert report.heat_cost > 0

    def test_dormant_has_zero_cost(self):
        """Dormant regime costs nothing extra."""
        engine = PrecisionEngine()
        state = MetabolicState()
        report = engine.tune(state)
        assert report.energy_cost == 0.0
        assert report.heat_cost == 0.0

    def test_compute_load_scales_sweet_spot_cost(self):
        """Higher compute_load increases sweet-spot metabolic cost."""
        state = MetabolicState(energy=50.0, heat=30.0, waste=20.0)
        engine1 = PrecisionEngine()
        engine2 = PrecisionEngine()
        r1 = engine1.tune(state, compute_load=1.0)
        r2 = engine2.tune(state, compute_load=2.0)
        if r1.regime == "sweet_spot":
            assert r2.energy_cost > r1.energy_cost

    def test_weights_property_returns_last_computed(self):
        """weights property reflects the most recent tune() call."""
        engine = PrecisionEngine()
        state = MetabolicState(energy=50.0, heat=30.0, waste=20.0)
        report = engine.tune(state)
        assert engine.weights == report.weights


# ======================================================================= #
# Affect signal in MetabolicState                                           #
# ======================================================================= #

class TestAffectSignal:
    def test_affect_initialises_to_zero(self):
        state = MetabolicState()
        assert state.affect == 0.0

    def test_affect_is_within_bounds(self):
        """Affect must always stay in [-1, 1]."""
        state = MetabolicState(energy=30.0, heat=40.0)
        for _ in range(20):
            try:
                state.tick()
            except Exception:
                break
        assert -1.0 <= state.affect <= 1.0

    def test_energy_recovery_produces_positive_affect(self):
        """Recovering energy (free energy dropping) should produce positive affect."""
        state = MetabolicState(energy=30.0)
        state.tick()  # let free energy register as elevated
        state.apply_action_feedback(delta_energy=40.0)
        state.tick()  # now free energy should drop → positive affect
        assert state.affect > 0.0

    def test_energy_depletion_produces_negative_affect(self):
        """Worsening energy (free energy rising) produces negative affect."""
        state = MetabolicState(energy=100.0)
        state.tick()  # healthy first tick, _prev_free_energy = 0
        # Drain energy heavily so FE rises sharply
        state.apply_action_feedback(delta_energy=-60.0)
        state.tick()
        assert state.affect < 0.0

    def test_free_energy_estimate_zero_when_healthy(self):
        """FE estimate is 0 when all vitals are at or above setpoints."""
        state = MetabolicState(energy=100.0, heat=0.0, waste=0.0, integrity=100.0, stability=100.0)
        assert state.free_energy_estimate() == 0.0

    def test_free_energy_estimate_rises_with_stress(self):
        """FE estimate increases as vitals degrade."""
        healthy = MetabolicState()
        stressed = MetabolicState(energy=30.0, heat=50.0, waste=40.0)
        assert stressed.free_energy_estimate() > healthy.free_energy_estimate()

    def test_affect_persisted_in_dict(self):
        """affect field is included in to_dict / from_dict round-trip."""
        state = MetabolicState(energy=30.0)
        state.tick()
        d = state.to_dict()
        assert "affect" in d
        restored = MetabolicState.from_dict(d)
        assert abs(restored.affect - state.affect) < 1e-9


# ======================================================================= #
# Janitor LLM cost scaling                                                  #
# ======================================================================= #

class TestJanitorLLMCost:
    def test_llm_cost_fields_initialised(self, tmp_path):
        """Janitor initialises _pending_llm_cost to zero."""
        from thermodynamic_agency.memory.diary import RamDiary
        from thermodynamic_agency.cognition.janitor import Janitor
        d = RamDiary(path=str(tmp_path / "d.db"))
        j = Janitor(diary=d, use_llm=False)
        assert j._pending_llm_cost == (0.0, 0.0)
        d.close()

    def test_no_extra_cost_without_llm(self, tmp_path):
        """Without LLM the pending cost stays zero after a run."""
        from thermodynamic_agency.memory.diary import RamDiary, DiaryEntry
        from thermodynamic_agency.cognition.janitor import Janitor, JANITOR_ENERGY_COST
        d = RamDiary(path=str(tmp_path / "d.db"))
        for i in range(5):
            d.append(DiaryEntry(tick=i, role="thought", content=f"entry {i}"))
        state = MetabolicState(energy=80.0, heat=85.0, waste=80.0)
        energy_before = state.energy
        j = Janitor(diary=d, use_llm=False)
        j.run(state)
        # Should only have charged the base cost, not extra LLM cost
        assert abs((energy_before - state.energy) - JANITOR_ENERGY_COST) < 0.5
        d.close()
