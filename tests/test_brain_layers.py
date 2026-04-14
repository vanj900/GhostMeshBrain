"""Tests for Layer 3 (ForwardModel) and Layer 4 (hierarchical masks)."""

import pytest

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.inference import (
    ForwardModel,
    ForwardPrediction,
    active_inference_step,
    generate_default_proposals,
)
from thermodynamic_agency.cognition.personality import (
    ALL_MASKS_EXTENDED,
    MaskRotator,
    _MASK_MAP_EXTENDED,
)


# ------------------------------------------------------------------ #
# ForwardModel (Layer 3 — Cerebellum)                                 #
# ------------------------------------------------------------------ #

class TestForwardModel:
    def test_initial_error_is_zero(self):
        fm = ForwardModel()
        assert fm.prediction_error_term() == 0.0

    def test_predict_with_no_history_returns_current_state(self):
        state = MetabolicState(energy=80.0, heat=20.0, waste=10.0, integrity=85.0, stability=80.0)
        fm = ForwardModel()
        pred = fm.predict(state)
        assert isinstance(pred, ForwardPrediction)
        assert len(pred.steps) == 3  # default n_steps
        # Without history, prediction mirrors current
        for step in pred.steps:
            assert step["energy"] == pytest.approx(80.0, rel=0.01)

    def test_update_accumulates_history(self):
        fm = ForwardModel(history_size=5)
        state = MetabolicState()
        for _ in range(4):
            state.tick()
            fm.update(state)
        # After 4 updates there should be history
        assert len(fm._history) == 4

    def test_predict_returns_n_steps(self):
        fm = ForwardModel(n_steps=5)
        state = MetabolicState()
        pred = fm.predict(state)
        assert len(pred.steps) == 5

    def test_prediction_error_non_zero_after_missed_forecast(self):
        """After calling predict() then update() with different state, error > 0."""
        fm = ForwardModel()
        state = MetabolicState(energy=80.0, heat=20.0, waste=10.0, integrity=85.0, stability=80.0)

        # Build some history
        for _ in range(5):
            state.tick()
            fm.update(state)

        # Make a prediction
        fm.predict(state)

        # Dramatically change state so prediction was wrong
        state.energy = 10.0
        state.heat = 80.0
        fm.update(state)

        # Should now have non-zero prediction error
        assert fm.prediction_error_term() > 0.0

    def test_prediction_error_term_non_negative(self):
        fm = ForwardModel()
        state = MetabolicState()
        for _ in range(10):
            state.tick()
            fm.update(state)
        assert fm.prediction_error_term() >= 0.0

    def test_smoothed_delta_with_minimal_history(self):
        """ForwardModel doesn't crash with only 1 history record."""
        fm = ForwardModel()
        state = MetabolicState()
        fm.update(state)
        pred = fm.predict(state)
        assert len(pred.steps) > 0

    def test_reward_discount_applied_in_active_inference(self):
        """reward_discount parameter should lower EFE scores in active_inference_step."""
        state = MetabolicState(energy=80.0, heat=10.0, waste=5.0, integrity=90.0, stability=90.0)
        proposals = generate_default_proposals(state)

        # Run without discount
        import copy
        state_a = copy.deepcopy(state)
        result_a = active_inference_step(state_a, proposals, reward_discount=0.0)

        state_b = copy.deepcopy(state)
        result_b = active_inference_step(state_b, proposals, reward_discount=0.15)

        # With discount, EFE scores should be lower for all proposals
        for name in result_a.efe_scores:
            if name in result_b.efe_scores:
                assert result_b.efe_scores[name] <= result_a.efe_scores[name] + 1e-9


# ------------------------------------------------------------------ #
# Hierarchical masks (Layer 4 — Prefrontal)                            #
# ------------------------------------------------------------------ #

class TestHierarchicalMasks:
    def test_all_masks_extended_includes_prefrontal(self):
        names = {m.name for m in ALL_MASKS_EXTENDED}
        assert "DefaultMode" in names
        assert "SalienceNet" in names
        assert "CentralExec" in names

    def test_all_masks_extended_still_has_originals(self):
        names = {m.name for m in ALL_MASKS_EXTENDED}
        for expected in ("Healer", "Judge", "Courier", "Dreamer", "Guardian"):
            assert expected in names

    def test_default_mode_efe_overrides(self):
        dm = _MASK_MAP_EXTENDED["DefaultMode"]
        assert "integrity" in dm.efe_precision_overrides
        assert "stability" in dm.efe_precision_overrides

    def test_salience_net_efe_overrides(self):
        sn = _MASK_MAP_EXTENDED["SalienceNet"]
        assert "energy" in sn.efe_precision_overrides
        assert "heat" in sn.efe_precision_overrides

    def test_central_exec_efe_overrides(self):
        ce = _MASK_MAP_EXTENDED["CentralExec"]
        assert "integrity" in ce.efe_precision_overrides

    def test_rotator_can_force_prefrontal_masks(self):
        rotator = MaskRotator(initial_mask="Guardian")
        for _ in range(10):
            rotator.tick(1)
        rotator.maybe_rotate(10, force="DefaultMode")
        assert rotator.active.name == "DefaultMode"

    def test_high_threat_rotates_to_salience_net(self):
        """Amygdala threat >= 0.6 should trigger SalienceNet rotation."""
        rotator = MaskRotator(initial_mask="Dreamer")
        # Tick past min_ticks
        for _ in range(10):
            rotator.tick(1)
        rotator.maybe_rotate(10, metabolic_hint="DECIDE", affect=0.0, threat_level=0.7)
        assert rotator.active.name == "SalienceNet"

    def test_positive_affect_rest_rotates_to_default_mode(self):
        """Strong positive affect + REST should prefer DefaultMode."""
        rotator = MaskRotator(initial_mask="Guardian")
        for _ in range(10):
            rotator.tick(1)
        rotator.maybe_rotate(10, metabolic_hint="REST", affect=0.6, threat_level=0.0)
        assert rotator.active.name == "DefaultMode"

    def test_healthy_decide_rotates_to_central_exec(self):
        """Healthy DECIDE with neutral affect should prefer CentralExec."""
        rotator = MaskRotator(initial_mask="Guardian")
        for _ in range(10):
            rotator.tick(1)
        rotator.maybe_rotate(10, metabolic_hint="DECIDE", affect=0.0, threat_level=0.0)
        assert rotator.active.name == "CentralExec"

    def test_maybe_rotate_old_signature_still_works(self):
        """maybe_rotate without affect/threat_level kwargs must not crash."""
        rotator = MaskRotator(initial_mask="Guardian")
        for _ in range(10):
            rotator.tick(1)
        # Original positional usage (no keyword args added)
        result = rotator.maybe_rotate(10, metabolic_hint="REST")
        assert result is not None
