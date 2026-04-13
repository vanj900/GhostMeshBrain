"""Tests for active inference — compute_efe and active_inference_step."""

import pytest

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.inference import (
    ActionProposal,
    active_inference_step,
    compute_efe,
    generate_default_proposals,
    InferenceResult,
)


class TestComputeEFE:
    def test_zero_delta_is_finite(self):
        state = MetabolicState()
        efe = compute_efe(state, {})
        assert isinstance(efe, float)
        assert efe >= 0.0

    def test_energy_increase_reduces_efe_for_starving(self):
        """A proposal that increases energy should have lower EFE when energy is low."""
        state = MetabolicState(energy=10.0)
        efe_gain = compute_efe(state, {"energy": 30.0})
        efe_loss = compute_efe(state, {"energy": -30.0})
        assert efe_gain < efe_loss

    def test_near_death_amplifies_efe(self):
        """EFE should be very high when proposed action approaches death threshold."""
        state = MetabolicState(energy=10.0)
        efe_near_death = compute_efe(state, {"energy": -9.0})
        efe_safe = compute_efe(state, {"energy": 5.0})
        assert efe_near_death > efe_safe

    def test_heat_increase_raises_efe(self):
        state = MetabolicState(heat=50.0)
        efe_hot = compute_efe(state, {"heat": 30.0})
        efe_cool = compute_efe(state, {"heat": -10.0})
        assert efe_hot > efe_cool


class TestActiveInferenceStep:
    def test_returns_inference_result(self):
        state = MetabolicState()
        proposals = generate_default_proposals(state)
        result = active_inference_step(state, proposals)
        assert isinstance(result, InferenceResult)
        assert result.selected in proposals

    def test_selects_lowest_efe(self):
        state = MetabolicState(energy=80.0, heat=20.0, waste=10.0)
        proposals = generate_default_proposals(state)
        result = active_inference_step(state, proposals)
        best_efe = result.efe_scores[result.selected.name]
        for score in result.efe_scores.values():
            assert best_efe <= score

    def test_raises_on_empty_proposals(self):
        state = MetabolicState()
        with pytest.raises(ValueError):
            active_inference_step(state, [])

    def test_efe_scores_all_proposals(self):
        state = MetabolicState()
        proposals = generate_default_proposals(state)
        result = active_inference_step(state, proposals)
        assert len(result.efe_scores) == len(proposals)

    def test_reasoning_contains_selection(self):
        state = MetabolicState()
        proposals = generate_default_proposals(state)
        result = active_inference_step(state, proposals)
        assert result.selected.name in result.reasoning


class TestDefaultProposals:
    def test_generates_multiple_proposals(self):
        state = MetabolicState()
        proposals = generate_default_proposals(state)
        assert len(proposals) >= 3

    def test_all_proposals_have_names(self):
        state = MetabolicState()
        for p in generate_default_proposals(state):
            assert p.name
            assert p.description
