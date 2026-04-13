"""Tests for MetabolicState and death exceptions."""

import pytest

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.core.exceptions import (
    EnergyDeathException,
    ThermalDeathException,
    MemoryCollapseException,
    EntropyDeathException,
    GhostDeathException,
)


class TestMetabolicState:
    def test_initial_values(self):
        state = MetabolicState()
        assert state.energy == 100.0
        assert state.heat == 0.0
        assert state.waste == 0.0
        assert state.integrity == 100.0
        assert state.stability == 100.0
        assert state.entropy == 0
        assert state.stage == "dormant"

    def test_tick_increments_entropy(self):
        state = MetabolicState()
        state.tick()
        assert state.entropy == 1
        state.tick()
        assert state.entropy == 2

    def test_tick_decays_energy(self):
        state = MetabolicState()
        initial_energy = state.energy
        state.tick()
        assert state.energy < initial_energy

    def test_tick_increases_heat(self):
        state = MetabolicState()
        state.tick()
        assert state.heat > 0.0

    def test_tick_returns_decide_when_healthy(self):
        state = MetabolicState(energy=90.0, heat=10.0, waste=5.0, integrity=90.0, stability=90.0)
        action = state.tick()
        assert action == "DECIDE"

    def test_tick_returns_forage_on_low_energy(self):
        state = MetabolicState(energy=20.0, heat=10.0, waste=5.0, integrity=90.0, stability=90.0)
        action = state.tick()
        assert action == "FORAGE"

    def test_tick_returns_rest_on_high_waste(self):
        state = MetabolicState(energy=80.0, heat=10.0, waste=80.0, integrity=90.0, stability=90.0)
        action = state.tick()
        assert action == "REST"

    def test_tick_returns_rest_on_high_heat(self):
        state = MetabolicState(energy=80.0, heat=85.0, waste=5.0, integrity=90.0, stability=90.0)
        action = state.tick()
        assert action == "REST"

    def test_tick_returns_repair_on_low_integrity(self):
        state = MetabolicState(energy=80.0, heat=10.0, waste=5.0, integrity=40.0, stability=90.0)
        action = state.tick()
        assert action == "REPAIR"

    def test_tick_returns_repair_on_low_stability(self):
        state = MetabolicState(energy=80.0, heat=10.0, waste=5.0, integrity=90.0, stability=35.0)
        action = state.tick()
        assert action == "REPAIR"

    def test_tick_raises_energy_death(self):
        state = MetabolicState(energy=0.05)
        with pytest.raises(EnergyDeathException) as exc_info:
            state.tick()
        assert exc_info.value.state  # snapshot attached

    def test_tick_raises_thermal_death(self):
        state = MetabolicState(heat=99.9)
        with pytest.raises(ThermalDeathException):
            state.tick()

    def test_tick_raises_memory_collapse(self):
        # Start integrity below the death threshold of 10.0
        state = MetabolicState(energy=80.0, integrity=9.9, heat=5.0, stability=90.0)
        with pytest.raises(MemoryCollapseException):
            state.tick()

    def test_tick_raises_entropy_death(self):
        state = MetabolicState(stability=0.04)
        with pytest.raises(EntropyDeathException):
            state.tick()

    def test_apply_action_feedback_clamps(self):
        state = MetabolicState(energy=50.0)
        state.apply_action_feedback(delta_energy=200.0)
        assert state.energy == 100.0  # clamped at max

        state.apply_action_feedback(delta_energy=-300.0)
        assert state.energy == 0.0  # clamped at min

    def test_health_score_range(self):
        state = MetabolicState()
        score = state.health_score()
        assert 0.0 <= score <= 100.0

    def test_to_dict_and_from_dict_roundtrip(self):
        state = MetabolicState(energy=75.0, heat=25.0, entropy=42, stage="emerging")
        data = state.to_dict()
        restored = MetabolicState.from_dict(data)
        assert restored.energy == state.energy
        assert restored.heat == state.heat
        assert restored.entropy == state.entropy
        assert restored.stage == state.stage

    def test_stage_evolves_to_emerging(self):
        state = MetabolicState()
        # Force past threshold
        state.entropy = 99
        state.tick()  # entropy becomes 100
        assert state.stage == "emerging"

    def test_death_exceptions_are_ghost_death(self):
        for exc_class in (
            EnergyDeathException,
            ThermalDeathException,
            MemoryCollapseException,
            EntropyDeathException,
        ):
            exc = exc_class("test", {"key": "val"})
            assert isinstance(exc, GhostDeathException)
            assert exc.state == {"key": "val"}

    def test_to_json_is_valid(self):
        import json
        state = MetabolicState()
        json_str = state.to_json()
        data = json.loads(json_str)
        assert "energy" in data
        assert "heat" in data
