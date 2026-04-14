"""Tests for stochastic environment events."""

from __future__ import annotations

import random

import pytest

from thermodynamic_agency.core.environment import (
    EnvironmentalEvent,
    sample_event,
    _NAMES,
    _WEIGHTS,
)

# All MetabolicState vital delta fields on EnvironmentalEvent
_VITAL_FIELDS = (
    "delta_energy", "delta_heat", "delta_waste", "delta_integrity", "delta_stability"
)


class TestEnvironmentalEvent:
    def test_is_null_when_all_zero(self):
        e = EnvironmentalEvent(name="null")
        assert e.is_null()

    def test_not_null_with_any_nonzero(self):
        e = EnvironmentalEvent(name="test", delta_energy=1.0)
        assert not e.is_null()

    def test_all_fields_default_zero(self):
        e = EnvironmentalEvent(name="x")
        for field in _VITAL_FIELDS:
            assert getattr(e, field) == 0.0


class TestSampleEvent:
    def test_returns_environmental_event(self):
        e = sample_event()
        assert isinstance(e, EnvironmentalEvent)

    def test_name_is_known_event(self):
        rng = random.Random(0)
        for _ in range(50):
            e = sample_event(rng=rng)
            assert e.name in _NAMES

    def test_calm_produces_positive_energy(self):
        """calm events should give energy (positive delta_energy)."""
        rng = random.Random(0)
        calm_events = []
        for _ in range(1000):
            e = sample_event(rng=rng)
            if e.name == "calm":
                calm_events.append(e)
        assert len(calm_events) > 0
        assert all(ev.delta_energy > 0 for ev in calm_events)

    def test_energy_drain_produces_negative_energy(self):
        rng = random.Random(0)
        drain_events = []
        for _ in range(1000):
            e = sample_event(rng=rng)
            if e.name == "energy_drain":
                drain_events.append(e)
        assert len(drain_events) > 0
        assert all(ev.delta_energy < 0 for ev in drain_events)

    def test_crisis_is_multidimensional(self):
        """crisis events should affect multiple vitals."""
        rng = random.Random(0)
        crisis_events = []
        for _ in range(5000):
            e = sample_event(rng=rng)
            if e.name == "crisis":
                crisis_events.append(e)
        assert len(crisis_events) > 0
        for ev in crisis_events[:5]:
            nonzero = sum(
                1 for f in _VITAL_FIELDS
                if getattr(ev, f) != 0.0
            )
            assert nonzero >= 2, "crisis should affect at least 2 vitals"

    def test_weights_sum_to_one(self):
        total = sum(_WEIGHTS)
        assert abs(total - 1.0) < 1e-6

    def test_seeded_rng_is_reproducible(self):
        rng1 = random.Random(999)
        rng2 = random.Random(999)
        names1 = [sample_event(rng=rng1).name for _ in range(20)]
        names2 = [sample_event(rng=rng2).name for _ in range(20)]
        assert names1 == names2

    def test_event_magnitudes_are_bounded(self):
        """No single event should kill the organism outright."""
        rng = random.Random(42)
        for _ in range(500):
            e = sample_event(rng=rng)
            # Energy drain should not be deeper than -15 in one hit
            assert e.delta_energy >= -15.0
            # Heat spike should not be more than +20 in one hit
            assert e.delta_heat <= 20.0
            # Waste flood should not be more than +25 in one hit
            assert e.delta_waste <= 25.0
