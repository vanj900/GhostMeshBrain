"""Tests for the limbic layer (Layer 2 — LimbicLayer, AmygdalaModule,
NucleusAccumbens, EpisodicBuffer)."""

import pytest

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.limbic import (
    AmygdalaModule,
    AmygdalaSignal,
    NucleusAccumbens,
    EpisodicBuffer,
    EpisodicSlot,
    LimbicLayer,
    LimbicSignal,
    AMYGDALA_THREAT_FE,
    ACCUMBENS_AFFECT_THRESHOLD,
    ACCUMBENS_MAX_DISCOUNT,
    DEFAULT_EPISODIC_CAPACITY,
    CONSOLIDATION_COST_PER_SLOT,
)


# ------------------------------------------------------------------ #
# AmygdalaModule                                                       #
# ------------------------------------------------------------------ #

class TestAmygdalaModule:
    def test_no_threat_below_threshold(self):
        """Below the threat threshold the amygdala is silent."""
        state = MetabolicState(energy=90.0, heat=10.0, waste=5.0, integrity=95.0, stability=95.0)
        amyg = AmygdalaModule()
        signal = amyg.evaluate(state)
        assert signal.threat_level == 0.0
        assert signal.precision_overrides == {}
        assert signal.affect_modifier == 0.0
        assert signal.heat_cost == 0.0

    def test_threat_fires_on_high_fe(self):
        """High free energy (poor vitals) should trigger a non-zero threat."""
        # Force very poor vitals — low energy, high heat, low integrity
        state = MetabolicState(energy=5.0, heat=70.0, waste=50.0, integrity=15.0, stability=20.0)
        fe = state.free_energy_estimate()
        assert fe > AMYGDALA_THREAT_FE, f"Expected FE > {AMYGDALA_THREAT_FE}, got {fe}"
        amyg = AmygdalaModule()
        signal = amyg.evaluate(state)
        assert signal.threat_level > 0.0
        assert signal.threat_level <= 1.0

    def test_threat_precision_overrides_populated(self):
        """When threat fires, precision_overrides must include survival vitals."""
        state = MetabolicState(energy=5.0, heat=70.0, waste=50.0, integrity=15.0, stability=20.0)
        amyg = AmygdalaModule()
        signal = amyg.evaluate(state)
        if signal.threat_level > 0:
            assert "energy" in signal.precision_overrides
            assert "heat" in signal.precision_overrides
            assert "integrity" in signal.precision_overrides

    def test_threat_affect_modifier_is_negative(self):
        """Amygdala should push affect more negative under threat."""
        state = MetabolicState(energy=5.0, heat=70.0, waste=50.0, integrity=15.0, stability=20.0)
        amyg = AmygdalaModule()
        signal = amyg.evaluate(state)
        if signal.threat_level > 0:
            assert signal.affect_modifier < 0.0

    def test_heat_cost_proportional_to_threat(self):
        """Amygdala activation costs heat proportional to threat level."""
        state = MetabolicState(energy=5.0, heat=70.0, waste=50.0, integrity=15.0, stability=20.0)
        amyg = AmygdalaModule()
        signal = amyg.evaluate(state)
        if signal.threat_level > 0:
            assert signal.heat_cost > 0.0
            # Heat cost should be proportional: threat * HEAT_COST_SCALE
            assert signal.heat_cost == pytest.approx(signal.threat_level * 0.15, rel=0.01)

    def test_threat_level_clamped_to_one(self):
        """Threat level must never exceed 1.0 regardless of FE."""
        state = MetabolicState(energy=0.1, heat=99.0, waste=99.0, integrity=11.0, stability=0.1)
        amyg = AmygdalaModule()
        signal = amyg.evaluate(state)
        assert signal.threat_level <= 1.0


# ------------------------------------------------------------------ #
# NucleusAccumbens                                                     #
# ------------------------------------------------------------------ #

class TestNucleusAccumbens:
    def test_no_discount_below_threshold(self):
        """Below affect threshold the accumbens returns zero discount."""
        state = MetabolicState()
        state.affect = ACCUMBENS_AFFECT_THRESHOLD - 0.1
        acc = NucleusAccumbens()
        assert acc.efe_discount(state) == 0.0

    def test_discount_grows_with_positive_affect(self):
        """Higher positive affect should yield higher EFE discount."""
        acc = NucleusAccumbens()

        state_low = MetabolicState()
        state_low.affect = ACCUMBENS_AFFECT_THRESHOLD + 0.1
        discount_low = acc.efe_discount(state_low)

        state_high = MetabolicState()
        state_high.affect = 0.9
        discount_high = acc.efe_discount(state_high)

        assert discount_high > discount_low

    def test_discount_capped_at_max(self):
        """Discount must never exceed ACCUMBENS_MAX_DISCOUNT."""
        acc = NucleusAccumbens()
        state = MetabolicState()
        state.affect = 1.0
        assert acc.efe_discount(state) <= ACCUMBENS_MAX_DISCOUNT

    def test_no_discount_on_negative_affect(self):
        """Negative affect should produce no discount."""
        acc = NucleusAccumbens()
        state = MetabolicState()
        state.affect = -0.8
        assert acc.efe_discount(state) == 0.0


# ------------------------------------------------------------------ #
# EpisodicBuffer                                                       #
# ------------------------------------------------------------------ #

class TestEpisodicBuffer:
    def test_empty_buffer_no_cost(self):
        buf = EpisodicBuffer(capacity=5)
        assert buf.consolidation_cost() == 0.0
        assert buf.active_count() == 0

    def test_within_capacity_no_cost(self):
        buf = EpisodicBuffer(capacity=5)
        for i in range(5):
            buf.push(EpisodicSlot(tick=i, content=f"slot {i}"))
        assert buf.consolidation_cost() == 0.0

    def test_overflow_generates_cost(self):
        capacity = 3
        buf = EpisodicBuffer(capacity=capacity)
        for i in range(capacity + 2):
            buf.push(EpisodicSlot(tick=i, content=f"slot {i}"))
        cost = buf.consolidation_cost()
        assert cost > 0.0
        expected = 2 * CONSOLIDATION_COST_PER_SLOT
        assert cost == pytest.approx(expected, rel=0.01)

    def test_flush_reduces_active_count(self):
        buf = EpisodicBuffer(capacity=3)
        for i in range(6):
            buf.push(EpisodicSlot(tick=i, content=f"slot {i}"))
        before = buf.active_count()
        flushed = buf.flush_oldest(n=3)
        after = buf.active_count()
        assert len(flushed) == 3
        assert after == before - 3

    def test_flush_returns_consolidated_slots(self):
        buf = EpisodicBuffer(capacity=5)
        for i in range(3):
            buf.push(EpisodicSlot(tick=i, content=f"slot {i}"))
        flushed = buf.flush_oldest(n=2)
        assert all(s.consolidated for s in flushed)

    def test_flush_more_than_available(self):
        """Flush should not fail if n > unconsolidated count."""
        buf = EpisodicBuffer(capacity=5)
        buf.push(EpisodicSlot(tick=0, content="only one"))
        flushed = buf.flush_oldest(n=10)
        assert len(flushed) == 1

    def test_total_count(self):
        buf = EpisodicBuffer(capacity=5)
        for i in range(4):
            buf.push(EpisodicSlot(tick=i, content=f"slot {i}"))
        assert buf.total_count() == 4


# ------------------------------------------------------------------ #
# LimbicLayer                                                          #
# ------------------------------------------------------------------ #

class TestLimbicLayer:
    def test_process_returns_limbic_signal(self):
        state = MetabolicState()
        layer = LimbicLayer()
        signal = layer.process(state)
        assert isinstance(signal, LimbicSignal)
        assert 0.0 <= signal.threat_level <= 1.0
        assert 0.0 <= signal.efe_discount <= ACCUMBENS_MAX_DISCOUNT
        assert signal.integrity_cost >= 0.0
        assert signal.heat_cost >= 0.0

    def test_push_episode_increments_buffer(self):
        layer = LimbicLayer(episodic_capacity=10)
        assert layer.episodic.active_count() == 0
        layer.push_episode(tick=1, content="test episode")
        assert layer.episodic.active_count() == 1

    def test_consolidate_drains_buffer(self):
        layer = LimbicLayer(episodic_capacity=5)
        for i in range(8):
            layer.push_episode(tick=i, content=f"ep {i}")
        before = layer.episodic.active_count()
        flushed = layer.consolidate(n=5)
        after = layer.episodic.active_count()
        assert len(flushed) == 5
        assert after == before - 5

    def test_high_fe_state_produces_threat(self):
        """Organism in severe distress should produce non-zero threat level."""
        state = MetabolicState(energy=5.0, heat=70.0, waste=60.0, integrity=15.0, stability=15.0)
        layer = LimbicLayer()
        signal = layer.process(state)
        assert signal.threat_level > 0.0
        assert signal.heat_cost > 0.0
        assert len(signal.precision_overrides) > 0

    def test_healthy_state_produces_no_threat(self):
        """Healthy organism should produce zero threat level."""
        state = MetabolicState(energy=90.0, heat=10.0, waste=5.0, integrity=95.0, stability=95.0)
        layer = LimbicLayer()
        signal = layer.process(state)
        assert signal.threat_level == 0.0
        assert signal.heat_cost == 0.0

    def test_positive_affect_yields_efe_discount(self):
        """Positive affect should activate the accumbens reward discount."""
        state = MetabolicState(energy=85.0, heat=10.0, waste=5.0, integrity=90.0, stability=90.0)
        state.affect = 0.8
        layer = LimbicLayer()
        signal = layer.process(state)
        assert signal.efe_discount > 0.0

    def test_status_returns_dict(self):
        layer = LimbicLayer()
        status = layer.status()
        assert "episodic_active" in status
        assert "episodic_total" in status
        assert "episodic_cost" in status

    def test_overflow_cost_reflected_in_signal(self):
        """When episodic buffer overflows, integrity_cost > 0."""
        layer = LimbicLayer(episodic_capacity=2)
        for i in range(5):
            layer.push_episode(tick=i, content=f"ep {i}")
        state = MetabolicState()
        signal = layer.process(state)
        assert signal.integrity_cost > 0.0
