"""Tests for EnvironmentStressor — flat, bursty, and hostile_windows modes.

Coverage
--------
- Constructor validation (prob bounds, intensity bounds)
- Zero-prob stressor never fires
- Flat mode: disturbances fire and mutate metabolic state correctly
- Flat mode: all four disturbance types produce the right vital changes
- Flat mode: seeded RNG produces reproducible event strings
- Bursty mode: fires at elevated rate; may produce chained events
- Hostile-window mode: tracks in_hostile_window flag correctly
- Hostile-window mode: [HOSTILE_WINDOW] prefix on events fired inside window
- Hostile-window mode: window expires after its duration
- Hostile-window mode: elevated intensity during window causes larger hits
- maybe_disturb returns str (empty or non-empty) in all modes
"""

from __future__ import annotations

import pytest

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.environment import (
    EnvironmentStressor,
    _DISTURBANCE_TYPES,
    _HW_ENTRY_PROB,
    _HW_MIN_DURATION,
    _HW_MAX_DURATION,
    _HW_STRESSOR_MULTIPLIER,
    _HW_INTENSITY_MULTIPLIER,
)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _fresh_state() -> MetabolicState:
    """A mid-range metabolic state that won't hit death thresholds during tests."""
    return MetabolicState(
        energy=80.0,
        heat=20.0,
        waste=10.0,
        integrity=80.0,
        stability=80.0,
    )


def _run_ticks(stressor: EnvironmentStressor, n: int = 500) -> list[str]:
    """Run *n* ticks and return all non-empty event strings."""
    events = []
    for _ in range(n):
        state = _fresh_state()   # fresh state each tick avoids death
        result = stressor.maybe_disturb(state)
        if result:
            events.append(result)
    return events


# ------------------------------------------------------------------ #
# Constructor validation                                              #
# ------------------------------------------------------------------ #

class TestConstructorValidation:
    def test_prob_below_zero_raises(self):
        with pytest.raises(ValueError, match="prob must be in"):
            EnvironmentStressor(prob=-0.01)

    def test_prob_above_one_raises(self):
        with pytest.raises(ValueError, match="prob must be in"):
            EnvironmentStressor(prob=1.01)

    def test_negative_intensity_raises(self):
        with pytest.raises(ValueError, match="intensity must be"):
            EnvironmentStressor(prob=0.5, intensity=-0.1)

    def test_valid_extremes_do_not_raise(self):
        EnvironmentStressor(prob=0.0)
        EnvironmentStressor(prob=1.0)
        EnvironmentStressor(prob=0.5, intensity=0.0)
        EnvironmentStressor(prob=0.5, intensity=10.0)

    def test_attributes_stored(self):
        s = EnvironmentStressor(prob=0.3, intensity=1.5, seed=42, mode="bursty")
        assert s.prob == 0.3
        assert s.intensity == 1.5
        assert s.mode == "bursty"

    def test_default_not_in_hostile_window(self):
        s = EnvironmentStressor(prob=0.5)
        assert s.in_hostile_window is False


# ------------------------------------------------------------------ #
# Zero-prob behaviour                                                 #
# ------------------------------------------------------------------ #

class TestZeroProb:
    def test_zero_prob_never_fires(self):
        s = EnvironmentStressor(prob=0.0)
        for _ in range(200):
            result = s.maybe_disturb(_fresh_state())
            assert result == ""

    def test_zero_prob_hostile_windows_never_fires(self):
        s = EnvironmentStressor(prob=0.0, mode="hostile_windows")
        for _ in range(200):
            result = s.maybe_disturb(_fresh_state())
            assert result == ""

    def test_zero_prob_bursty_never_fires(self):
        s = EnvironmentStressor(prob=0.0, mode="bursty")
        for _ in range(200):
            result = s.maybe_disturb(_fresh_state())
            assert result == ""


# ------------------------------------------------------------------ #
# Flat mode                                                           #
# ------------------------------------------------------------------ #

class TestFlatMode:
    def test_returns_string(self):
        s = EnvironmentStressor(prob=1.0, seed=0)
        result = s.maybe_disturb(_fresh_state())
        assert isinstance(result, str)

    def test_prob_1_always_fires(self):
        s = EnvironmentStressor(prob=1.0, seed=7)
        for _ in range(50):
            result = s.maybe_disturb(_fresh_state())
            assert result != ""

    def test_prob_low_fires_sometimes(self):
        s = EnvironmentStressor(prob=0.2, seed=42)
        events = _run_ticks(s, 500)
        # At prob=0.2 over 500 ticks we expect roughly 100 events; any >10 is fine
        assert len(events) > 10

    def test_event_contains_known_type(self):
        s = EnvironmentStressor(prob=1.0, seed=0)
        for _ in range(30):
            result = s.maybe_disturb(_fresh_state())
            assert any(t in result for t in _DISTURBANCE_TYPES)

    def test_energy_drain_reduces_energy(self):
        """Force an energy_drain event and verify energy went down."""
        s = EnvironmentStressor(prob=1.0, seed=0)
        # Iterate until we hit an energy_drain
        for _ in range(200):
            state = _fresh_state()
            before = state.energy
            result = s.maybe_disturb(state)
            if "energy_drain" in result:
                assert state.energy < before
                break
        else:
            pytest.skip("seed did not produce energy_drain in 200 ticks")

    def test_heat_burst_increases_heat(self):
        s = EnvironmentStressor(prob=1.0, seed=0)
        for _ in range(200):
            state = _fresh_state()
            before = state.heat
            result = s.maybe_disturb(state)
            if "heat_burst" in result:
                assert state.heat > before
                break
        else:
            pytest.skip("seed did not produce heat_burst in 200 ticks")

    def test_waste_dump_increases_waste(self):
        s = EnvironmentStressor(prob=1.0, seed=0)
        for _ in range(200):
            state = _fresh_state()
            before = state.waste
            result = s.maybe_disturb(state)
            if "waste_dump" in result:
                assert state.waste > before
                break
        else:
            pytest.skip("seed did not produce waste_dump in 200 ticks")

    def test_stability_quake_reduces_stability(self):
        s = EnvironmentStressor(prob=1.0, seed=0)
        for _ in range(200):
            state = _fresh_state()
            before = state.stability
            result = s.maybe_disturb(state)
            if "stability_quake" in result:
                assert state.stability < before
                break
        else:
            pytest.skip("seed did not produce stability_quake in 200 ticks")

    def test_seeded_rng_reproducible(self):
        s1 = EnvironmentStressor(prob=0.5, seed=123)
        s2 = EnvironmentStressor(prob=0.5, seed=123)
        for _ in range(40):
            state1, state2 = _fresh_state(), _fresh_state()
            r1 = s1.maybe_disturb(state1)
            r2 = s2.maybe_disturb(state2)
            assert r1 == r2

    def test_intensity_scales_magnitude(self):
        """Higher intensity → larger numeric impact on vitals."""
        # Use energy_drain as a measurable proxy
        def total_energy_loss(intensity: float, n: int = 300) -> float:
            s = EnvironmentStressor(prob=1.0, intensity=intensity, seed=5)
            total = 0.0
            for _ in range(n):
                state = _fresh_state()
                s.maybe_disturb(state)
                total += (80.0 - state.energy)
            return total

        low = total_energy_loss(0.5)
        high = total_energy_loss(2.0)
        assert high > low


# ------------------------------------------------------------------ #
# Bursty mode                                                         #
# ------------------------------------------------------------------ #

class TestBurstyMode:
    def test_fires_at_elevated_rate_vs_flat(self):
        """Bursty at prob=0.3 should fire more often than flat at prob=0.3
        because its effective trigger is prob*2."""
        flat = EnvironmentStressor(prob=0.3, seed=99, mode="flat")
        bursty = EnvironmentStressor(prob=0.3, seed=99, mode="bursty")
        flat_events = _run_ticks(flat, 1000)
        bursty_events = _run_ticks(bursty, 1000)
        assert len(bursty_events) >= len(flat_events)

    def test_chained_events_contain_plus(self):
        """With prob=1 the chain roll frequently fires, producing 'X + Y'."""
        s = EnvironmentStressor(prob=1.0, seed=0, mode="bursty")
        chained = []
        for _ in range(500):
            result = s.maybe_disturb(_fresh_state())
            if " + " in result:
                chained.append(result)
        # With prob=1 and chain rate 0.3, expect many chained events
        assert len(chained) > 50

    def test_chained_event_both_halves_are_valid(self):
        s = EnvironmentStressor(prob=1.0, seed=0, mode="bursty")
        for _ in range(200):
            result = s.maybe_disturb(_fresh_state())
            if " + " in result:
                left, right = result.split(" + ", 1)
                assert any(t in left for t in _DISTURBANCE_TYPES)
                assert any(t in right for t in _DISTURBANCE_TYPES)
                return
        pytest.skip("no chained event encountered")

    def test_returns_string(self):
        s = EnvironmentStressor(prob=0.5, seed=1, mode="bursty")
        for _ in range(30):
            assert isinstance(s.maybe_disturb(_fresh_state()), str)


# ------------------------------------------------------------------ #
# Hostile-window mode                                                 #
# ------------------------------------------------------------------ #

class TestHostileWindowMode:
    def _force_window(self, s: EnvironmentStressor, ticks: int = 200) -> None:
        """Directly set the stressor into a hostile window for testing."""
        s._in_hostile_window = True
        s._hostile_ticks_remaining = ticks

    def test_returns_string(self):
        s = EnvironmentStressor(prob=0.5, seed=3, mode="hostile_windows")
        assert isinstance(s.maybe_disturb(_fresh_state()), str)

    def test_events_inside_window_have_prefix(self):
        """Events fired during a hostile window must carry the [HOSTILE_WINDOW] prefix."""
        s = EnvironmentStressor(prob=1.0, seed=0, mode="hostile_windows")
        self._force_window(s)
        for _ in range(20):
            result = s.maybe_disturb(_fresh_state())
            if result:
                assert result.startswith("[HOSTILE_WINDOW]"), result

    def test_events_outside_window_lack_prefix(self):
        """Events fired outside a hostile window must NOT carry the prefix."""
        s = EnvironmentStressor(prob=1.0, seed=0, mode="hostile_windows")
        # Ensure we are definitely outside a window
        s._in_hostile_window = False
        s._hostile_ticks_remaining = 0
        # Also zero the entry probability temporarily via monkey-patch to prevent entry
        import thermodynamic_agency.cognition.environment as env_mod
        original = env_mod._HW_ENTRY_PROB
        env_mod._HW_ENTRY_PROB = 0.0
        try:
            for _ in range(50):
                result = s.maybe_disturb(_fresh_state())
                if result:
                    assert not result.startswith("[HOSTILE_WINDOW]"), result
        finally:
            env_mod._HW_ENTRY_PROB = original

    def test_window_expires_after_duration(self):
        """in_hostile_window returns False once the countdown reaches zero.

        Use prob=1.0 so maybe_disturb actually enters _maybe_disturb_hostile_windows
        and decrements the countdown every tick.
        """
        s = EnvironmentStressor(prob=1.0, seed=0, mode="hostile_windows")
        # Force a window of exactly 5 ticks
        s._in_hostile_window = True
        s._hostile_ticks_remaining = 5
        for i in range(5):
            assert s.in_hostile_window, f"expected window open at iteration {i}"
            s.maybe_disturb(_fresh_state())
        assert not s.in_hostile_window

    def test_window_flag_transitions(self):
        """Run many ticks and check window transitions are logically consistent."""
        s = EnvironmentStressor(prob=0.5, seed=42, mode="hostile_windows")
        transitions = []
        prev = False
        for _ in range(5000):
            s.maybe_disturb(_fresh_state())
            cur = s.in_hostile_window
            if cur != prev:
                transitions.append(cur)
                prev = cur
        # There should be at least some window openings and closings
        assert True in transitions
        assert False in transitions

    def test_elevated_intensity_during_window(self):
        """Hostile-window events should be larger than baseline events
        because intensity is multiplied by _HW_INTENSITY_MULTIPLIER."""
        baseline_losses = []
        window_losses = []

        import thermodynamic_agency.cognition.environment as env_mod

        # Baseline: force OUTSIDE window and collect energy losses
        s = EnvironmentStressor(prob=1.0, intensity=1.0, seed=7, mode="hostile_windows")
        s._in_hostile_window = False
        original_entry = env_mod._HW_ENTRY_PROB
        env_mod._HW_ENTRY_PROB = 0.0
        try:
            for _ in range(200):
                state = _fresh_state()
                s.maybe_disturb(state)
                baseline_losses.append(80.0 - state.energy)
        finally:
            env_mod._HW_ENTRY_PROB = original_entry

        # Window: force INSIDE window and collect energy losses
        s2 = EnvironmentStressor(prob=1.0, intensity=1.0, seed=7, mode="hostile_windows")
        s2._in_hostile_window = True
        s2._hostile_ticks_remaining = 10_000  # never expires during this test
        for _ in range(200):
            state = _fresh_state()
            s2.maybe_disturb(state)
            window_losses.append(80.0 - state.energy)

        avg_baseline = sum(baseline_losses) / len(baseline_losses)
        avg_window = sum(window_losses) / len(window_losses)
        # Window intensity is 1.8× baseline — average loss should be larger
        assert avg_window > avg_baseline

    def test_window_stressor_prob_multiplied(self):
        """Inside a window the effective prob is multiplied by _HW_STRESSOR_MULTIPLIER,
        so at prob=0.5 the window hit-rate should be higher than outside."""
        import thermodynamic_agency.cognition.environment as env_mod

        def count_hits(in_window: bool, n: int = 1000) -> int:
            s = EnvironmentStressor(prob=0.5, seed=11, mode="hostile_windows")
            if in_window:
                s._in_hostile_window = True
                s._hostile_ticks_remaining = n + 1
            else:
                s._in_hostile_window = False
                # Prevent window entry
            original_entry = env_mod._HW_ENTRY_PROB
            env_mod._HW_ENTRY_PROB = 0.0
            try:
                hits = 0
                for _ in range(n):
                    if s.maybe_disturb(_fresh_state()):
                        hits += 1
                return hits
            finally:
                env_mod._HW_ENTRY_PROB = original_entry

        outside_hits = count_hits(in_window=False)
        inside_hits = count_hits(in_window=True)
        assert inside_hits > outside_hits

    def test_seeded_is_reproducible(self):
        s1 = EnvironmentStressor(prob=0.4, seed=55, mode="hostile_windows")
        s2 = EnvironmentStressor(prob=0.4, seed=55, mode="hostile_windows")
        for _ in range(60):
            st1, st2 = _fresh_state(), _fresh_state()
            assert s1.maybe_disturb(st1) == s2.maybe_disturb(st2)
