"""Tests for CollapseProbe — rolling-window phase-transition detector."""

import math

import pytest

from thermodynamic_agency.cognition.collapse_probe import (
    CollapseProbe,
    CollapseSnapshot,
    _shannon_entropy,
    _GUARDIAN_MASKS,
    _DREAMER_MASKS,
)


class TestShannonEntropy:
    def test_uniform_two_classes(self):
        h = _shannon_entropy([50, 50], 100)
        assert abs(h - 1.0) < 1e-9

    def test_uniform_four_classes(self):
        h = _shannon_entropy([25, 25, 25, 25], 100)
        assert abs(h - 2.0) < 1e-9

    def test_pure_single_class(self):
        h = _shannon_entropy([100], 100)
        assert h == 0.0

    def test_empty_returns_zero(self):
        assert _shannon_entropy([], 0) == 0.0

    def test_partial_mix(self):
        # 75/25 split should have entropy between 0 and 1
        h = _shannon_entropy([75, 25], 100)
        assert 0.0 < h < 1.0


class TestCollapseProbeEmpty:
    def test_empty_probe_returns_zero_snapshot(self):
        probe = CollapseProbe()
        snap = probe._empty_snapshot()
        assert snap.ticks_in_window == 0
        assert snap.pre_collapse_score == 0.0
        assert not snap.is_near_transition
        assert snap.plasticity_index == 0.0


class TestCollapseProbeBasic:
    def _make_probe(self, window: int = 50) -> CollapseProbe:
        return CollapseProbe(window=window, detection_threshold=0.65)

    def _tick(
        self,
        probe: CollapseProbe,
        action: str = "DECIDE",
        mask: str = "Guardian",
        free_energy: float = 20.0,
        allostatic_load: float = 30.0,
        energy: float = 80.0,
        heat: float = 20.0,
    ) -> CollapseSnapshot:
        return probe.update(
            action=action,
            mask=mask,
            free_energy=free_energy,
            allostatic_load=allostatic_load,
            energy=energy,
            heat=heat,
        )

    def test_single_tick_accumulates(self):
        probe = self._make_probe()
        snap = self._tick(probe)
        assert snap.ticks_in_window == 1

    def test_window_capped(self):
        probe = self._make_probe(window=10)
        for _ in range(20):
            snap = self._tick(probe)
        assert snap.ticks_in_window == 10

    def test_pure_guardian_fraction_one(self):
        probe = self._make_probe()
        for _ in range(30):
            self._tick(probe, mask="Guardian")
        snap = self._tick(probe, mask="Guardian")
        assert abs(snap.guardian_fraction - 1.0) < 1e-9
        assert snap.dreamer_fraction == 0.0

    def test_pure_dreamer_fraction_one(self):
        probe = self._make_probe()
        for _ in range(30):
            self._tick(probe, mask="Dreamer")
        snap = self._tick(probe, mask="Dreamer")
        assert abs(snap.dreamer_fraction - 1.0) < 1e-9
        assert snap.guardian_fraction == 0.0

    def test_mixed_masks_entropy_nonzero(self):
        probe = self._make_probe()
        masks = ["Guardian", "Dreamer", "Healer", "Judge", "Courier"]
        for i in range(50):
            self._tick(probe, mask=masks[i % len(masks)])
        snap = self._tick(probe, mask="Guardian")
        assert snap.mask_entropy > 0.0

    def test_pure_single_action_entropy_zero(self):
        probe = self._make_probe()
        for _ in range(30):
            self._tick(probe, action="DECIDE")
        snap = self._tick(probe, action="DECIDE")
        assert snap.action_entropy == 0.0

    def test_uniform_four_actions_entropy_two(self):
        probe = self._make_probe(window=100)
        actions = ["FORAGE", "REST", "REPAIR", "DECIDE"]
        for i in range(100):
            self._tick(probe, action=actions[i % 4])
        snap = self._tick(probe, action="FORAGE")
        # 25 each out of 101 — close to 2 bits
        assert snap.action_entropy > 1.9

    def test_plasticity_index_high_when_dreamer_dominant(self):
        probe = self._make_probe()
        for _ in range(40):
            self._tick(probe, mask="Dreamer")
        # add a tiny guardian presence so guardian_fraction > 0
        self._tick(probe, mask="Guardian")
        snap = self._tick(probe, mask="Dreamer")
        assert snap.plasticity_index > 1.0

    def test_plasticity_index_low_when_guardian_dominant(self):
        probe = self._make_probe()
        for _ in range(40):
            self._tick(probe, mask="Guardian")
        snap = self._tick(probe, mask="Guardian")
        # plasticity_index << 1 when only guardian
        assert snap.plasticity_index < 0.1

    def test_pre_collapse_score_bounded(self):
        probe = self._make_probe()
        for _ in range(50):
            snap = self._tick(probe, mask="Guardian", action="REPAIR",
                              allostatic_load=80.0, free_energy=60.0)
        assert 0.0 <= snap.pre_collapse_score <= 1.0

    def test_near_transition_fires_under_sustained_guardian_load(self):
        """Sustained Guardian dominance + rising AL should trigger transition."""
        probe = CollapseProbe(window=200, detection_threshold=0.35)
        for i in range(200):
            snap = probe.update(
                action="REPAIR",
                mask="Guardian",
                free_energy=55.0,
                allostatic_load=60.0 + i * 0.1,
                energy=40.0,
                heat=50.0,
            )
        assert snap.is_near_transition

    def test_near_transition_false_in_healthy_dreamer_regime(self):
        """Dreamer-dominated, low-stress regime should NOT flag transition."""
        probe = CollapseProbe(window=100, detection_threshold=0.65)
        for _ in range(100):
            snap = probe.update(
                action="DECIDE",
                mask="Dreamer",
                free_energy=10.0,
                allostatic_load=10.0,
                energy=80.0,
                heat=15.0,
            )
        assert not snap.is_near_transition

    def test_derivatives_track_rising_allostatic(self):
        probe = self._make_probe(window=200)
        for i in range(50):
            self._tick(probe, allostatic_load=i * 1.0)
        snap = self._tick(probe, allostatic_load=50.0)
        # d_allostatic should be positive (load rising)
        assert snap.d_allostatic > 0.0

    def test_derivatives_track_falling_energy(self):
        probe = self._make_probe(window=200)
        for i in range(50):
            self._tick(probe, energy=80.0 - i * 0.5)
        snap = self._tick(probe, energy=55.0)
        # d_energy should be negative (energy falling)
        assert snap.d_energy < 0.0

    def test_precision_weights_stored(self):
        probe = self._make_probe(window=20)
        snap = probe.update(
            action="DECIDE",
            mask="Guardian",
            free_energy=25.0,
            allostatic_load=30.0,
            energy=70.0,
            heat=25.0,
            precision_weights={"energy": 3.5, "heat": 2.8, "waste": 1.0,
                               "integrity": 2.0, "stability": 1.5},
        )
        assert snap.mean_precision_energy > 0.0
        assert snap.mean_precision_heat > 0.0

    def test_efe_components_stored(self):
        probe = self._make_probe(window=20)
        snap = probe.update(
            action="DECIDE",
            mask="Guardian",
            free_energy=25.0,
            allostatic_load=30.0,
            energy=70.0,
            heat=25.0,
            efe_accuracy=120.5,
            efe_complexity=4.3,
        )
        assert snap.mean_efe_accuracy > 0.0
        assert snap.mean_efe_complexity > 0.0

    def test_reset_clears_state(self):
        probe = self._make_probe(window=20)
        for _ in range(10):
            self._tick(probe, mask="Guardian")
        probe.reset()
        snap = self._tick(probe, mask="Guardian")
        assert snap.ticks_in_window == 1

    def test_snapshot_window_field_matches_config(self):
        probe = CollapseProbe(window=42)
        snap = probe.update(
            action="DECIDE", mask="Dreamer",
            free_energy=5.0, allostatic_load=5.0,
            energy=90.0, heat=10.0,
        )
        assert snap.window == 42


class TestCollapseProbeIntegration:
    """Integration tests that verify the probe behaves correctly over a full
    simulated lifecycle with regime shifts."""

    def test_phase_transition_detected_after_regime_shift(self):
        """Start in dreamer regime, shift to guardian attractor; probe detects it."""
        probe = CollapseProbe(window=100, detection_threshold=0.50)

        # Phase 1: 100 ticks of healthy Dreamer regime
        for _ in range(100):
            probe.update(
                action="DECIDE", mask="Dreamer",
                free_energy=12.0, allostatic_load=15.0,
                energy=80.0, heat=18.0,
            )

        # Phase 2: shift to stressed Guardian attractor
        snap_before = probe.update(
            action="DECIDE", mask="Dreamer",
            free_energy=12.0, allostatic_load=15.0,
            energy=80.0, heat=18.0,
        )
        assert not snap_before.is_near_transition  # still healthy

        # Flood the 100-tick window with Guardian + high AL
        for i in range(100):
            snap = probe.update(
                action="REPAIR", mask="Guardian",
                free_energy=55.0, allostatic_load=65.0 + i * 0.1,
                energy=45.0, heat=55.0,
            )

        # After 100 ticks of guardian dominance, transition should be flagged
        assert snap.is_near_transition
        assert snap.guardian_fraction > 0.5

    def test_plasticity_restores_after_dreamer_recovery(self):
        """After Guardian dominance, returning to Dreamer raises plasticity_index."""
        probe = CollapseProbe(window=50)

        # 50 ticks of Guardian
        for _ in range(50):
            probe.update(
                action="REPAIR", mask="Guardian",
                free_energy=50.0, allostatic_load=60.0,
                energy=40.0, heat=55.0,
            )

        # 50 ticks of Dreamer recovery
        for _ in range(50):
            snap = probe.update(
                action="DECIDE", mask="Dreamer",
                free_energy=10.0, allostatic_load=10.0,
                energy=80.0, heat=15.0,
            )

        # After full window of Dreamer, plasticity should be > 1
        assert snap.plasticity_index > 1.0
