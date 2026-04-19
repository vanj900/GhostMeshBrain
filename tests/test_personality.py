"""Tests for personality masks and MaskRotator."""

import pytest

from thermodynamic_agency.cognition.personality import (
    MaskRotator,
    ALL_MASKS,
    _MASK_MAP,
)


class TestMaskRotator:
    def test_initial_mask(self):
        rotator = MaskRotator(initial_mask="Guardian")
        assert rotator.active.name == "Guardian"

    def test_all_masks_accessible(self):
        names = {m.name for m in ALL_MASKS}
        assert "Healer" in names
        assert "Judge" in names
        assert "Courier" in names
        assert "Dreamer" in names
        assert "Guardian" in names

    def test_force_rotate(self):
        rotator = MaskRotator(initial_mask="Guardian")
        # Tick past min_ticks
        for _ in range(10):
            rotator.tick(1)
        rotator.maybe_rotate(10, force="Healer")
        assert rotator.active.name == "Healer"

    def test_metabolic_hint_rotates_mask(self):
        rotator = MaskRotator(initial_mask="Guardian")
        # Tick past min_ticks
        for _ in range(10):
            rotator.tick(1)
        rotator.maybe_rotate(10, metabolic_hint="REST")
        assert rotator.active.name == "Dreamer"

    def test_status_returns_dict(self):
        rotator = MaskRotator()
        status = rotator.status()
        assert "active_mask" in status
        assert "ticks_active" in status
        assert "description" in status

    def test_mask_has_efe_overrides(self):
        guardian = _MASK_MAP["Guardian"]
        assert "energy" in guardian.efe_precision_overrides

    def test_min_ticks_prevents_early_rotation(self):
        rotator = MaskRotator(initial_mask="Guardian")
        # Don't tick past min_ticks
        rotator.maybe_rotate(0, metabolic_hint="REST")
        # Should still be Guardian because ticks_active < min_ticks
        assert rotator.active.name == "Guardian"

    def test_evolved_stage_dreamer_boost_after_long_non_exploratory(self):
        """At evolved stage with neutral affect and DECIDE, rotate to Dreamer after 20+ ticks."""
        rotator = MaskRotator(initial_mask="CentralExec")
        # Simulate 25 ticks in CentralExec
        for _ in range(25):
            rotator.tick(1)
        result = rotator.maybe_rotate(
            25,
            metabolic_hint="DECIDE",
            affect=0.0,      # neutral affect
            stage="evolved",
        )
        assert result.name == "Dreamer", (
            "Evolved stage with neutral affect + long CentralExec dwell should rotate to Dreamer"
        )

    def test_evolved_stage_dreamer_boost_not_premature(self):
        """At evolved stage with fewer than 20 ticks active, don't force Dreamer."""
        rotator = MaskRotator(initial_mask="CentralExec")
        # Only 10 ticks (below the 20-tick threshold)
        for _ in range(10):
            rotator.tick(1)
        result = rotator.maybe_rotate(
            10,
            metabolic_hint="DECIDE",
            affect=0.0,
            stage="evolved",
        )
        # Should stay in CentralExec (normal DECIDE→CentralExec routing)
        assert result.name == "CentralExec"

    def test_evolved_stage_dreamer_boost_not_when_stressed(self):
        """At evolved stage with strong negative affect, Guardian takes priority over Dreamer."""
        rotator = MaskRotator(initial_mask="CentralExec")
        for _ in range(25):
            rotator.tick(1)
        result = rotator.maybe_rotate(
            25,
            metabolic_hint="REPAIR",
            affect=-0.5,     # strongly negative affect
            stage="evolved",
        )
        # Negative affect + REPAIR → Guardian, not Dreamer
        assert result.name == "Guardian"

    def test_non_evolved_stage_no_dreamer_bias(self):
        """At aware/dormant stage the Dreamer bias should not apply."""
        rotator = MaskRotator(initial_mask="CentralExec")
        for _ in range(25):
            rotator.tick(1)
        result = rotator.maybe_rotate(
            25,
            metabolic_hint="DECIDE",
            affect=0.0,
            stage="aware",   # not evolved
        )
        # Normal routing: DECIDE + neutral affect → CentralExec (no Dreamer bias)
        assert result.name == "CentralExec"

    def test_stage_parameter_is_optional(self):
        """maybe_rotate should work fine without passing stage (defaults to dormant)."""
        rotator = MaskRotator(initial_mask="Guardian")
        for _ in range(10):
            rotator.tick(1)
        # Should not raise even without stage kwarg
        result = rotator.maybe_rotate(10, metabolic_hint="REST")
        assert result.name is not None
