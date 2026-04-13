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
