"""Integration test — run the full pulse loop for a few ticks."""

import json
import os
import tempfile

import pytest

from thermodynamic_agency.pulse import GhostMesh
from thermodynamic_agency.core.metabolic import MetabolicState


@pytest.fixture
def tmp_state_file(tmp_path):
    return str(tmp_path / "ghost_metabolic.json")


@pytest.fixture
def tmp_diary_path(tmp_path):
    return str(tmp_path / "ghost_diary.db")


class TestPulseIntegration:
    def test_run_multiple_ticks_without_death(self, tmp_state_file, tmp_diary_path, monkeypatch):
        monkeypatch.setenv("GHOST_STATE_FILE", tmp_state_file)
        monkeypatch.setenv("GHOST_DIARY_PATH", tmp_diary_path)
        monkeypatch.setenv("GHOST_HUD", "0")   # suppress terminal output
        monkeypatch.setenv("GHOST_PULSE", "0")  # no sleep

        mesh = GhostMesh()
        mesh.run(max_ticks=5)

        # State should have been persisted
        assert os.path.exists(tmp_state_file)
        with open(tmp_state_file) as f:
            data = json.load(f)
        assert data["entropy"] >= 5

    def test_state_persists_between_instances(self, tmp_state_file, tmp_diary_path, monkeypatch):
        monkeypatch.setenv("GHOST_STATE_FILE", tmp_state_file)
        monkeypatch.setenv("GHOST_DIARY_PATH", tmp_diary_path)
        monkeypatch.setenv("GHOST_HUD", "0")
        monkeypatch.setenv("GHOST_PULSE", "0")

        mesh1 = GhostMesh()
        mesh1.run(max_ticks=3)
        entropy_after_first = mesh1.state.entropy

        mesh2 = GhostMesh()
        assert mesh2.state.entropy == entropy_after_first

    def test_diary_records_entries(self, tmp_state_file, tmp_diary_path, monkeypatch):
        monkeypatch.setenv("GHOST_STATE_FILE", tmp_state_file)
        monkeypatch.setenv("GHOST_DIARY_PATH", tmp_diary_path)
        monkeypatch.setenv("GHOST_HUD", "0")
        monkeypatch.setenv("GHOST_PULSE", "0")

        mesh = GhostMesh()
        mesh.run(max_ticks=3)
        count = mesh.diary.entry_count()
        # At least the awakening message + 3 action entries
        assert count >= 1

    def test_forage_increases_energy(self, tmp_state_file, tmp_diary_path, monkeypatch):
        monkeypatch.setenv("GHOST_STATE_FILE", tmp_state_file)
        monkeypatch.setenv("GHOST_DIARY_PATH", tmp_diary_path)
        monkeypatch.setenv("GHOST_HUD", "0")
        monkeypatch.setenv("GHOST_PULSE", "0")

        mesh = GhostMesh()
        mesh.state.energy = 15.0  # below FORAGE threshold
        initial_energy = mesh.state.energy
        mesh._forage()
        assert mesh.state.energy > initial_energy

    def test_autonomic_intervention_logs_diary_entry(
        self, tmp_state_file, tmp_diary_path, monkeypatch
    ):
        """Autonomic intervention must fire and write a diary entry when CollapseProbe
        flags near_transition at evolved stage."""
        monkeypatch.setenv("GHOST_STATE_FILE", tmp_state_file)
        monkeypatch.setenv("GHOST_DIARY_PATH", tmp_diary_path)
        monkeypatch.setenv("GHOST_HUD", "0")
        monkeypatch.setenv("GHOST_PULSE", "0")
        monkeypatch.setenv("GHOST_ENV_EVENTS", "0")

        from thermodynamic_agency.cognition.collapse_probe import CollapseSnapshot

        mesh = GhostMesh()
        # Inject an evolved state so the stage gate passes
        mesh.state.stage = "evolved"

        # Inject a fake near-transition snapshot from the "previous" tick
        fake_snap = CollapseSnapshot(
            window=500,
            ticks_in_window=200,
            action_entropy=0.2,
            mask_entropy=0.3,
            guardian_fraction=0.65,
            dreamer_fraction=0.03,
            plasticity_index=0.04,
            mean_free_energy=25.0,
            d_allostatic=0.6,
            d_energy=-0.1,
            d_heat=0.05,
            mean_precision_energy=2.0,
            mean_precision_heat=3.0,
            mean_precision_waste=1.5,
            mean_precision_integrity=2.5,
            mean_precision_stability=2.0,
            mean_efe_accuracy=5.0,
            mean_efe_complexity=2.0,
            pre_collapse_score=0.45,
            is_near_transition=True,
        )
        mesh._last_collapse_snapshot = fake_snap

        entries_before = mesh.diary.entry_count()
        # Seed a non-zero precision weight for heat so relaxation can be verified
        mesh._last_precision_weights = {"heat": 3.0, "stability": 2.5, "energy": 2.0}
        heat_prec_before = mesh._last_precision_weights["heat"]
        stability_prec_before = mesh._last_precision_weights["stability"]
        energy_before = mesh.state.energy
        heat_before = mesh.state.heat
        waste_before = mesh.state.waste

        mesh._apply_autonomic_intervention()

        entries_after = mesh.diary.entry_count()

        # Should have added a diary entry
        assert entries_after > entries_before
        recent = mesh.diary.recent(n=5)
        intervention_entries = [
            e for e in recent if "AUTONOMIC_INTERVENTION" in e.content
        ]
        assert intervention_entries, "Expected an AUTONOMIC_INTERVENTION diary entry"
        # Mask should now be Dreamer
        assert mesh.rotator.active.name == "Dreamer"
        # Precision relaxation: heat and stability weights should be 20% lower
        assert mesh._last_precision_weights["heat"] == pytest.approx(
            heat_prec_before * 0.80, abs=1e-6
        ), "heat precision weight should be reduced by 20%"
        assert mesh._last_precision_weights["stability"] == pytest.approx(
            stability_prec_before * 0.80, abs=1e-6
        ), "stability precision weight should be reduced by 20%"
        # Non-relaxed vital should be unchanged
        assert mesh._last_precision_weights["energy"] == pytest.approx(2.0)
        # Metabolic cost: ΔE=-1.5, ΔT=+0.8, ΔW=+0.5
        assert mesh.state.energy == pytest.approx(energy_before - 1.5, abs=1e-6)
        assert mesh.state.heat == pytest.approx(heat_before + 0.8, abs=1e-6)
        assert mesh.state.waste == pytest.approx(waste_before + 0.5, abs=1e-6)
