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
