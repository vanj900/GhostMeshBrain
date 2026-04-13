"""Tests for Janitor and Surgeon subsystems."""

import pytest

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.memory.diary import RamDiary, DiaryEntry
from thermodynamic_agency.cognition.janitor import Janitor
from thermodynamic_agency.cognition.surgeon import Surgeon, BeliefPrior


@pytest.fixture
def diary(tmp_path):
    db_path = str(tmp_path / "subsys_diary.db")
    d = RamDiary(path=db_path)
    yield d
    d.close()


class TestJanitor:
    def test_run_on_empty_diary_returns_zero_report(self, diary):
        state = MetabolicState(heat=85.0, waste=80.0)
        janitor = Janitor(diary=diary, use_llm=False)
        report = janitor.run(state)
        assert report.entries_compressed == 0
        assert report.insights_generated == 0

    def test_run_compresses_entries(self, diary):
        for i in range(10):
            diary.append(DiaryEntry(tick=i, role="thought", content=f"thought number {i}"))
        state = MetabolicState(heat=85.0, waste=80.0)
        janitor = Janitor(diary=diary, use_llm=False)
        report = janitor.run(state, compress_last_n=10)
        assert report.entries_compressed == 10
        assert report.insights_generated > 0

    def test_run_reduces_heat_and_waste(self, diary):
        diary.append(DiaryEntry(tick=1, role="thought", content="content"))
        state = MetabolicState(heat=85.0, waste=80.0)
        initial_heat = state.heat
        initial_waste = state.waste
        janitor = Janitor(diary=diary, use_llm=False)
        janitor.run(state)
        assert state.heat < initial_heat
        assert state.waste < initial_waste

    def test_run_costs_energy(self, diary):
        diary.append(DiaryEntry(tick=1, role="thought", content="content"))
        state = MetabolicState(energy=50.0)
        janitor = Janitor(diary=diary, use_llm=False)
        janitor.run(state)
        assert state.energy < 50.0

    def test_insights_stored_in_diary(self, diary):
        for i in range(5):
            diary.append(DiaryEntry(tick=i, role="action", content=f"action {i}"))
        state = MetabolicState(heat=85.0, waste=80.0)
        janitor = Janitor(diary=diary, use_llm=False)
        janitor.run(state)
        insights = diary.insights()
        assert len(insights) >= 1


class TestSurgeon:
    def test_run_improves_integrity(self, diary):
        state = MetabolicState(integrity=40.0, stability=35.0)
        surgeon = Surgeon(diary=diary)
        initial_integrity = state.integrity
        surgeon.run(state)
        assert state.integrity > initial_integrity

    def test_run_improves_stability(self, diary):
        state = MetabolicState(integrity=40.0, stability=35.0)
        surgeon = Surgeon(diary=diary)
        initial_stability = state.stability
        surgeon.run(state)
        assert state.stability > initial_stability

    def test_run_costs_energy(self, diary):
        state = MetabolicState(energy=80.0, integrity=40.0)
        surgeon = Surgeon(diary=diary)
        surgeon.run(state)
        assert state.energy < 80.0

    def test_run_logs_to_diary(self, diary):
        state = MetabolicState(integrity=40.0)
        surgeon = Surgeon(diary=diary)
        surgeon.run(state)
        entries = diary.all_entries()
        assert any(e.role == "repair" for e in entries)

    def test_report_fields(self, diary):
        state = MetabolicState(integrity=40.0, stability=35.0)
        surgeon = Surgeon(diary=diary)
        report = surgeon.run(state)
        assert report.beliefs_audited >= 0
        assert report.integrity_gain > 0
        assert report.stability_gain > 0
        assert isinstance(report.diagnosis, str)

    def test_identify_frozen_priors(self, diary):
        # Set a prior with high precision + many errors
        priors = [BeliefPrior(name="bad_prior", value=True, precision=4.0, error_count=5)]
        surgeon = Surgeon(diary=diary, priors=priors)
        state = MetabolicState()
        frozen = surgeon._identify_frozen(state)
        assert any(p.name == "bad_prior" for p in frozen)
