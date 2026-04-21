"""Tests for LineageTracker fitness metrics and MortalityLineageExperiment."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import csv as csv_mod
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortality_lineage_experiment import MortalityLineageExperiment, _CSV_FIELDS
from thermodynamic_agency.evolution.lineage import (
    Lineage,
    LineageTracker,
    DEFAULT_MUTATION_RATE,
    _generate_lineage_id,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_lineage(
    life_number: int = 1,
    lifespan: int = 100,
    dreamer_fraction: float = 0.5,
    guardian_fraction: float = 0.3,
    plasticity_index: float = 1.0,
    parent_id: str | None = None,
    cause_of_death: str = "EnergyDeathException",
) -> Lineage:
    return Lineage(
        lineage_id=_generate_lineage_id(),
        parent_id=parent_id,
        life_number=life_number,
        lifespan=lifespan,
        interiority_score=0.5,
        dreamer_fraction=dreamer_fraction,
        guardian_fraction=guardian_fraction,
        plasticity_index=plasticity_index,
        cause_of_death=cause_of_death,
        top_q_entries=[],
        mask_preferences={},
        mutation_rate=DEFAULT_MUTATION_RATE,
    )


def _tracker_with_records(*records: Lineage) -> LineageTracker:
    tmpdir = tempfile.mkdtemp()
    tracker = LineageTracker(path=os.path.join(tmpdir, "lineage.jsonl"))
    for rec in records:
        tracker.record(rec)
    return tracker


# ── lineage_fitness ───────────────────────────────────────────────────────────


class TestLineageFitness:
    def test_empty_tracker_returns_empty(self):
        tmpdir = tempfile.mkdtemp()
        tracker = LineageTracker(path=os.path.join(tmpdir, "lineage.jsonl"))
        assert tracker.lineage_fitness() == []

    def test_single_record_returns_score_in_range(self):
        tracker = _tracker_with_records(_make_lineage(lifespan=100, dreamer_fraction=0.5))
        scores = tracker.lineage_fitness()
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_highest_dreamer_fraction_gets_highest_score_when_lifespan_equal(self):
        t = _tracker_with_records(
            _make_lineage(lifespan=100, dreamer_fraction=0.8),
            _make_lineage(lifespan=100, dreamer_fraction=0.2),
        )
        scores = t.lineage_fitness(lifespan_weight=0.0, dreamer_weight=1.0)
        assert scores[0] > scores[1]

    def test_highest_lifespan_gets_highest_score_when_dreamer_equal(self):
        t = _tracker_with_records(
            _make_lineage(lifespan=200, dreamer_fraction=0.5),
            _make_lineage(lifespan=50, dreamer_fraction=0.5),
        )
        scores = t.lineage_fitness(lifespan_weight=1.0, dreamer_weight=0.0)
        assert scores[0] > scores[1]

    def test_max_lifespan_entry_normalises_to_one_component(self):
        t = _tracker_with_records(
            _make_lineage(lifespan=100, dreamer_fraction=0.0),
        )
        scores = t.lineage_fitness(lifespan_weight=1.0, dreamer_weight=0.0)
        assert scores[0] == pytest.approx(1.0)

    def test_scores_bounded_zero_to_one(self):
        records = [_make_lineage(lifespan=i * 10, dreamer_fraction=i * 0.1) for i in range(1, 11)]
        t = _tracker_with_records(*records)
        scores = t.lineage_fitness()
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_weights_sum_effect_is_correct(self):
        t = _tracker_with_records(
            _make_lineage(lifespan=100, dreamer_fraction=1.0),
        )
        # With equal lifespan and full dreamer, score = 0.6*1 + 0.4*1 = 1.0
        scores = t.lineage_fitness(lifespan_weight=0.6, dreamer_weight=0.4)
        assert scores[0] == pytest.approx(1.0)


# ── plasticity_selection_signal ───────────────────────────────────────────────


class TestPlasticitySelectionSignal:
    def test_fewer_than_3_returns_zero(self):
        t = _tracker_with_records(
            _make_lineage(lifespan=100, dreamer_fraction=0.5),
        )
        assert t.plasticity_selection_signal() == 0.0

    def test_two_records_returns_zero(self):
        t = _tracker_with_records(
            _make_lineage(lifespan=100, dreamer_fraction=0.5),
            _make_lineage(lifespan=200, dreamer_fraction=0.8),
        )
        assert t.plasticity_selection_signal() == 0.0

    def test_perfect_positive_correlation(self):
        """dreamer_fraction increases monotonically → lifespan of next generation
        also increases monotonically → r should be close to +1."""
        records = [
            _make_lineage(lifespan=100 * (i + 1), dreamer_fraction=0.1 * (i + 1))
            for i in range(5)
        ]
        t = _tracker_with_records(*records)
        r = t.plasticity_selection_signal()
        assert r > 0.8, f"Expected strong positive r, got {r}"

    def test_perfect_negative_correlation(self):
        """dreamer_fraction decreases as lifespan increases → r close to −1."""
        records = [
            _make_lineage(lifespan=100 * (i + 1), dreamer_fraction=0.5 - 0.08 * i)
            for i in range(5)
        ]
        t = _tracker_with_records(*records)
        r = t.plasticity_selection_signal()
        assert r < -0.8, f"Expected strong negative r, got {r}"

    def test_zero_variance_dreamer_returns_zero(self):
        """All dreamer fractions identical → no variance → r = 0."""
        records = [_make_lineage(lifespan=100 * (i + 1), dreamer_fraction=0.5) for i in range(5)]
        t = _tracker_with_records(*records)
        r = t.plasticity_selection_signal()
        assert r == 0.0

    def test_result_in_minus_one_plus_one(self):
        import random as _random
        rng = _random.Random(99)
        records = [
            _make_lineage(lifespan=rng.randint(50, 500), dreamer_fraction=rng.uniform(0.0, 1.0))
            for _ in range(10)
        ]
        t = _tracker_with_records(*records)
        r = t.plasticity_selection_signal()
        assert -1.0 <= r <= 1.0

    def test_empty_tracker_returns_zero(self):
        tmpdir = tempfile.mkdtemp()
        t = LineageTracker(path=os.path.join(tmpdir, "lineage.jsonl"))
        assert t.plasticity_selection_signal() == 0.0


# ── MortalityLineageExperiment smoke test ────────────────────────────────────


class TestMortalityLineageExperimentSmoke:
    """Smoke tests: run a tiny experiment and check output shape."""

    def _run_tiny(self, immortal: bool, tmpdir: str) -> list:
        exp = MortalityLineageExperiment(
            n_generations=2,
            ticks_per_life=50,
            runs_per_config=1,
            output_dir=tmpdir,
            seed_base=7,
            stressor_probs=[0.0, 0.2],
            stressor_intensities=[1.0],
            include_immortal=immortal,
        )
        _, csv_path = exp.run()
        with open(csv_path, newline="") as fh:
            return list(csv_mod.DictReader(fh))

    def test_mortal_only_produces_correct_row_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = self._run_tiny(immortal=False, tmpdir=tmpdir)
            # 2 stressor_probs × 1 intensity × 1 run × 2 generations = 4 rows
            assert len(rows) == 4

    def test_immortal_produces_both_mortal_and_immortal_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = self._run_tiny(immortal=True, tmpdir=tmpdir)
            mortal_rows = [r for r in rows if r["immortal"].lower() == "false"]
            immortal_rows = [r for r in rows if r["immortal"].lower() == "true"]
            assert len(mortal_rows) > 0
            assert len(immortal_rows) > 0

    def test_output_has_required_columns(self):
        from scripts.mortality_lineage_experiment import _CSV_FIELDS

        with tempfile.TemporaryDirectory() as tmpdir:
            rows = self._run_tiny(immortal=False, tmpdir=tmpdir)
            for field in _CSV_FIELDS:
                assert field in rows[0], f"Missing column: {field}"

    def test_generation_values_are_sequential(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = self._run_tiny(immortal=False, tmpdir=tmpdir)
            mortal = [r for r in rows if r["immortal"].lower() == "false"
                      and r["stressor_prob"] == "0.0"]
            gens = sorted(int(r["generation"]) for r in mortal)
            assert gens == list(range(len(gens)))

    def test_plasticity_index_is_non_negative(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = self._run_tiny(immortal=False, tmpdir=tmpdir)
            for r in rows:
                assert float(r["plasticity_index"]) >= 0.0

    def test_fractions_in_unit_interval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = self._run_tiny(immortal=False, tmpdir=tmpdir)
            for r in rows:
                df = float(r["dreamer_fraction"])
                gf = float(r["guardian_fraction"])
                assert 0.0 <= df <= 1.0, f"dreamer_fraction out of range: {df}"
                assert 0.0 <= gf <= 1.0, f"guardian_fraction out of range: {gf}"
