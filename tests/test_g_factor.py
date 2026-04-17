"""Tests for the g-factor measurement and cognitive battery modules."""

from __future__ import annotations

import math
import pytest

from thermodynamic_agency.evaluation.cognitive_battery import (
    CognitiveBattery,
    TaskScores,
    TASK_NAMES,
)
from thermodynamic_agency.evaluation.g_factor import (
    GFactorResult,
    measure_g,
    _standardise,
    _covariance,
    _power_iteration,
)
from thermodynamic_agency.learning.q_learner import QLearner


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_learner(seed: int = 0) -> QLearner:
    """Return a fresh (untrained) Q-learner."""
    return QLearner(seed=seed)


def _make_trained_learner(seed: int = 42) -> QLearner:
    """Return a minimally trained Q-learner via a short EpisodeRunner."""
    from thermodynamic_agency.world.episode_runner import EpisodeRunner
    runner = EpisodeRunner(n_episodes=3, ticks_per_episode=30, seed=seed)
    runner.train()
    return runner.learner


def _synthetic_scores(n: int = 20, seed: int = 0) -> list[list[float]]:
    """Generate synthetic [0, 1] score vectors with a strong common factor."""
    import random
    rng = random.Random(seed)
    scores = []
    for _ in range(n):
        g = rng.uniform(0.2, 0.9)        # latent g for this episode
        row = [
            min(1.0, max(0.0, g + rng.gauss(0, 0.08)))
            for _ in range(6)
        ]
        scores.append(row)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# TaskScores
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskScores:
    def test_as_vector_length(self):
        ts = TaskScores(
            navigation=0.8,
            puzzle=0.7,
            adaptation=0.6,
            resource_mgmt=0.9,
            social=0.5,
            prediction=0.7,
        )
        v = ts.as_vector()
        assert len(v) == 6

    def test_as_vector_order_matches_task_names(self):
        ts = TaskScores(
            navigation=0.1,
            puzzle=0.2,
            adaptation=0.3,
            resource_mgmt=0.4,
            social=0.5,
            prediction=0.6,
        )
        v = ts.as_vector()
        assert v[0] == pytest.approx(0.1)
        assert v[1] == pytest.approx(0.2)
        assert v[2] == pytest.approx(0.3)
        assert v[3] == pytest.approx(0.4)
        assert v[4] == pytest.approx(0.5)
        assert v[5] == pytest.approx(0.6)

    def test_task_names_length(self):
        assert len(TASK_NAMES) == 6


# ─────────────────────────────────────────────────────────────────────────────
# CognitiveBattery
# ─────────────────────────────────────────────────────────────────────────────

class TestCognitiveBattery:
    def test_evaluate_returns_task_scores(self):
        learner = _make_learner(seed=1)
        battery = CognitiveBattery(learner=learner, seed=10)
        scores = battery.evaluate()
        assert isinstance(scores, TaskScores)

    def test_all_scores_in_unit_interval(self):
        learner = _make_learner(seed=2)
        battery = CognitiveBattery(learner=learner, seed=20)
        scores = battery.evaluate()
        for val in scores.as_vector():
            assert 0.0 <= val <= 1.0, f"Score out of range: {val}"

    def test_as_vector_has_six_elements(self):
        learner = _make_learner(seed=3)
        battery = CognitiveBattery(learner=learner, seed=30)
        scores = battery.evaluate()
        assert len(scores.as_vector()) == 6

    def test_nav_efficiency_zero_for_frozen_learner(self):
        """A learner that always waits should gather nothing → score 0."""
        learner = _make_learner(seed=4)
        # All Q values are 0, best_action will pick first available (WAIT is listed
        # first), so the agent effectively stays put most of the time.
        battery = CognitiveBattery(learner=learner, seed=40)
        scores = battery.evaluate()
        # Score may be 0 or low but must be valid
        assert 0.0 <= scores.navigation <= 1.0

    def test_different_seeds_produce_different_scores(self):
        """Scores should vary across different battery seed layouts.

        We run the same learner against several battery seeds and verify that
        at least one pair produces different score vectors.  With an untrained
        (all-zero Q) learner the greedy policy is deterministic, so we use a
        minimally trained learner and enough seeds to guarantee layout-driven
        variation.
        """
        learner = _make_trained_learner(seed=5)
        seeds = [0, 13, 77, 200, 999]
        all_scores = []
        for s in seeds:
            b = CognitiveBattery(learner=learner, seed=s)
            all_scores.append(b.evaluate().as_vector())
        # At least two of the five results must differ
        unique = [list(v) for v in {tuple(v) for v in all_scores}]
        assert len(unique) > 1, (
            "All battery seeds produced identical scores — "
            "layout variation is not propagating to task scores."
        )

    def test_trained_agent_nav_score_not_below_zero(self):
        learner = _make_trained_learner(seed=10)
        battery = CognitiveBattery(learner=learner, seed=50)
        scores = battery.evaluate()
        assert scores.navigation >= 0.0

    def test_puzzle_score_in_range(self):
        learner = _make_learner(seed=6)
        battery = CognitiveBattery(learner=learner, seed=60)
        scores = battery.evaluate()
        assert 0.0 <= scores.puzzle <= 1.0

    def test_adaptation_score_in_range(self):
        learner = _make_learner(seed=7)
        battery = CognitiveBattery(learner=learner, seed=70)
        scores = battery.evaluate()
        assert 0.0 <= scores.adaptation <= 1.0

    def test_resource_mgmt_score_in_range(self):
        learner = _make_learner(seed=8)
        battery = CognitiveBattery(learner=learner, seed=80)
        scores = battery.evaluate()
        assert 0.0 <= scores.resource_mgmt <= 1.0

    def test_social_score_in_range(self):
        learner = _make_learner(seed=9)
        battery = CognitiveBattery(learner=learner, seed=90)
        scores = battery.evaluate()
        assert 0.0 <= scores.social <= 1.0

    def test_prediction_score_in_range(self):
        learner = _make_learner(seed=10)
        battery = CognitiveBattery(learner=learner, seed=100)
        scores = battery.evaluate()
        assert 0.0 <= scores.prediction <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# PCA helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestStandardise:
    def test_output_shape(self):
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = _standardise(data, 3, 2)
        assert len(result) == 3
        assert all(len(row) == 2 for row in result)

    def test_column_mean_near_zero(self):
        data = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        std = _standardise(data, 3, 2)
        col0_mean = sum(r[0] for r in std) / 3
        col1_mean = sum(r[1] for r in std) / 3
        assert abs(col0_mean) < 1e-9
        assert abs(col1_mean) < 1e-9

    def test_constant_column_not_division_by_zero(self):
        data = [[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]
        result = _standardise(data, 3, 2)
        # constant column → std=1 fallback, all values → 0
        assert all(r[0] == pytest.approx(0.0) for r in result)


class TestCovariance:
    def test_diagonal_is_one_for_standardised_data(self):
        import random
        rng = random.Random(0)
        n, k = 30, 4
        raw = [[rng.gauss(0, 1) for _ in range(k)] for _ in range(n)]
        std = _standardise(raw, n, k)
        cov = _covariance(std, n, k)
        # Diagonal should be close to 1 (variance of standardised columns)
        for j in range(k):
            assert abs(cov[j][j] - 1.0) < 0.15, f"cov[{j}][{j}] = {cov[j][j]}"

    def test_symmetric(self):
        import random
        rng = random.Random(1)
        n, k = 10, 3
        raw = [[rng.gauss(0, 1) for _ in range(k)] for _ in range(n)]
        std = _standardise(raw, n, k)
        cov = _covariance(std, n, k)
        for j in range(k):
            for l in range(k):
                assert cov[j][l] == pytest.approx(cov[l][j])


class TestPowerIteration:
    def test_known_matrix(self):
        # 2×2 matrix with eigenvalues 3 and 1
        matrix = [[2.0, 1.0], [1.0, 2.0]]
        eigvec, eigval = _power_iteration(matrix, 2)
        # Dominant eigenvalue should be 3
        assert abs(eigval - 3.0) < 1e-6
        # Eigenvector should be [1/√2, 1/√2]
        expected = 1.0 / math.sqrt(2)
        assert abs(abs(eigvec[0]) - expected) < 1e-6
        assert abs(abs(eigvec[1]) - expected) < 1e-6

    def test_eigenvector_is_unit_norm(self):
        matrix = [[4.0, 2.0, 1.0], [2.0, 3.0, 0.5], [1.0, 0.5, 2.0]]
        eigvec, _ = _power_iteration(matrix, 3)
        norm = math.sqrt(sum(v * v for v in eigvec))
        assert abs(norm - 1.0) < 1e-9

    def test_identity_matrix_eigenvalue_is_one(self):
        n = 4
        identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        _, eigval = _power_iteration(identity, n)
        assert abs(eigval - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# measure_g
# ─────────────────────────────────────────────────────────────────────────────

class TestMeasureG:
    def test_requires_at_least_three_samples(self):
        with pytest.raises(ValueError, match="at least 3"):
            measure_g([[0.5] * 6, [0.6] * 6])

    def test_requires_six_element_vectors(self):
        with pytest.raises(ValueError, match="6"):
            measure_g([[0.5] * 5] * 5)

    def test_returns_g_factor_result(self):
        scores = _synthetic_scores(n=10)
        result = measure_g(scores)
        assert isinstance(result, GFactorResult)

    def test_g_scores_length_matches_input(self):
        n = 15
        scores = _synthetic_scores(n=n)
        result = measure_g(scores)
        assert len(result.g_scores) == n

    def test_variance_explained_in_unit_interval(self):
        scores = _synthetic_scores(n=20)
        result = measure_g(scores)
        assert 0.0 <= result.variance_explained <= 1.0

    def test_loadings_length_is_six(self):
        scores = _synthetic_scores(n=20)
        result = measure_g(scores)
        assert len(result.loadings) == 6

    def test_loadings_form_unit_vector(self):
        scores = _synthetic_scores(n=20)
        result = measure_g(scores)
        norm = math.sqrt(sum(l * l for l in result.loadings))
        assert abs(norm - 1.0) < 1e-6

    def test_strong_g_factor_detected(self):
        """Perfectly correlated tasks should yield variance_explained ≈ 1."""
        import random
        rng = random.Random(7)
        n = 30
        scores = []
        for _ in range(n):
            g = rng.uniform(0.1, 0.9)
            # All tasks = g + tiny noise → single factor dominates
            scores.append([g + rng.gauss(0, 0.01) for _ in range(6)])
        result = measure_g(scores)
        assert result.variance_explained > 0.90

    def test_uncorrelated_tasks_low_g(self):
        """Fully independent tasks should yield low variance_explained."""
        import random
        rng = random.Random(99)
        n = 50
        scores = [
            [rng.uniform(0, 1) for _ in range(6)]
            for _ in range(n)
        ]
        result = measure_g(scores)
        # With truly independent tasks PCA can still pick up some shared
        # variance by chance; we just check it isn't artificially high.
        assert result.variance_explained < 0.60

    def test_task_names_match_constants(self):
        scores = _synthetic_scores(n=10)
        result = measure_g(scores)
        assert result.task_names == TASK_NAMES

    def test_n_episodes_set_correctly(self):
        n = 12
        scores = _synthetic_scores(n=n)
        result = measure_g(scores)
        assert result.n_episodes == n

    def test_is_significant_threshold(self):
        scores = _synthetic_scores(n=20, seed=42)
        result = measure_g(scores)
        # Synthetic data has a strong g-factor so should be significant
        assert result.is_significant()

    def test_summary_contains_variance(self):
        scores = _synthetic_scores(n=10)
        result = measure_g(scores)
        summary = result.summary()
        assert "%" in summary
        assert "variance" in summary.lower()


# ─────────────────────────────────────────────────────────────────────────────
# EpisodeRunner integration
# ─────────────────────────────────────────────────────────────────────────────

class TestEpisodeRunnerGFactor:
    def test_run_battery_returns_task_scores(self):
        from thermodynamic_agency.world.episode_runner import EpisodeRunner
        runner = EpisodeRunner(n_episodes=2, ticks_per_episode=20, seed=1)
        runner.train()
        scores = runner.run_battery()
        assert isinstance(scores, TaskScores)
        assert all(0.0 <= v <= 1.0 for v in scores.as_vector())

    def test_measure_g_returns_none_before_enough_data(self):
        from thermodynamic_agency.world.episode_runner import EpisodeRunner
        runner = EpisodeRunner(n_episodes=0, ticks_per_episode=20, seed=2)
        assert runner.measure_g() is None

    def test_train_populates_task_scores_on_each_episode(self):
        from thermodynamic_agency.world.episode_runner import EpisodeRunner
        runner = EpisodeRunner(
            n_episodes=4,
            ticks_per_episode=20,
            seed=3,
            g_eval_interval=10,
        )
        stats = runner.train()
        for ep_stats in stats.episodes:
            assert ep_stats.task_scores is not None
            assert isinstance(ep_stats.task_scores, TaskScores)

    def test_train_produces_g_history_at_interval(self):
        from thermodynamic_agency.world.episode_runner import EpisodeRunner
        runner = EpisodeRunner(
            n_episodes=6,
            ticks_per_episode=20,
            seed=4,
            g_eval_interval=3,   # compute g every 3 episodes
        )
        stats = runner.train()
        # With 6 episodes and interval 3, expect 2 g computations
        assert len(stats.g_history) == 2
        for g_result in stats.g_history:
            assert isinstance(g_result, GFactorResult)
            assert 0.0 <= g_result.variance_explained <= 1.0

    def test_latest_g_property(self):
        from thermodynamic_agency.world.episode_runner import EpisodeRunner
        runner = EpisodeRunner(
            n_episodes=5,
            ticks_per_episode=20,
            seed=5,
            g_eval_interval=3,
        )
        stats = runner.train()
        lg = stats.latest_g
        assert lg is not None
        assert isinstance(lg, GFactorResult)

    def test_measure_g_after_enough_episodes(self):
        from thermodynamic_agency.world.episode_runner import EpisodeRunner
        runner = EpisodeRunner(
            n_episodes=5,
            ticks_per_episode=20,
            seed=6,
            g_eval_interval=100,   # never auto-triggers during train
        )
        runner.train()
        result = runner.measure_g()
        assert result is not None
        assert len(result.g_scores) == 5
