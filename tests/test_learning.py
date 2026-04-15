"""Tests for the learning subsystems: reward, Q-learner, world model, experience buffer."""

from __future__ import annotations

import math

import pytest

from thermodynamic_agency.world.grid_world import GridWorld, WorldAction, CellType
from thermodynamic_agency.learning.reward import compute_reward, RewardSignal
from thermodynamic_agency.learning.experience_buffer import Experience, ExperienceBuffer
from thermodynamic_agency.learning.q_learner import QLearner, encode_state
from thermodynamic_agency.learning.world_model import WorldModel


# ─────────────────────────────────────────────────────────────────────────────
# Reward signal tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardSignal:
    _BASE_VITALS = {
        "energy": 70.0, "heat": 20.0, "waste": 10.0,
        "integrity": 90.0, "stability": 90.0,
    }

    def test_survival_bonus_when_alive(self):
        sig = compute_reward(self._BASE_VITALS, self._BASE_VITALS, gathered=False)
        assert sig.survival == pytest.approx(0.1)
        assert sig.total > 0

    def test_death_penalty(self):
        sig = compute_reward(self._BASE_VITALS, self._BASE_VITALS, gathered=False, alive=False)
        assert sig.survival == pytest.approx(-10.0)
        assert sig.total == pytest.approx(-10.0)

    def test_gathering_food_gives_positive_resource_reward(self):
        before = dict(self._BASE_VITALS)
        after = dict(before)
        after["energy"] = before["energy"] + 15.0   # food gathered
        sig = compute_reward(before, after, gathered=True)
        assert sig.resource > 0.0

    def test_gathering_water_reduces_heat_reward(self):
        before = dict(self._BASE_VITALS)
        after = dict(before)
        after["heat"] = before["heat"] - 12.0   # water gathered
        sig = compute_reward(before, after, gathered=True)
        assert sig.resource > 0.0

    def test_hazard_penalty_for_high_heat_gain(self):
        before = dict(self._BASE_VITALS)
        after = dict(before)
        after["heat"] = before["heat"] + 10.0   # radiation hit
        sig = compute_reward(before, after, gathered=False)
        assert sig.hazard < 0.0

    def test_hazard_penalty_for_high_waste_gain(self):
        before = dict(self._BASE_VITALS)
        after = dict(before)
        after["waste"] = before["waste"] + 15.0   # toxin hit
        sig = compute_reward(before, after, gathered=False)
        assert sig.hazard < 0.0

    def test_small_heat_increase_not_penalised(self):
        before = dict(self._BASE_VITALS)
        after = dict(before)
        after["heat"] = before["heat"] + 2.0   # within passive-decay range
        sig = compute_reward(before, after, gathered=False)
        assert sig.hazard == pytest.approx(0.0)

    def test_total_equals_sum_of_components(self):
        before = dict(self._BASE_VITALS)
        after = dict(before)
        after["energy"] = before["energy"] + 10.0
        sig = compute_reward(before, after, gathered=True)
        assert sig.total == pytest.approx(
            sig.survival + sig.resource + sig.hazard + sig.internal
        )


# ─────────────────────────────────────────────────────────────────────────────
# Experience buffer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExperienceBuffer:
    def _make_exp(self, tick: int, reward: float = 0.1) -> Experience:
        return Experience(
            tick=tick,
            state_key=(0, 0, 0, "empty", False, False, False),
            action="wait",
            reward=reward,
            next_state_key=(0, 0, 0, "empty", False, False, False),
        )

    def test_push_and_len(self):
        buf = ExperienceBuffer()
        buf.push(self._make_exp(1))
        assert len(buf) == 1

    def test_maxlen_evicts_oldest(self):
        buf = ExperienceBuffer(maxlen=3)
        for i in range(5):
            buf.push(self._make_exp(i))
        assert len(buf) == 3
        assert buf.recent(3)[0].tick == 2   # oldest surviving

    def test_sample_returns_correct_count(self):
        buf = ExperienceBuffer(seed=0)
        for i in range(10):
            buf.push(self._make_exp(i))
        batch = buf.sample(5)
        assert len(batch) == 5

    def test_sample_capped_at_buffer_size(self):
        buf = ExperienceBuffer(seed=0)
        for i in range(3):
            buf.push(self._make_exp(i))
        batch = buf.sample(10)
        assert len(batch) == 3

    def test_recent_order(self):
        buf = ExperienceBuffer()
        for i in range(5):
            buf.push(self._make_exp(i))
        recent = buf.recent(3)
        assert [e.tick for e in recent] == [2, 3, 4]

    def test_avg_reward(self):
        buf = ExperienceBuffer()
        buf.push(self._make_exp(1, reward=0.0))
        buf.push(self._make_exp(2, reward=1.0))
        assert buf.avg_reward() == pytest.approx(0.5)

    def test_empty_sample_returns_empty(self):
        buf = ExperienceBuffer()
        assert buf.sample(5) == []


# ─────────────────────────────────────────────────────────────────────────────
# Q-learner tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_state(energy_bin=1, heat_bin=0, waste_bin=0,
                cell="empty", food=False, hazard=False, on_res=False) -> tuple:
    return (energy_bin, heat_bin, waste_bin, cell, food, hazard, on_res)


class TestQLearner:
    def test_initial_q_values_zero(self):
        q = QLearner()
        s = _make_state()
        assert q.q_value(s, "north") == 0.0
        assert q.q_value(s, "wait") == 0.0

    def test_update_increases_q_for_positive_reward(self):
        q = QLearner(alpha=0.5, gamma=0.0)
        s = _make_state()
        s2 = _make_state(energy_bin=2)
        q.update(s, "gather", reward=1.0, next_state_key=s2, done=True)
        assert q.q_value(s, "gather") == pytest.approx(0.5)

    def test_update_decreases_q_for_negative_reward(self):
        q = QLearner(alpha=1.0, gamma=0.0)
        s = _make_state()
        s2 = _make_state()
        q.update(s, "north", reward=-1.0, next_state_key=s2, done=True)
        assert q.q_value(s, "north") == pytest.approx(-1.0)

    def test_bellman_backup_uses_max_next_q(self):
        q = QLearner(alpha=1.0, gamma=0.9)
        s = _make_state()
        s2 = _make_state(energy_bin=2)
        # Prime next state Q value
        q.update(s2, "gather", reward=1.0, next_state_key=s2, done=True)
        # Update from s → s2
        q.update(s, "north", reward=0.0, next_state_key=s2, done=False,
                 next_actions=["gather", "north"])
        # Expected: 0 + 1.0 * (0 + 0.9 * 1.0) = 0.9
        assert q.q_value(s, "north") == pytest.approx(0.9)

    def test_best_action_returns_highest_q(self):
        q = QLearner(ucb_weight=0.0)
        s = _make_state()
        s2 = _make_state()
        q.update(s, "gather", reward=5.0, next_state_key=s2, done=True)
        q.update(s, "north", reward=1.0, next_state_key=s2, done=True)
        assert q.best_action(s, ["gather", "north"]) == "gather"

    def test_epsilon_greedy_explores(self):
        """With epsilon=1.0 all selections should be random."""
        q = QLearner(epsilon=1.0, seed=0)
        s = _make_state()
        # Prime one action very high
        s2 = _make_state()
        q.update(s, "gather", reward=100.0, next_state_key=s2, done=True)
        actions = ["gather", "north", "south", "east", "west", "wait"]
        chosen = {q.select_action(s, actions) for _ in range(50)}
        # Exploration should have picked non-"gather" actions too
        assert len(chosen) > 1

    def test_epsilon_decays_on_end_episode(self):
        q = QLearner(epsilon=0.4, epsilon_decay=0.5)
        q.end_episode()
        assert q.epsilon == pytest.approx(0.2)
        q.end_episode()
        assert q.epsilon == pytest.approx(0.1)

    def test_epsilon_floored_at_epsilon_min(self):
        q = QLearner(epsilon=0.1, epsilon_min=0.05, epsilon_decay=0.1)
        for _ in range(10):
            q.end_episode()
        assert q.epsilon == pytest.approx(0.05)

    def test_table_size_grows_with_updates(self):
        q = QLearner()
        s = _make_state()
        s2 = _make_state(energy_bin=2)
        assert q.table_size == 0
        q.update(s, "north", 0.1, s2, done=False)
        assert q.table_size == 1

    def test_encode_state_deterministic(self):
        world = GridWorld(seed=0)
        obs = world.get_observation()
        vitals = {"energy": 80.0, "heat": 20.0, "waste": 5.0,
                  "integrity": 90.0, "stability": 90.0}
        s1 = encode_state(vitals, obs)
        s2 = encode_state(vitals, obs)
        assert s1 == s2

    def test_encode_state_different_for_different_vitals(self):
        world = GridWorld(seed=0)
        obs = world.get_observation()
        vitals_low_energy = {"energy": 10.0, "heat": 20.0, "waste": 5.0,
                             "integrity": 90.0, "stability": 90.0}
        vitals_high_energy = {"energy": 90.0, "heat": 20.0, "waste": 5.0,
                              "integrity": 90.0, "stability": 90.0}
        s_low = encode_state(vitals_low_energy, obs)
        s_high = encode_state(vitals_high_energy, obs)
        assert s_low != s_high

    def test_policy_snapshot_covers_visited_states(self):
        q = QLearner(ucb_weight=0.0, seed=0)
        s = _make_state()
        s2 = _make_state(energy_bin=2)
        q.update(s, "gather", reward=1.0, next_state_key=s2, done=True)
        snap = q.policy_snapshot()
        assert s in snap
        assert snap[s] == "gather"


# ─────────────────────────────────────────────────────────────────────────────
# World model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWorldModel:
    def test_expected_reward_zero_for_unseen(self):
        m = WorldModel()
        s = _make_state()
        assert m.expected_reward(s, "north") == 0.0

    def test_update_and_expected_reward(self):
        m = WorldModel()
        s = _make_state()
        s2 = _make_state(energy_bin=2)
        m.update(s, "gather", reward=0.6, next_state_key=s2)
        assert m.expected_reward(s, "gather") == pytest.approx(0.6)

    def test_expected_reward_averages_multiple_obs(self):
        m = WorldModel()
        s = _make_state()
        s2 = _make_state(energy_bin=2)
        m.update(s, "north", reward=0.2, next_state_key=s2)
        m.update(s, "north", reward=0.4, next_state_key=s2)
        assert m.expected_reward(s, "north") == pytest.approx(0.3)

    def test_uncertainty_decreases_with_visits(self):
        m = WorldModel()
        s = _make_state()
        s2 = _make_state()
        unc_0 = m.uncertainty(s, "north")   # 1.0 (never visited)
        m.update(s, "north", 0.1, s2)
        unc_1 = m.uncertainty(s, "north")
        m.update(s, "north", 0.1, s2)
        unc_2 = m.uncertainty(s, "north")
        assert unc_0 > unc_1 > unc_2

    def test_uncertainty_is_one_for_unseen(self):
        m = WorldModel()
        s = _make_state()
        assert m.uncertainty(s, "gather") == pytest.approx(1.0)

    def test_predict_next_state_mode(self):
        m = WorldModel()
        s = _make_state()
        s2 = _make_state(energy_bin=2)
        s3 = _make_state(energy_bin=0)
        m.update(s, "gather", 0.6, s2)
        m.update(s, "gather", 0.6, s2)
        m.update(s, "gather", 0.6, s3)
        predicted = m.predict_next_state(s, "gather")
        assert predicted == s2   # s2 is the modal next state

    def test_predict_next_state_none_for_unseen(self):
        m = WorldModel()
        s = _make_state()
        assert m.predict_next_state(s, "north") is None

    def test_best_action_by_model(self):
        m = WorldModel()
        s = _make_state()
        s2 = _make_state()
        m.update(s, "gather", reward=1.0, next_state_key=s2)
        m.update(s, "north", reward=0.1, next_state_key=s2)
        best = m.best_action_by_model(s, ["gather", "north"], explore_bonus=0.0)
        assert best == "gather"

    def test_model_size_grows_with_updates(self):
        m = WorldModel()
        s = _make_state()
        s2 = _make_state()
        assert m.model_size == 0
        m.update(s, "north", 0.1, s2)
        assert m.model_size == 1
        m.update(s, "south", 0.1, s2)
        assert m.model_size == 2
