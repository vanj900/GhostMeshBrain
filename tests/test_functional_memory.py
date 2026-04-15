"""Tests for functional memory: WorkingMemory and EpisodicStore."""

from __future__ import annotations

import pytest

from thermodynamic_agency.memory.working_memory import WorkingMemory, WorkingMemorySlot
from thermodynamic_agency.memory.episodic_store import EpisodicStore, EpisodicMemory


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wm_slot(tick: int, state_key: tuple, action: str = "wait",
             reward: float = 0.1, cell_type: str = "empty") -> WorkingMemorySlot:
    return WorkingMemorySlot(
        tick=tick, state_key=state_key, action=action,
        reward=reward, cell_type=cell_type,
    )


def _ep_state(energy_bin: int = 1, heat_bin: int = 0, waste_bin: int = 0,
              cell: str = "empty", food: bool = False,
              hazard: bool = False, on_res: bool = False) -> tuple:
    return (energy_bin, heat_bin, waste_bin, cell, food, hazard, on_res)


# ─────────────────────────────────────────────────────────────────────────────
# WorkingMemory tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkingMemory:
    def test_push_and_len(self):
        wm = WorkingMemory(capacity=5)
        wm.push(_wm_slot(1, _ep_state()))
        assert len(wm) == 1

    def test_capacity_evicts_oldest(self):
        wm = WorkingMemory(capacity=3)
        for i in range(5):
            wm.push(_wm_slot(i, _ep_state()))
        assert len(wm) == 3
        assert wm.recent(3)[0].tick == 2

    def test_recent_returns_most_recent(self):
        wm = WorkingMemory(capacity=10)
        for i in range(7):
            wm.push(_wm_slot(i, _ep_state()))
        recent = wm.recent(3)
        ticks = [s.tick for s in recent]
        assert ticks == [4, 5, 6]

    def test_avg_recent_reward(self):
        wm = WorkingMemory(capacity=10)
        wm.push(_wm_slot(1, _ep_state(), reward=0.0))
        wm.push(_wm_slot(2, _ep_state(), reward=1.0))
        assert wm.avg_recent_reward(n=2) == pytest.approx(0.5)

    def test_avg_recent_reward_empty_returns_zero(self):
        wm = WorkingMemory()
        assert wm.avg_recent_reward() == 0.0

    def test_has_hazard_nearby_recently_true(self):
        wm = WorkingMemory()
        wm.push(_wm_slot(1, _ep_state(), cell_type="radiation"))
        assert wm.has_hazard_nearby_recently(n=5)

    def test_has_hazard_nearby_recently_false_when_no_hazard(self):
        wm = WorkingMemory()
        for i in range(5):
            wm.push(_wm_slot(i, _ep_state(), cell_type="empty"))
        assert not wm.has_hazard_nearby_recently()

    def test_best_action_for_state_returns_highest_reward(self):
        wm = WorkingMemory(capacity=20)
        s = _ep_state()
        wm.push(_wm_slot(1, s, action="north", reward=0.2))
        wm.push(_wm_slot(2, s, action="gather", reward=0.8))
        wm.push(_wm_slot(3, s, action="wait", reward=0.1))
        assert wm.best_action_for_state(s) == "gather"

    def test_best_action_for_state_none_when_unseen(self):
        wm = WorkingMemory()
        s = _ep_state(energy_bin=2)
        assert wm.best_action_for_state(s) is None

    def test_reward_trend_positive_for_improving(self):
        wm = WorkingMemory(capacity=20)
        for i in range(10):
            wm.push(_wm_slot(i, _ep_state(), reward=float(i) * 0.1))
        trend = wm.reward_trend(window=10)
        assert trend > 0

    def test_reward_trend_negative_for_declining(self):
        wm = WorkingMemory(capacity=20)
        for i in range(10):
            wm.push(_wm_slot(i, _ep_state(), reward=float(9 - i) * 0.1))
        trend = wm.reward_trend(window=10)
        assert trend < 0

    def test_reward_trend_zero_for_flat(self):
        wm = WorkingMemory(capacity=20)
        for i in range(5):
            wm.push(_wm_slot(i, _ep_state(), reward=0.5))
        trend = wm.reward_trend(window=5)
        assert trend == pytest.approx(0.0)

    def test_last_action_is_most_recent(self):
        wm = WorkingMemory()
        wm.push(_wm_slot(1, _ep_state(), action="north"))
        wm.push(_wm_slot(2, _ep_state(), action="gather"))
        assert wm.last_action() == "gather"

    def test_last_action_none_when_empty(self):
        wm = WorkingMemory()
        assert wm.last_action() is None


# ─────────────────────────────────────────────────────────────────────────────
# EpisodicStore tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEpisodicStore:
    def test_record_and_len(self):
        store = EpisodicStore()
        s = _ep_state()
        store.record(tick=1, state_key=s, action="gather", reward=0.5, next_state_key=s)
        assert len(store) == 1

    def test_total_recorded_counts_all(self):
        store = EpisodicStore(maxlen=2)
        s = _ep_state()
        for i in range(5):
            store.record(tick=i, state_key=s, action="wait", reward=0.1, next_state_key=s)
        assert store.total_recorded == 5
        assert len(store) == 2   # capped at maxlen

    def test_recall_similar_exact_match(self):
        store = EpisodicStore()
        s = _ep_state(energy_bin=0, cell="food")
        s_other = _ep_state(energy_bin=2, cell="empty")
        store.record(1, s, "gather", 0.9, s)
        store.record(2, s_other, "north", 0.1, s_other)
        recalled = store.recall_similar(s, n=1)
        assert len(recalled) == 1
        assert recalled[0].state_key == s

    def test_recall_similar_returns_n_closest(self):
        store = EpisodicStore()
        s_target = _ep_state(energy_bin=0, heat_bin=0, waste_bin=0)
        # Identical state
        store.record(1, s_target, "gather", 0.9, s_target)
        # One field different
        s_close = _ep_state(energy_bin=1, heat_bin=0, waste_bin=0)
        store.record(2, s_close, "north", 0.3, s_close)
        # Two fields different
        s_far = _ep_state(energy_bin=2, heat_bin=1, waste_bin=0)
        store.record(3, s_far, "south", 0.1, s_far)
        recalled = store.recall_similar(s_target, n=2)
        # Most similar should come first
        assert recalled[0].state_key == s_target
        assert recalled[1].state_key == s_close

    def test_best_action_for_state_highest_mean_reward(self):
        store = EpisodicStore()
        s = _ep_state()
        store.record(1, s, "gather", reward=0.8, next_state_key=s)
        store.record(2, s, "north", reward=0.2, next_state_key=s)
        store.record(3, s, "gather", reward=0.9, next_state_key=s)
        best = store.best_action_for_state(s, k_similar=5)
        assert best == "gather"

    def test_best_action_none_for_empty_store(self):
        store = EpisodicStore()
        s = _ep_state()
        assert store.best_action_for_state(s) is None

    def test_avg_reward_for_action(self):
        store = EpisodicStore()
        s = _ep_state()
        store.record(1, s, "gather", reward=0.6, next_state_key=s)
        store.record(2, s, "gather", reward=0.4, next_state_key=s)
        store.record(3, s, "north", reward=0.1, next_state_key=s)
        avg = store.avg_reward_for_action(s, "gather", k_similar=10)
        assert avg == pytest.approx(0.5)

    def test_risky_states_identifies_low_reward_states(self):
        store = EpisodicStore()
        s_bad = _ep_state(energy_bin=0, cell="radiation")
        s_good = _ep_state(energy_bin=2, cell="food")
        for _ in range(5):
            store.record(1, s_bad, "north", reward=-0.5, next_state_key=s_bad)
        for _ in range(5):
            store.record(2, s_good, "gather", reward=0.7, next_state_key=s_good)
        risky = store.risky_states(threshold=-0.2)
        assert s_bad in risky
        assert s_good not in risky

    def test_avg_reward_recent(self):
        store = EpisodicStore()
        s = _ep_state()
        for _ in range(5):
            store.record(1, s, "wait", reward=0.2, next_state_key=s)
        assert store.avg_reward_recent(n=5) == pytest.approx(0.2)

    def test_similarity_exact_match_is_one(self):
        s = _ep_state()
        sim = EpisodicStore._similarity(s, s)
        assert sim == pytest.approx(1.0)

    def test_similarity_totally_different_is_low(self):
        a = (0, 0, 0, "empty", False, False, False)
        b = (2, 2, 2, "food", True, True, True)
        sim = EpisodicStore._similarity(a, b)
        assert sim == pytest.approx(0.0)

    def test_memory_influences_decisions(self):
        """Memory retrieval should surface better actions for known states."""
        store = EpisodicStore()
        s = _ep_state(energy_bin=0, cell="food")
        # Record many positive experiences with "gather"
        for _ in range(20):
            store.record(0, s, "gather", reward=0.8, next_state_key=s)
        # Record few negative with "north"
        for _ in range(5):
            store.record(0, s, "north", reward=-0.2, next_state_key=s)
        best = store.best_action_for_state(s, k_similar=30)
        assert best == "gather", "Episodic memory should recommend gather in food state"
