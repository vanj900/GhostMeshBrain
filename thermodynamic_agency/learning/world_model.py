"""WorldModel — learned tabular transition and reward model.

Maintains frequency counts of observed (state, action) → next_state
transitions and cumulative reward.  Used for:

- Model-based action selection: choose the action with highest expected
  reward (plus exploration bonus for rarely-tried pairs).
- Uncertainty estimation: 1/√(visit_count + 1) encourages exploration of
  novel state-action pairs.
- Planning look-ahead (optional): callers can query predicted next states
  to roll out a short forward plan without interacting with the environment.

The model improves incrementally with every call to ``update()`` — no
batch training required.  Memory grows proportionally to the number of
distinct (state, action) pairs observed; with the current state abstraction
(≈1 500 states × 6 actions = 9 000 pairs) the table stays comfortably small.

Usage
-----
    model = WorldModel()
    model.update(s, "north", reward=0.1, next_state=s2)
    best_a = model.best_action_by_model(s, ["north", "south", "gather"])
    unc = model.uncertainty(s, "north")   # 1.0 = never tried; → 0 with visits
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _TransitionEntry:
    visit_count: int = 0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0
    next_state_counts: dict = field(default_factory=dict)


class WorldModel:
    """Tabular model of environment transitions and expected rewards.

    Learns incrementally from experience with no hyperparameters.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple, _TransitionEntry] = defaultdict(_TransitionEntry)

    # ── Write ─────────────────────────────────────────────────────────────────

    def update(
        self,
        state_key: tuple,
        action: str,
        reward: float,
        next_state_key: tuple,
    ) -> None:
        """Record one observed transition (s, a, r, s')."""
        key = (state_key, action)
        entry = self._entries[key]
        entry.visit_count += 1
        entry.reward_sum += reward
        entry.reward_sq_sum += reward * reward
        nsc = entry.next_state_counts
        nsc[next_state_key] = nsc.get(next_state_key, 0) + 1

    # ── Query ─────────────────────────────────────────────────────────────────

    def predict_next_state(self, state_key: tuple, action: str) -> tuple | None:
        """Return the most frequently observed next state for (s, a).

        Returns None if the pair has never been visited.
        """
        entry = self._entries.get((state_key, action))
        if entry is None or not entry.next_state_counts:
            return None
        return max(entry.next_state_counts, key=lambda k: entry.next_state_counts[k])

    def expected_reward(self, state_key: tuple, action: str) -> float:
        """Empirical mean reward for (s, a).  Returns 0.0 for unseen pairs."""
        entry = self._entries.get((state_key, action))
        if entry is None or entry.visit_count == 0:
            return 0.0
        return entry.reward_sum / entry.visit_count

    def reward_variance(self, state_key: tuple, action: str) -> float:
        """Sample variance of observed rewards for (s, a)."""
        entry = self._entries.get((state_key, action))
        if entry is None or entry.visit_count < 2:
            return 0.0
        n = entry.visit_count
        mean = entry.reward_sum / n
        return entry.reward_sq_sum / n - mean ** 2

    def uncertainty(self, state_key: tuple, action: str) -> float:
        """Exploration uncertainty ∈ (0, 1].

        1.0 for never-visited pairs; approaches 0 as visits accumulate.
        """
        entry = self._entries.get((state_key, action))
        n = entry.visit_count if entry is not None else 0
        return 1.0 / math.sqrt(n + 1)

    def visit_count(self, state_key: tuple, action: str) -> int:
        """Number of times (s, a) has been observed."""
        entry = self._entries.get((state_key, action))
        return entry.visit_count if entry is not None else 0

    def best_action_by_model(
        self,
        state_key: tuple,
        actions: list[str],
        explore_bonus: float = 0.10,
    ) -> str | None:
        """Choose the action maximising expected_reward + explore_bonus × uncertainty.

        When ``explore_bonus`` > 0 the model biases toward less-explored actions,
        providing model-based exploration guidance complementary to the Q-learner.

        Returns None if ``actions`` is empty.
        """
        if not actions:
            return None
        best_a = None
        best_v = -1e9
        for a in actions:
            v = self.expected_reward(state_key, a) + explore_bonus * self.uncertainty(state_key, a)
            if v > best_v:
                best_v = v
                best_a = a
        return best_a

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def model_size(self) -> int:
        """Number of (state, action) pairs recorded."""
        return len(self._entries)

    def coverage(self, total_pairs: int) -> float:
        """Fraction of all possible (s, a) pairs that have been visited at least once."""
        if total_pairs <= 0:
            return 0.0
        visited = sum(1 for e in self._entries.values() if e.visit_count > 0)
        return min(1.0, visited / total_pairs)
