"""QLearner — tabular Q-learning with ε-greedy exploration.

The Q-learner maps (state_key, action) → estimated cumulative reward.
States are compact tuples produced by ``encode_state()``, making the
Q-table efficient for tabular storage with the sizes encountered here
(at most a few thousand unique states).

Exploration strategy
--------------------
- ε-greedy with geometric decay: probability ε of random action, else greedy.
- UCB bonus (optional): adds a count-based optimism bonus to Q-values during
  greedy selection, encouraging visits to rarely-tried actions.

Learning update
---------------
Standard TD(0) Bellman update::

    Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]

Usage
-----
    learner = QLearner(alpha=0.15, gamma=0.95, epsilon=0.4)
    s = encode_state(vitals_dict, obs)
    action = learner.select_action(s, available_actions)
    # ... execute action, observe reward and next obs ...
    s2 = encode_state(next_vitals, next_obs)
    learner.update(s, action, reward, s2, done=False)
    # At episode end:
    learner.end_episode()
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from thermodynamic_agency.world.grid_world import WorldObservation


# ─────────────────────────────────────────────────────────────────────────────
# State abstraction
# ─────────────────────────────────────────────────────────────────────────────

# Bin thresholds (exclusive upper bounds)
_ENERGY_BINS: list[float] = [30.0, 70.0]   # bins: low(<30), med(30-70), high(>70)
_HEAT_BINS: list[float] = [30.0, 60.0]     # bins: low, med, high
_WASTE_BINS: list[float] = [20.0, 50.0]    # bins: low, med, high


def _bin(value: float, thresholds: list[float]) -> int:
    for i, t in enumerate(thresholds):
        if value < t:
            return i
    return len(thresholds)


def encode_state(
    vitals: dict[str, float],
    obs: WorldObservation,
) -> tuple:
    """Encode metabolic vitals + world observation into a hashable state key.

    The encoding discretises continuous vitals into ordinal bins and
    extracts salient Boolean features from the observation.  The result
    is a short tuple suitable as a dict key for the Q-table.

    State space size: 3 × 3 × 3 × 7 × 2 × 2 × 2 ≈ 1 512 states.

    Parameters
    ----------
    vitals:
        Dict of vital-sign values (keys: energy, heat, waste, integrity,
        stability).  Mirrors ``MetabolicState.to_dict()``.
    obs:
        Current ``WorldObservation`` from the grid world.

    Returns
    -------
    tuple
        (energy_bin, heat_bin, waste_bin, cell_type,
         food_visible, hazard_visible, on_resource)
    """
    energy_bin = _bin(vitals.get("energy", 100.0), _ENERGY_BINS)
    heat_bin = _bin(vitals.get("heat", 0.0), _HEAT_BINS)
    waste_bin = _bin(vitals.get("waste", 0.0), _WASTE_BINS)
    cell = obs.current_cell
    food_visible = bool(obs.nearby_resources)
    hazard_visible = bool(obs.nearby_hazards)
    on_resource = obs.has_resource_here()
    return (energy_bin, heat_bin, waste_bin, cell, food_visible, hazard_visible, on_resource)


# ─────────────────────────────────────────────────────────────────────────────
# Q-learner
# ─────────────────────────────────────────────────────────────────────────────

class QLearner:
    """Tabular Q-learning agent.

    Parameters
    ----------
    alpha:
        Learning rate (0 < α ≤ 1, default 0.15).
    gamma:
        Discount factor (0 < γ ≤ 1, default 0.95).
    epsilon:
        Initial exploration probability for ε-greedy (default 0.40).
    epsilon_min:
        Floor for epsilon after decay (default 0.05).
    epsilon_decay:
        Multiplicative decay applied per ``end_episode()`` call (default 0.97).
    ucb_weight:
        Weight on UCB exploration bonus during greedy selection (default 0.1).
        Set to 0 to disable UCB.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 0.40,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.97,
        ucb_weight: float = 0.10,
        seed: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.ucb_weight = ucb_weight

        self._q: dict[tuple, float] = {}           # (state_key, action) → Q value
        self._visit_counts: dict[tuple, int] = {}  # (state_key, action) → visit count
        self._episode_count: int = 0
        self._total_updates: int = 0
        self._rng = random.Random(seed)

    # ── Q-table access ────────────────────────────────────────────────────────

    def q_value(self, state_key: tuple, action: str) -> float:
        """Return Q(s, a), defaulting to 0.0 for unseen pairs."""
        return self._q.get((state_key, action), 0.0)

    def best_action(
        self,
        state_key: tuple,
        actions: Sequence[str],
    ) -> str:
        """Return the action maximising Q(s,·) with an optional UCB bonus.

        The UCB bonus adds small optimism proportional to 1/√(visit_count+1),
        ensuring infrequently-tried actions remain attractive.
        """
        best_a: str | None = None
        best_v = -math.inf
        total = sum(self._visit_counts.get((state_key, a), 0) for a in actions)
        log_total = math.log(total + 1)

        for a in actions:
            q = self._q.get((state_key, a), 0.0)
            if self.ucb_weight > 0:
                n = self._visit_counts.get((state_key, a), 0)
                ucb = self.ucb_weight * math.sqrt(2.0 * log_total / (n + 1))
                q += ucb
            if q > best_v:
                best_v = q
                best_a = a

        return best_a or self._rng.choice(list(actions))

    def select_action(
        self,
        state_key: tuple,
        actions: Sequence[str],
    ) -> str:
        """ε-greedy action selection.

        With probability ε, chooses uniformly at random (exploration).
        Otherwise, delegates to ``best_action()`` (exploitation + UCB).
        """
        if not actions:
            raise ValueError("No actions available for selection")
        if self._rng.random() < self.epsilon:
            return self._rng.choice(list(actions))
        return self.best_action(state_key, actions)

    # ── TD update ─────────────────────────────────────────────────────────────

    def update(
        self,
        state_key: tuple,
        action: str,
        reward: float,
        next_state_key: tuple,
        done: bool,
        next_actions: Sequence[str] | None = None,
    ) -> float:
        """Apply one TD(0) Bellman update.

        Parameters
        ----------
        state_key, action:
            The state-action pair whose Q-value is being updated.
        reward:
            Observed immediate reward.
        next_state_key:
            Encoded state after the transition.
        done:
            True if the episode terminated (no future reward).
        next_actions:
            Available actions in ``next_state_key``.  Defaults to all
            WorldAction values when None.

        Returns
        -------
        float
            Absolute TD error |target − old_Q|.
        """
        sa_key = (state_key, action)
        self._visit_counts[sa_key] = self._visit_counts.get(sa_key, 0) + 1

        old_q = self._q.get(sa_key, 0.0)

        if done:
            target = reward
        else:
            if next_actions is None:
                from thermodynamic_agency.world.grid_world import WorldAction
                next_actions = [a.value for a in WorldAction]
            max_next_q = max(
                self._q.get((next_state_key, a), 0.0) for a in next_actions
            )
            target = reward + self.gamma * max_next_q

        new_q = old_q + self.alpha * (target - old_q)
        self._q[sa_key] = new_q
        self._total_updates += 1
        return abs(target - old_q)

    # ── Episode management ────────────────────────────────────────────────────

    def end_episode(self) -> None:
        """Decay ε at the end of each training episode."""
        self._episode_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def total_updates(self) -> int:
        return self._total_updates

    @property
    def table_size(self) -> int:
        """Number of (state, action) pairs with stored Q-values."""
        return len(self._q)

    def avg_q_for_state(self, state_key: tuple, actions: Sequence[str]) -> float:
        """Mean Q-value across all actions in a given state."""
        vals = [self._q.get((state_key, a), 0.0) for a in actions]
        return sum(vals) / len(vals) if vals else 0.0

    def policy_snapshot(self) -> dict[tuple, str]:
        """Return the current greedy policy as a dict of state → best_action.

        Only covers states that appear in the Q-table.
        """
        from thermodynamic_agency.world.grid_world import WorldAction
        all_actions = [a.value for a in WorldAction]
        states = {sk for (sk, _) in self._q}
        return {s: self.best_action(s, all_actions) for s in states}
