"""EpisodicStore — long-term episodic memory with similarity-based retrieval.

Stores past (state, action, reward, outcome) tuples and supports recall of
experiences similar to a given query state.  Retrieved memories influence
decision-making by providing action recommendations and risk indicators
drawn from the organism's own history.

Similarity metric
-----------------
Two state keys are compared by counting matching components (Hamming
similarity).  This is fast, deterministic, and tolerates the mixed
discrete types in the Q-learner's state encoding.

Retrieval methods
-----------------
recall_similar(state_key, n)
    Return the n most similar stored episodes.
best_action_for_state(state_key)
    Action with the highest average reward in similar episodes.
avg_reward_for_action(state_key, action)
    Expected reward for a (state-like, action) pair from memory.
risky_states(threshold)
    Set of state keys where historical reward is below threshold
    (used for risk avoidance in planning).

Usage
-----
    store = EpisodicStore(maxlen=5_000)
    mem = store.record(tick=1, state_key=s, action="gather",
                       reward=0.6, next_state_key=s2)
    similar = store.recall_similar(s, n=5)
    best_a  = store.best_action_for_state(s)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class EpisodicMemory:
    """One stored episode (single time-step experience)."""

    id: int
    tick: int
    state_key: tuple
    action: str
    reward: float
    next_state_key: tuple
    outcome_vitals: dict = field(default_factory=dict)


class EpisodicStore:
    """Bounded episodic memory with similarity-based retrieval.

    Parameters
    ----------
    maxlen:
        Maximum episodes retained; oldest are evicted when full (default 5 000).
    """

    def __init__(self, maxlen: int = 5_000) -> None:
        self._store: deque[EpisodicMemory] = deque(maxlen=maxlen)
        self._id_counter: int = 0

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(
        self,
        tick: int,
        state_key: tuple,
        action: str,
        reward: float,
        next_state_key: tuple,
        outcome_vitals: dict | None = None,
    ) -> EpisodicMemory:
        """Store one experience and return the created EpisodicMemory."""
        mem = EpisodicMemory(
            id=self._id_counter,
            tick=tick,
            state_key=state_key,
            action=action,
            reward=reward,
            next_state_key=next_state_key,
            outcome_vitals=outcome_vitals or {},
        )
        self._store.append(mem)
        self._id_counter += 1
        return mem

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def recall_similar(
        self, state_key: tuple, n: int = 5
    ) -> list[EpisodicMemory]:
        """Return the *n* most similar stored episodes to *state_key*.

        Similarity is the fraction of matching components in the state key
        tuple (Hamming similarity ∈ [0, 1]).  Ties are broken by recency
        (more recent episodes first among equal-similarity candidates).
        """
        if not self._store:
            return []
        scored = [
            (self._similarity(state_key, mem.state_key), mem)
            for mem in self._store
        ]
        # Sort descending by similarity (stable sort preserves insertion order
        # for equal-similarity items, which are already newest-last in the deque)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:n]]

    def best_action_for_state(
        self, state_key: tuple, k_similar: int = 10
    ) -> str | None:
        """Return the action with the highest mean reward in similar states.

        Queries the top-*k* similar episodes and averages rewards per action.
        Returns None if the store is empty.
        """
        similar = self.recall_similar(state_key, k_similar)
        if not similar:
            return None
        action_rewards: dict[str, list[float]] = {}
        for mem in similar:
            action_rewards.setdefault(mem.action, []).append(mem.reward)
        return max(
            action_rewards,
            key=lambda a: sum(action_rewards[a]) / len(action_rewards[a]),
        )

    def avg_reward_for_action(
        self, state_key: tuple, action: str, k_similar: int = 20
    ) -> float:
        """Mean historical reward for *action* in states similar to *state_key*."""
        similar = self.recall_similar(state_key, k_similar)
        rewards = [m.reward for m in similar if m.action == action]
        return sum(rewards) / len(rewards) if rewards else 0.0

    def risky_states(
        self, threshold: float = -0.20, recent_n: int = 200
    ) -> set[tuple]:
        """Return state keys whose recent mean reward is below *threshold*.

        Only considers the last *recent_n* episodes for efficiency.  The
        caller can use this set to penalise or avoid risky states in planning.
        """
        recent = list(self._store)[-recent_n:]
        state_rewards: dict[tuple, list[float]] = {}
        for mem in recent:
            state_rewards.setdefault(mem.state_key, []).append(mem.reward)
        return {
            sk
            for sk, rewards in state_rewards.items()
            if sum(rewards) / len(rewards) < threshold
        }

    # ── Introspection ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._store)

    @property
    def total_recorded(self) -> int:
        """Total number of episodes ever recorded (including evicted ones)."""
        return self._id_counter

    def avg_reward_recent(self, n: int = 100) -> float:
        """Mean reward over the last *n* stored episodes."""
        recent = list(self._store)[-n:]
        if not recent:
            return 0.0
        return sum(m.reward for m in recent) / len(recent)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _similarity(a: tuple, b: tuple) -> float:
        """Hamming similarity — fraction of matching positions."""
        length = min(len(a), len(b))
        if length == 0:
            return 0.0
        matches = sum(1 for x, y in zip(a, b) if x == y)
        return matches / length
