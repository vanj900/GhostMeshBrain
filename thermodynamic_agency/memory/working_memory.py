"""WorkingMemory — short-term recency queue for immediate decision support.

Working memory holds the most recent N experiences and exposes aggregate
signals that the planning layer can query before selecting an action:

- ``avg_recent_reward()`` — is the agent doing well right now?
- ``has_hazard_nearby_recently()`` — avoid moving toward that direction?
- ``best_action_for_state()`` — what worked last time in this state?
- ``reward_trend()`` — is the situation improving or deteriorating?

All methods are O(capacity) worst-case and operate in constant amortised
time, making them suitable for per-tick use.

Usage
-----
    wm = WorkingMemory(capacity=20)
    wm.push(WorkingMemorySlot(tick=1, state_key=s, action="north",
                              reward=0.1, cell_type="empty"))
    if wm.avg_recent_reward() < 0:
        # Do something defensive
        ...
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class WorkingMemorySlot:
    """One entry in working memory."""

    tick: int
    state_key: tuple
    action: str
    reward: float
    cell_type: str                              # cell type the agent was on
    metabolic_snapshot: dict = field(default_factory=dict)  # vitals at this tick


class WorkingMemory:
    """Bounded short-term memory for immediate decision-making.

    Parameters
    ----------
    capacity:
        Number of recent slots to retain (default 20).
    """

    def __init__(self, capacity: int = 20) -> None:
        self._slots: deque[WorkingMemorySlot] = deque(maxlen=capacity)

    def push(self, slot: WorkingMemorySlot) -> None:
        """Append a new slot, evicting the oldest if at capacity."""
        self._slots.append(slot)

    def recent(self, n: int = 10) -> list[WorkingMemorySlot]:
        """Return the *n* most recent slots (oldest first)."""
        return list(self._slots)[-n:]

    def recent_rewards(self, n: int = 10) -> list[float]:
        """Return the rewards of the *n* most recent slots."""
        return [s.reward for s in self.recent(n)]

    def avg_recent_reward(self, n: int = 10) -> float:
        """Mean reward over the *n* most recent ticks."""
        rewards = self.recent_rewards(n)
        return sum(rewards) / len(rewards) if rewards else 0.0

    def has_hazard_nearby_recently(self, n: int = 5) -> bool:
        """True if a hazard cell appeared in any of the last *n* slots."""
        return any(
            s.cell_type in ("radiation", "toxin") for s in self.recent(n)
        )

    def best_action_for_state(self, state_key: tuple) -> str | None:
        """Return the action with the highest reward observed for *state_key*.

        Searches the full working memory buffer.  Returns None if the state
        has never been seen in the current window.
        """
        candidates = [s for s in self._slots if s.state_key == state_key]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.reward).action

    def reward_trend(self, window: int = 10) -> float:
        """Estimate the slope of recent rewards (positive = improving).

        Uses a simple least-squares linear regression over the last *window*
        rewards.  Returns 0.0 when insufficient data is available.
        """
        rewards = self.recent_rewards(window)
        n = len(rewards)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(rewards) / n
        numerator = sum((i - x_mean) * (r - y_mean) for i, r in enumerate(rewards))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    def last_action(self) -> str | None:
        """Return the most recently executed action, or None if empty."""
        if not self._slots:
            return None
        return self._slots[-1].action

    def __len__(self) -> int:
        return len(self._slots)
