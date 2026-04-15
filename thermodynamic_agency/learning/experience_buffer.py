"""ExperienceBuffer — fixed-capacity (s, a, r, s') replay buffer.

Stores individual experiences from environment interaction and supports
both uniform random sampling (for minibatch updates) and recency access.

Usage
-----
    buf = ExperienceBuffer(maxlen=10_000)
    buf.push(Experience(tick=1, state_key=s, action="north", reward=0.1,
                        next_state_key=s2))
    batch = buf.sample(32)
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Experience:
    """Single transition tuple (s, a, r, s', done)."""

    tick: int
    state_key: tuple
    action: str
    reward: float
    next_state_key: tuple
    done: bool = False
    metadata: dict = field(default_factory=dict)


class ExperienceBuffer:
    """Fixed-capacity circular replay buffer.

    Parameters
    ----------
    maxlen:
        Maximum stored experiences; oldest are silently evicted when full.
    seed:
        Optional RNG seed for reproducible ``sample()`` calls.
    """

    def __init__(self, maxlen: int = 10_000, seed: int | None = None) -> None:
        self._buffer: deque[Experience] = deque(maxlen=maxlen)
        self._rng = random.Random(seed)

    def push(self, exp: Experience) -> None:
        """Append one experience."""
        self._buffer.append(exp)

    def sample(self, n: int) -> list[Experience]:
        """Return up to *n* experiences sampled uniformly at random."""
        n = min(n, len(self._buffer))
        if n == 0:
            return []
        return self._rng.sample(list(self._buffer), n)

    def recent(self, n: int = 20) -> list[Experience]:
        """Return the *n* most recent experiences (oldest first)."""
        return list(self._buffer)[-n:]

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def total_stored(self) -> int:
        return len(self._buffer)

    def cumulative_reward(self) -> float:
        """Sum of all stored rewards."""
        return sum(e.reward for e in self._buffer)

    def avg_reward(self) -> float:
        """Mean reward over all stored experiences."""
        n = len(self._buffer)
        return self.cumulative_reward() / n if n else 0.0
