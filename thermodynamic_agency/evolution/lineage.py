"""Lineage — cross-life evolutionary memory for GhostMesh.

When an organism dies its ``Lineage`` record is serialised and appended to
the family-tree file.  On next instantiation the ``LineageTracker`` can load
the parent's record and seed the new Q-table with mutated prior weights,
satisfying the Genesis Doctrine constraint (prior-seeding, not direct
self-replication).

Design notes
------------
- ``Lineage`` is a pure dataclass — no side effects, fully serialisable.
- ``LineageTracker`` manages the on-disk family tree (JSONL format).
- Q-table mutation: each entry is multiplied by ``(1 + N(0, mutation_rate))``.
  Default ``mutation_rate`` is 0.05 — small enough that the inherited prior
  is a soft nudge rather than a deterministic inheritance.
- Lineage records are append-only.  Each run adds one record.
- The ``dreamer_fraction`` lineage field is the key plasticity signal:
  organisms descended from high-dreamer ancestors should trend toward longer
  lifespans.  This is tracked in ``dreamer_fractions`` across the family tree.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any


# Default lineage file path — overridable via env var
LINEAGE_FILE = os.environ.get(
    "GHOST_LINEAGE_FILE",
    "/tmp/ghost_lineage.jsonl",
)

# Default per-generation mutation rate for Q-table inheritance
DEFAULT_MUTATION_RATE: float = 0.05

# Number of top Q-table entries to carry forward as generational priors
TOP_Q_ENTRIES: int = 5

# Small prior boost applied to inherited Q-values (before noise scaling)
# This ensures the child starts with a slight preference for paths the
# parent found rewarding without inheriting deterministic strategy.
Q_PRIOR_BOOST: float = 0.1


@dataclass
class Lineage:
    """Generational record saved when a GhostMesh organism dies.

    Attributes
    ----------
    lineage_id:
        Unique identifier for this lineage record (UUID-style string).
    parent_id:
        Lineage ID of the parent organism, or ``None`` for the first generation.
    life_number:
        Sequential life counter within the organism's reincarnation chain.
    lifespan:
        Number of ticks the organism survived.
    interiority_score:
        Final interiority score (0–1) from MetaCognitiveSelfModel.
    dreamer_fraction:
        Fraction of ticks spent in Dreamer/plastic masks from CollapseProbe.
    guardian_fraction:
        Fraction of ticks spent in Guardian masks from CollapseProbe.
    plasticity_index:
        Final plasticity_index reading from CollapseProbe.
    cause_of_death:
        Exception class name that terminated the organism.
    top_q_entries:
        Top-``TOP_Q_ENTRIES`` (state_key, action, q_value) tuples serialised
        as dicts.  These seed the next organism's Q-table.
    mask_preferences:
        Fractional dwell-time per mask name during this life.
    mutation_rate:
        Per-generation Q-value noise scale (default 0.05).
    timestamp:
        Wall-clock Unix timestamp at the moment of death.
    """

    lineage_id: str
    parent_id: str | None
    life_number: int
    lifespan: int
    interiority_score: float
    dreamer_fraction: float
    guardian_fraction: float
    plasticity_index: float
    cause_of_death: str
    top_q_entries: list[dict[str, Any]]
    mask_preferences: dict[str, float]
    mutation_rate: float = DEFAULT_MUTATION_RATE
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Lineage":
        return cls(**data)


class LineageTracker:
    """Manages the cross-life family tree of GhostMesh organisms.

    Records are appended to a JSONL file so multiple runs accumulate a
    persistent evolutionary history.

    Parameters
    ----------
    path:
        Path to the JSONL lineage file (default: ``LINEAGE_FILE``).
    mutation_rate:
        Per-generation mutation rate for Q-value inheritance.
    """

    def __init__(
        self,
        path: str = LINEAGE_FILE,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
    ) -> None:
        self._path = path
        self._mutation_rate = mutation_rate
        self._lineages: list[Lineage] = []
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, lineage: Lineage) -> None:
        """Append a new lineage record and persist it immediately.

        Parameters
        ----------
        lineage:
            The completed Lineage record for the just-terminated organism.
        """
        self._lineages.append(lineage)
        self._append_to_file(lineage)

    def get_parent(self, lineage_id: str) -> Lineage | None:
        """Return the Lineage record with the given ID, or None."""
        for rec in reversed(self._lineages):
            if rec.lineage_id == lineage_id:
                return rec
        return None

    def latest(self) -> Lineage | None:
        """Return the most recently recorded Lineage, or None if empty."""
        return self._lineages[-1] if self._lineages else None

    def family_tree(self) -> list[Lineage]:
        """Return all recorded lineages in chronological order."""
        return list(self._lineages)

    def dreamer_fractions(self) -> list[float]:
        """Return the dreamer_fraction for each generation in order.

        This is the primary plasticity selection signal: higher dreamer
        fractions in ancestors should correlate with longer offspring lifespans.
        """
        return [lin.dreamer_fraction for lin in self._lineages]

    def lineage_fitness(
        self,
        lifespan_weight: float = 0.6,
        dreamer_weight: float = 0.4,
    ) -> list[float]:
        """Return a per-generation composite fitness score.

        The fitness proxy combines lifespan (normalised by the maximum observed
        lifespan across the lineage) and dreamer_fraction (the primary
        plasticity signal).  Weights are configurable so callers can emphasise
        survival vs. plasticity when comparing lineage strategies.

        Parameters
        ----------
        lifespan_weight:
            Weight applied to the normalised lifespan component (default 0.6).
        dreamer_weight:
            Weight applied to the dreamer_fraction component (default 0.4).

        Returns
        -------
        list[float]
            Fitness scores in chronological generation order.  Returns an
            empty list when fewer than two generations have been recorded.
        """
        if not self._lineages:
            return []
        lifespans = [lin.lifespan for lin in self._lineages]
        _max_ls = max(lifespans)
        max_lifespan = _max_ls if _max_ls > 0 else 1
        scores: list[float] = []
        for lin in self._lineages:
            norm_lifespan = lin.lifespan / max_lifespan
            score = lifespan_weight * norm_lifespan + dreamer_weight * lin.dreamer_fraction
            scores.append(round(score, 6))
        return scores

    def plasticity_selection_signal(self) -> float:
        """Pearson correlation between dreamer_fraction[t] and lifespan[t+1].

        Measures whether high plasticity in one generation predicts longer
        survival in the next — the key cross-generational selection signal.

        Returns
        -------
        float
            Pearson r in [-1, 1].  Positive values mean plastic parents tend
            to have longer-lived offspring.  Returns 0.0 when fewer than three
            paired observations are available (too few for meaningful statistics).
        """
        if len(self._lineages) < 3:
            return 0.0
        # Build paired (dreamer_fraction[t], lifespan[t+1]) arrays
        xs = [lin.dreamer_fraction for lin in self._lineages[:-1]]
        ys = [lin.lifespan for lin in self._lineages[1:]]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        denom = math.sqrt(var_x * var_y)
        if denom == 0.0:
            return 0.0
        return round(cov / denom, 6)

    def seed_q_table(
        self,
        q_table: dict,
        parent_id: str | None = None,
        rng: random.Random | None = None,
    ) -> int:
        """Apply prior boosts from the parent's top Q-table entries.

        Parameters
        ----------
        q_table:
            The current Q-learner's Q-table dict (mutated in-place).
            Keys are ``(state_key, action)`` tuples; values are floats.
        parent_id:
            Lineage ID of the parent organism.  If None, uses the latest
            recorded lineage.
        rng:
            Optional RNG for reproducible mutation noise.

        Returns
        -------
        int
            Number of Q-table entries seeded (0 if no parent found).
        """
        if not self._lineages:
            return 0
        parent = self.get_parent(parent_id) if parent_id else self.latest()
        if parent is None:
            return 0

        _rng = rng or random.Random()
        seeded = 0
        for entry in parent.top_q_entries:
            try:
                state_key_raw = entry.get("state_key")
                action = entry.get("action")
                q_value = float(entry.get("q_value", 0.0))
                if state_key_raw is None or action is None:
                    continue
        # Reconstructed state_key: lists are re-converted to tuples for Q-table
        # key compatibility; all other types (strings, tuples) pass through.
                state_key = tuple(state_key_raw) if isinstance(state_key_raw, list) else state_key_raw
                if not isinstance(state_key, (str, tuple)):
                    continue
                # Apply prior boost + Gaussian mutation noise
                noise = _rng.gauss(0.0, self._mutation_rate)
                boosted = (q_value + Q_PRIOR_BOOST) * (1.0 + noise)
                key = (state_key, action)
                if key not in q_table or abs(q_table[key]) < abs(boosted):
                    q_table[key] = boosted
                seeded += 1
            except (KeyError, TypeError, ValueError):
                continue
        return seeded

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load existing lineage records from disk (if the file exists)."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        self._lineages.append(Lineage.from_dict(data))
                    except (json.JSONDecodeError, TypeError, KeyError):
                        continue
        except OSError:
            pass

    def _append_to_file(self, lineage: Lineage) -> None:
        """Append one record to the JSONL file."""
        try:
            with open(self._path, "a") as fh:
                fh.write(json.dumps(lineage.to_dict()) + "\n")
        except OSError:
            pass


# ── Helpers used by pulse.py to build Lineage records on death ──────────────

def _generate_lineage_id() -> str:
    """Generate a short unique ID for a lineage record."""
    import hashlib
    raw = f"{time.time()}-{random.random()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def extract_top_q_entries(
    q_table: dict,
    n: int = TOP_Q_ENTRIES,
) -> list[dict[str, Any]]:
    """Extract the top-n Q-table entries by absolute value.

    Parameters
    ----------
    q_table:
        Q-learner Q-table dict with keys ``(state_key, action)`` and float values.
    n:
        Number of top entries to extract (default 5).

    Returns
    -------
    list[dict]
        Serialisable list of ``{"state_key": ..., "action": ..., "q_value": ...}``
        dicts sorted by descending absolute Q-value.
    """
    if not q_table:
        return []
    sorted_entries = sorted(q_table.items(), key=lambda kv: abs(kv[1]), reverse=True)
    result = []
    for (state_key, action), q_val in sorted_entries[:n]:
        result.append({
            "state_key": list(state_key) if hasattr(state_key, "__iter__") and not isinstance(state_key, str) else state_key,
            "action": action,
            "q_value": float(q_val),
        })
    return result
