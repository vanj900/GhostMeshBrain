"""RunLogger — structured per-tick vital-sign logging for post-run analysis.

Every heartbeat the pulse loop records a :class:`TickRecord` snapshot.
Records accumulate in memory and are optionally streamed to a JSONL file
(one JSON object per line) specified by ``GHOST_LOG_FILE``.

Usage
-----
    from thermodynamic_agency.run_logger import RunLogger, TickRecord

    logger = RunLogger(path="/tmp/run.jsonl")
    logger.record(TickRecord(...))
    print(logger.summary())
    logger.close()
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import TextIO


@dataclass
class TickRecord:
    """Snapshot of organism state at a single tick."""

    tick: int
    action: str                 # FORAGE / REST / REPAIR / DECIDE
    mask: str                   # active personality mask name
    energy: float
    heat: float
    waste: float
    integrity: float
    stability: float
    affect: float               # [-1, +1] pleasure signal
    free_energy: float          # scalar surprise proxy
    precision_regime: str       # "dormant" / "sweet_spot" / "overload"
    health_score: float         # aggregate 0-100 wellness
    stage: str                  # "dormant" / "emerging" / "aware" / "evolved"
    ethics_blocks: int = 0      # proposals blocked by ethics gate this tick
    stressor_event: str = ""    # description of environmental disturbance, or ""
    allostatic_load: float = 0.0  # accumulated allostatic load [0, 100]
    decide_streak: int = 0        # consecutive DECIDE ticks


class RunLogger:
    """Accumulates TickRecord snapshots and optionally streams them to a JSONL file.

    Parameters
    ----------
    path:
        File path for JSONL output.  If ``None`` or ``""``, records are
        kept in memory only.

    Notes
    -----
    Callers **must** call :meth:`close` when finished to flush and release
    the underlying file handle.  Alternatively, use the instance as a
    context manager::

        with RunLogger(path="/tmp/run.jsonl") as logger:
            logger.record(TickRecord(...))
    """

    def __init__(self, path: str | None = None) -> None:
        self._path: str | None = path
        self._records: list[TickRecord] = []
        self._fh: TextIO | None = None
        if path:
            self._fh = open(path, "w")  # noqa: SIM115  (intentional long-lived file)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def record(self, rec: TickRecord) -> None:
        """Append a tick record; flush to disk if a file is open."""
        self._records.append(rec)
        if self._fh is not None:
            self._fh.write(json.dumps(asdict(rec)) + "\n")
            self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle (no-op if none)."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # Context manager support
    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def records(self) -> list[TickRecord]:
        """Read-only view of accumulated records."""
        return list(self._records)

    def summary(self) -> dict:
        """Aggregate statistics over all recorded ticks.

        Returns an empty dict if no ticks have been recorded yet.
        """
        n = len(self._records)
        if not n:
            return {}

        action_counts: dict[str, int] = {}
        mask_counts: dict[str, int] = {}
        regime_counts: dict[str, int] = {}
        for r in self._records:
            action_counts[r.action] = action_counts.get(r.action, 0) + 1
            mask_counts[r.mask] = mask_counts.get(r.mask, 0) + 1
            if r.precision_regime:
                regime_counts[r.precision_regime] = (
                    regime_counts.get(r.precision_regime, 0) + 1
                )

        affects = [r.affect for r in self._records]
        fes = [r.free_energy for r in self._records]
        healths = [r.health_score for r in self._records]

        return {
            "total_ticks": n,
            "final_tick": self._records[-1].tick,
            "final_stage": self._records[-1].stage,
            "action_distribution": action_counts,
            "mask_distribution": mask_counts,
            "precision_regime_distribution": regime_counts,
            "affect": {
                "mean": sum(affects) / n,
                "min": min(affects),
                "max": max(affects),
            },
            "free_energy": {
                "mean": sum(fes) / n,
                "min": min(fes),
                "max": max(fes),
            },
            "health": {
                "mean": sum(healths) / n,
                "min": min(healths),
                "max": max(healths),
            },
            "total_ethics_blocks": sum(r.ethics_blocks for r in self._records),
            "total_stressor_events": sum(
                1 for r in self._records if r.stressor_event
            ),
        }
