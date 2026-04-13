"""Personality masks — the organism's adaptive behavioral modes.

Masks rotate during adapt cycles or in response to metabolic cues.  Each mask
biases action selection (shifts predicted-delta weights in EFE computation)
and adjusts the organism's "voice" in diary entries.

Built-in masks
--------------
Healer   – prioritises integrity and stability; cautious
Judge    – prioritises ethics screening; conservative
Courier  – prioritises speed and resource acquisition; aggressive
Dreamer  – prioritises reflection and insight consolidation; introspective
Guardian – prioritises survival (energy + heat management); defensive
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Mask:
    """A personality mask with metabolic biases."""

    name: str
    description: str

    # Multipliers applied to proposal predicted_deltas during EFE scoring.
    # Keys match MetabolicState vitals.  Values > 1 amplify that dimension's
    # contribution to EFE (making the mask more sensitive to changes there).
    efe_precision_overrides: dict[str, float] = field(default_factory=dict)

    # Diary voice prefix
    voice_prefix: str = ""

    # Condition under which this mask activates automatically
    auto_activate: Callable[["MaskState"], bool] | None = field(
        default=None, repr=False, compare=False
    )

    # Minimum ticks before the mask can be rotated away
    min_ticks: int = 5


@dataclass
class MaskState:
    """Tracks which mask is active and for how long."""

    active_mask: Mask
    activated_at_tick: int = 0
    ticks_active: int = 0


# ------------------------------------------------------------------ #
# Built-in mask definitions                                           #
# ------------------------------------------------------------------ #

def _healer() -> Mask:
    return Mask(
        name="Healer",
        description="Prioritises integrity and stability; cautious and restorative.",
        efe_precision_overrides={"integrity": 3.5, "stability": 3.0, "heat": 2.0},
        voice_prefix="[Healer] ",
        auto_activate=lambda ms: ms.active_mask.name != "Healer",
        min_ticks=3,
    )


def _judge() -> Mask:
    return Mask(
        name="Judge",
        description="Heightened ethics sensitivity; conservative and deliberate.",
        efe_precision_overrides={"integrity": 4.0, "waste": 2.5},
        voice_prefix="[Judge] ",
        min_ticks=4,
    )


def _courier() -> Mask:
    return Mask(
        name="Courier",
        description="Speed and resource acquisition; aggressive foraging.",
        efe_precision_overrides={"energy": 3.0, "heat": 0.5},
        voice_prefix="[Courier] ",
        auto_activate=lambda ms: ms.active_mask.name != "Courier",
        min_ticks=3,
    )


def _dreamer() -> Mask:
    return Mask(
        name="Dreamer",
        description="Deep reflection and insight consolidation; introspective.",
        efe_precision_overrides={"integrity": 2.5, "stability": 2.5, "waste": 1.5},
        voice_prefix="[Dreamer] ",
        min_ticks=5,
    )


def _guardian() -> Mask:
    return Mask(
        name="Guardian",
        description="Survival-first; energy conservation and heat management.",
        efe_precision_overrides={"energy": 4.0, "heat": 3.5, "stability": 2.0},
        voice_prefix="[Guardian] ",
        auto_activate=lambda ms: ms.active_mask.name != "Guardian",
        min_ticks=3,
    )


ALL_MASKS: list[Mask] = [_healer(), _judge(), _courier(), _dreamer(), _guardian()]
_MASK_MAP: dict[str, Mask] = {m.name: m for m in ALL_MASKS}


# ------------------------------------------------------------------ #
# Mask rotator                                                        #
# ------------------------------------------------------------------ #

class MaskRotator:
    """Manages mask activation and rotation."""

    def __init__(self, initial_mask: str = "Guardian") -> None:
        self._state = MaskState(
            active_mask=_MASK_MAP.get(initial_mask, ALL_MASKS[0]),
            activated_at_tick=0,
        )

    @property
    def active(self) -> Mask:
        return self._state.active_mask

    def tick(self, current_tick: int) -> None:
        """Advance the mask state by one tick."""
        self._state.ticks_active += 1

    def maybe_rotate(
        self,
        current_tick: int,
        metabolic_hint: str = "DECIDE",
        force: str | None = None,
    ) -> Mask:
        """Possibly rotate to a new mask based on metabolic state or forcing.

        Parameters
        ----------
        current_tick:
            Current entropy tick from MetabolicState.
        metabolic_hint:
            Action token from tick() — biases mask selection.
        force:
            If provided, force-activate this mask by name.

        Returns
        -------
        Mask
            The active mask after (possible) rotation.
        """
        if force and force in _MASK_MAP:
            self._activate(_MASK_MAP[force], current_tick)
            return self._state.active_mask

        if self._state.ticks_active < self._state.active_mask.min_ticks:
            return self._state.active_mask

        # Metabolic hint → preferred mask
        preferred: dict[str, str] = {
            "FORAGE": "Courier",
            "REST": "Dreamer",
            "REPAIR": "Healer",
            "DECIDE": "Judge",
        }
        preferred_name = preferred.get(metabolic_hint, "Guardian")
        if preferred_name in _MASK_MAP and preferred_name != self._state.active_mask.name:
            self._activate(_MASK_MAP[preferred_name], current_tick)

        return self._state.active_mask

    def _activate(self, mask: Mask, tick: int) -> None:
        self._state = MaskState(
            active_mask=mask,
            activated_at_tick=tick,
            ticks_active=0,
        )

    def status(self) -> dict:
        return {
            "active_mask": self._state.active_mask.name,
            "ticks_active": self._state.ticks_active,
            "description": self._state.active_mask.description,
        }
