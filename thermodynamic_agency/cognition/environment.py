"""EnvironmentStressor — stochastic external disturbances for the GhostMesh organism.

The organism's passive decay makes it slowly degrade toward death on a
predictable schedule.  An ``EnvironmentStressor`` injects *unpredictable*
shocks so that the agent must respond reactively (foraging proactively,
managing heat under surprise bursts) rather than just following the
deterministic decay curve.

Disturbance types
-----------------
energy_drain    — sudden compute demand strips energy (simulates external load)
heat_burst      — spike in context congestion / thermal event
waste_dump      — noisy input injection swells waste accumulation
stability_quake — entropic disturbance lowers stability

Environment modes (``mode`` parameter)
----------------------------------------
"flat"              — independent Bernoulli disturbances every tick (original)
"bursty"            — clusters of back-to-back disturbances
"hostile_windows"   — occasional structured threat periods with elevated stressor
                      rate and stronger magnitude; simulates "storm windows"

Configuration (environment variables)
--------------------------------------
GHOST_STRESSOR_PROB       float in [0, 1], default 0.0 (disabled)
GHOST_STRESSOR_INTENSITY  float ≥ 0,       default 1.0 (magnitude scale)
GHOST_STRESSOR_SEED       int (optional)   fixed seed for reproducibility

Usage
-----
    stressor = EnvironmentStressor(prob=0.05, intensity=1.0, mode="hostile_windows")
    event = stressor.maybe_disturb(state)
    # event is "" if no disturbance fired, else a human-readable string.
"""

from __future__ import annotations

import random
from typing import Literal

from thermodynamic_agency.core.metabolic import MetabolicState

# Magnitude ranges for each disturbance type at intensity=1.0.
# Expressed as (min, max) deltas applied to the relevant vital.
_RANGES: dict[str, tuple[float, float]] = {
    "energy_drain":     (5.0, 20.0),   # delta_energy (negative)
    "heat_burst":       (5.0, 15.0),   # delta_heat   (positive)
    "waste_dump":       (8.0, 25.0),   # delta_waste  (positive)
    "stability_quake":  (3.0, 12.0),   # delta_stability (negative)
}

_DISTURBANCE_TYPES: list[str] = list(_RANGES.keys())

EnvironmentMode = Literal["flat", "bursty", "hostile_windows"]

# Hostile-window parameters
_HW_ENTRY_PROB: float = 0.01           # probability per tick of entering a hostile window
_HW_MIN_DURATION: int = 10             # minimum hostile-window length in ticks
_HW_MAX_DURATION: int = 25             # maximum hostile-window length in ticks
_HW_STRESSOR_MULTIPLIER: float = 2.0   # stressor probability multiplier during a window
_HW_INTENSITY_MULTIPLIER: float = 1.8  # magnitude multiplier during a window


class EnvironmentStressor:
    """Injects random external shocks into the organism's metabolic state.

    Parameters
    ----------
    prob:
        Probability (0–1) of a disturbance firing on any given tick.
        At prob=0.05 roughly one tick in twenty gets an event.
    intensity:
        Scales the magnitude of disturbances.  Values > 1.0 make the
        environment harsher; values < 1.0 make it gentler.
    seed:
        Optional RNG seed for reproducible experiments.
    mode:
        Environment mode controlling disturbance structure:
        ``"flat"`` (default) — independent events each tick;
        ``"bursty"`` — events cluster in short bursts;
        ``"hostile_windows"`` — occasional structured threat periods with
        elevated rate and stronger magnitude (Phase 5).
    """

    def __init__(
        self,
        prob: float = 0.0,
        intensity: float = 1.0,
        seed: int | None = None,
        mode: EnvironmentMode = "flat",
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        if intensity < 0.0:
            raise ValueError(f"intensity must be ≥ 0, got {intensity}")
        self.prob = prob
        self.intensity = intensity
        self.mode = mode
        self._rng = random.Random(seed)

        # Hostile-window state (Phase 5)
        self._in_hostile_window: bool = False
        self._hostile_ticks_remaining: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def in_hostile_window(self) -> bool:
        """Whether the stressor is currently in a hostile window period."""
        return self._in_hostile_window

    def maybe_disturb(self, state: MetabolicState) -> str:
        """Possibly apply one disturbance to *state* this tick.

        Parameters
        ----------
        state:
            MetabolicState — mutated in-place via ``apply_action_feedback()``
            if a disturbance fires.

        Returns
        -------
        str
            Human-readable event description, or ``""`` if no event fired.
        """
        if self.prob <= 0.0:
            return ""

        if self.mode == "hostile_windows":
            return self._maybe_disturb_hostile_windows(state)
        if self.mode == "bursty":
            return self._maybe_disturb_bursty(state)
        # Default: "flat" — original independent Bernoulli
        if self._rng.random() > self.prob:
            return ""
        event_type = self._rng.choice(_DISTURBANCE_TYPES)
        return self._apply(event_type, state)

    # ------------------------------------------------------------------ #
    # Mode implementations                                                 #
    # ------------------------------------------------------------------ #

    def _maybe_disturb_hostile_windows(self, state: MetabolicState) -> str:
        """Hostile-window mode: structured threat periods (Phase 5).

        Each tick there is a small chance of entering a hostile window that
        lasts 10–25 ticks.  During a window, the stressor probability and
        magnitude are both amplified, creating clustered crisis events that
        reward anticipatory behaviour rather than reactive firefighting.
        """
        # Advance (or enter) hostile window
        if self._in_hostile_window:
            self._hostile_ticks_remaining -= 1
            if self._hostile_ticks_remaining <= 0:
                self._in_hostile_window = False
        elif self._rng.random() < _HW_ENTRY_PROB:
            self._in_hostile_window = True
            self._hostile_ticks_remaining = self._rng.randint(
                _HW_MIN_DURATION, _HW_MAX_DURATION
            )

        # Determine effective stressor rate and intensity
        if self._in_hostile_window:
            effective_prob = min(1.0, self.prob * _HW_STRESSOR_MULTIPLIER)
            effective_intensity = self.intensity * _HW_INTENSITY_MULTIPLIER
        else:
            effective_prob = self.prob
            effective_intensity = self.intensity

        if self._rng.random() > effective_prob:
            return ""

        event_type = self._rng.choice(_DISTURBANCE_TYPES)
        event_str = self._apply(event_type, state, intensity_override=effective_intensity)
        if self._in_hostile_window:
            return f"[HOSTILE_WINDOW] {event_str}"
        return event_str

    def _maybe_disturb_bursty(self, state: MetabolicState) -> str:
        """Bursty mode: back-to-back disturbances within short windows.

        With twice the base probability, but disturbances come in tight
        clusters — each event increases the chance of a follow-on event in
        the same tick via a second independent roll at the same rate.
        """
        if self._rng.random() > self.prob * 2.0:
            return ""
        event_type = self._rng.choice(_DISTURBANCE_TYPES)
        result = self._apply(event_type, state)
        # Chain: small chance of a second immediate hit in the same tick
        if result and self._rng.random() < 0.3:
            event_type2 = self._rng.choice(_DISTURBANCE_TYPES)
            result2 = self._apply(event_type2, state)
            if result2:
                return f"{result} + {result2}"
        return result

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _apply(
        self,
        event_type: str,
        state: MetabolicState,
        intensity_override: float | None = None,
    ) -> str:
        intensity = intensity_override if intensity_override is not None else self.intensity
        lo, hi = _RANGES[event_type]
        magnitude = self._rng.uniform(lo, hi) * intensity

        if event_type == "energy_drain":
            state.apply_action_feedback(delta_energy=-magnitude)
            return f"energy_drain(-{magnitude:.1f})"

        if event_type == "heat_burst":
            state.apply_action_feedback(delta_heat=magnitude)
            return f"heat_burst(+{magnitude:.1f})"

        if event_type == "waste_dump":
            state.apply_action_feedback(delta_waste=magnitude)
            return f"waste_dump(+{magnitude:.1f})"

        if event_type == "stability_quake":
            state.apply_action_feedback(delta_stability=-magnitude)
            return f"stability_quake(-{magnitude:.1f})"

        return ""  # unreachable but keeps linters happy
