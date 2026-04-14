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

Configuration (environment variables)
--------------------------------------
GHOST_STRESSOR_PROB       float in [0, 1], default 0.0 (disabled)
GHOST_STRESSOR_INTENSITY  float ≥ 0,       default 1.0 (magnitude scale)
GHOST_STRESSOR_SEED       int (optional)   fixed seed for reproducibility

Usage
-----
    stressor = EnvironmentStressor(prob=0.05, intensity=1.0)
    event = stressor.maybe_disturb(state)
    # event is "" if no disturbance fired, else a human-readable string.
"""

from __future__ import annotations

import random

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
    """

    def __init__(
        self,
        prob: float = 0.0,
        intensity: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        if intensity < 0.0:
            raise ValueError(f"intensity must be ≥ 0, got {intensity}")
        self.prob = prob
        self.intensity = intensity
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

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
        if self.prob <= 0.0 or self._rng.random() > self.prob:
            return ""
        event_type = self._rng.choice(_DISTURBANCE_TYPES)
        return self._apply(event_type, state)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _apply(self, event_type: str, state: MetabolicState) -> str:
        lo, hi = _RANGES[event_type]
        magnitude = self._rng.uniform(lo, hi) * self.intensity

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
