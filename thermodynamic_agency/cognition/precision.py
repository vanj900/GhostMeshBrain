"""PrecisionEngine — dynamic precision control for the GhostMesh organism.

Precision (inverse variance) weights determine how much the agent
"cares about" prediction errors on each vital dimension.  In Free Energy
Principle terms, precision weighting is the mechanism by which attention
and arousal are allocated.

The PrecisionEngine tunes these weights based on the organism's current
metabolic state and affect signal:

- **Sweet-spot** (moderate stress, positive or neutral affect):
  Boost precision on salient vitals → sharper attention, faster learning.
- **Overload** (extreme stress: heat/waste critical, affect strongly negative):
  Dampen precision to prevent catastrophic over-correction; risk managed via
  ethics immune pruning.
- **Dormant** (healthy, low surprise):
  Return to baseline — conserve metabolic effort.

Usage
-----
    engine = PrecisionEngine()
    weights = engine.tune(state)
    # weights is a dict[str, float] suitable for use in compute_efe / Surgeon
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from thermodynamic_agency.core.metabolic import MetabolicState

# Base precision weights (mirrors _PRECISION in inference.py)
BASE_PRECISION: dict[str, float] = {
    "energy": 2.0,
    "heat": 1.5,
    "waste": 1.0,
    "integrity": 1.8,
    "stability": 1.4,
}

# Bounds on how far precision can be amplified or damped
PRECISION_MIN: float = 0.3
PRECISION_MAX: float = 6.0

# Metabolic thresholds that define the "sweet spot" band
# Below lower bound → dormant (too cold, cheap predictions)
# Above upper bound → overload (too hot, risk of collapse)
STRESS_LOWER: float = 8.0    # free-energy below this → cheap/dormant
STRESS_UPPER: float = 45.0   # free-energy above this → overload damping

# Bell-curve arousal: controls the width of the attention peak in the sweet-spot.
# Smaller = sharper peak (attention spikes narrowly at mid-stress);
# larger = broader peak (attention elevated across most of the sweet-spot band).
_AROUSAL_VARIANCE: float = 0.10

# How strongly deviations from setpoints boost precision in sweet-spot mode.
# Larger = more aggressive sharpening = more metabolic cost.
_AROUSAL_BOOST_SCALE: float = 0.8


@dataclass
class PrecisionReport:
    """Records what the precision engine did this step."""

    weights: dict[str, float]
    free_energy: float
    regime: str          # "dormant", "sweet_spot", "overload"
    affect: float
    # Metabolic cost paid to sharpen attention (charged to state externally)
    energy_cost: float = 0.0
    heat_cost: float = 0.0


class PrecisionEngine:
    """Dynamic precision controller — tunes attention under metabolic pressure."""

    def __init__(self) -> None:
        self._weights: dict[str, float] = dict(BASE_PRECISION)
        # Mutable per-instance base — the SelfModEngine (Phase 4) may adjust
        # individual entries within [PRECISION_MIN, PRECISION_MAX].
        self.base_precision: dict[str, float] = dict(BASE_PRECISION)

    # ------------------------------------------------------------------ #
    # Main interface                                                       #
    # ------------------------------------------------------------------ #

    def tune(self, state: MetabolicState, compute_load: float = 1.0) -> PrecisionReport:
        """Compute and return precision weights for the current metabolic state.

        The weights are stored internally and returned as a ``PrecisionReport``.
        The *caller* is responsible for applying the metabolic cost reported in
        the result to the MetabolicState.

        Parameters
        ----------
        state:
            Current MetabolicState (read-only within this method).
        compute_load:
            Current computational burden — scales the metabolic cost of
            sharpening attention.

        Returns
        -------
        PrecisionReport
            Contains updated weights, regime classification, and the metabolic
            costs that should be charged to the state.
        """
        fe = state.free_energy_estimate()
        affect = state.affect

        if fe <= STRESS_LOWER:
            regime = "dormant"
            weights = self._dormant_weights()
            energy_cost = 0.0
            heat_cost = 0.0
        elif fe >= STRESS_UPPER:
            regime = "overload"
            weights, energy_cost, heat_cost = self._overload_weights(
                state, fe, compute_load
            )
        else:
            regime = "sweet_spot"
            weights, energy_cost, heat_cost = self._sweet_spot_weights(
                state, fe, affect, compute_load
            )

        self._weights = weights
        return PrecisionReport(
            weights=dict(weights),
            free_energy=fe,
            regime=regime,
            affect=affect,
            energy_cost=energy_cost,
            heat_cost=heat_cost,
        )

    @property
    def weights(self) -> dict[str, float]:
        """Current precision weights (last computed by ``tune()``)."""
        return dict(self._weights)

    # ------------------------------------------------------------------ #
    # Regime-specific weight calculators                                   #
    # ------------------------------------------------------------------ #

    def _dormant_weights(self) -> dict[str, float]:
        """Relax toward base precision — low metabolic cost."""
        return {k: v * 0.8 for k, v in self.base_precision.items()}

    def _sweet_spot_weights(
        self,
        state: MetabolicState,
        fe: float,
        affect: float,
        compute_load: float,
    ) -> tuple[dict[str, float], float, float]:
        """Sharpen attention on whichever vitals are most surprising.

        Returns (weights, energy_cost, heat_cost).
        """
        # Arousal multiplier — peaks at the mid-point of the sweet spot
        normalised = (fe - STRESS_LOWER) / (STRESS_UPPER - STRESS_LOWER)
        # Bell-curve: peak at ~0.5 normalised stress (mid sweet-spot)
        arousal = math.exp(-((normalised - 0.5) ** 2) / _AROUSAL_VARIANCE)

        # Boost precision on vitals that are furthest from setpoint
        weights: dict[str, float] = {}
        for vital, base in self.base_precision.items():
            deviation = self._vital_deviation(state, vital)
            boost = 1.0 + arousal * deviation * _AROUSAL_BOOST_SCALE
            weights[vital] = min(PRECISION_MAX, base * boost)

        # Positive affect (surprise resolving) slightly damps cost
        cost_scale = compute_load * (1.0 - 0.3 * max(0.0, affect))
        total_boost = sum(w - b for w, b in zip(weights.values(), self.base_precision.values()))
        energy_cost = max(0.0, total_boost * 0.04 * cost_scale)
        heat_cost = max(0.0, total_boost * 0.02 * cost_scale)

        return weights, energy_cost, heat_cost

    def _overload_weights(
        self,
        state: MetabolicState,
        fe: float,
        compute_load: float,
    ) -> tuple[dict[str, float], float, float]:
        """Dampen precision to prevent cascade; flag survival dimensions.

        Under overload the organism risks over-pruning — the ethics immune
        system catches truly dangerous actions, but here we reduce the
        precision on all dimensions except the most immediately lethal ones
        (energy and heat).
        """
        overload_factor = min(1.0, (fe - STRESS_UPPER) / 45.0)
        damp = 1.0 - 0.5 * overload_factor  # damp by up to 50%

        weights: dict[str, float] = {}
        for vital, base in self.base_precision.items():
            if vital in ("energy", "heat"):
                # Keep survival precision high even under overload
                weights[vital] = min(PRECISION_MAX, base * (1.0 + overload_factor))
            else:
                weights[vital] = max(PRECISION_MIN, base * damp)

        # Overload is cheap because we're dampening, not sharpening
        energy_cost = 0.0
        heat_cost = 0.0
        return weights, energy_cost, heat_cost

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _vital_deviation(state: MetabolicState, vital: str) -> float:
        """Normalised deviation of a vital from its setpoint (0–1)."""
        setpoints = {
            "energy": 80.0,
            "heat": 20.0,
            "waste": 10.0,
            "integrity": 85.0,
            "stability": 80.0,
        }
        ranges = {
            "energy": 80.0,
            "heat": 80.0,
            "waste": 90.0,
            "integrity": 85.0,
            "stability": 80.0,
        }
        current = {
            "energy": state.energy,
            "heat": state.heat,
            "waste": state.waste,
            "integrity": state.integrity,
            "stability": state.stability,
        }
        sp = setpoints.get(vital, 50.0)
        rng = ranges.get(vital, 50.0)
        val = current.get(vital, sp)
        return min(1.0, abs(val - sp) / rng)
