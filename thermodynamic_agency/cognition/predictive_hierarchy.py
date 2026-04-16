"""Hierarchical Predictive Coding — Phase 3.

Implements a three-level predictive-processing stack in the Friston/active
inference tradition.  Each layer maintains beliefs (predictions) about the
layer below it.  Prediction errors propagate upward; predictions propagate
downward.  Precision weighting gates what actually gets through at each
interface.

Layers
------
Level 0 — Brainstem / Spinal cord
    Raw vital signs from MetabolicState.  No internal model — just emits
    the current physiological reality as its "message upward".

Level 1 — Limbic / Subcortical
    Maintains a running model of what the brainstem *should* be doing given
    the organism's current drives.  Receives raw vitals from below, computes
    precision-weighted prediction errors, and sends a condensed error signal
    upward.  Also receives top-down predictions from prefrontal and adjusts
    its model accordingly.

Level 2 — Prefrontal / Cortical
    Maintains the organism's highest-level generative model — beliefs about
    the long-run trajectory of vitals.  Receives condensed error signals
    from limbic and generates top-down predictions for what the limbic layer
    should be observing.  These predictions flow back down to Level 1.

Information flow
----------------
    Upward (bottom-up):
        L0 raw_vitals → L1 computes prediction_error → L2 updates priors

    Downward (top-down):
        L2 emits predictions → L1 uses them to sharpen/damp its own model
        L1 emits adjusted predictions → L0 influences future action
            (only indirectly via precision passed to EFE)

Metabolic cost
--------------
Running the hierarchy is cheap by default.  The precision-weighting step
costs a small amount of heat proportional to the total weighted error
magnitude (more surprise = more neural activation = more heat).

Usage
-----
    hierarchy = PredictiveHierarchy()

    # Each heartbeat, after limbic processing:
    signal = hierarchy.update(state, limbic_signal)

    # Pass signal.top_down_precision into _decide() to refine EFE weighting.
    # Pass signal.hierarchical_error as an additional EFE penalty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections import deque
from typing import TYPE_CHECKING

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.limbic import LimbicSignal

if TYPE_CHECKING:
    from thermodynamic_agency.cognition.homeostasis import HomeostasisAdapter

# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #

# Vital names processed at each level
_VITALS: tuple[str, ...] = ("energy", "heat", "waste", "integrity", "stability")

# Setpoints the prefrontal layer "wants" to observe
_PREFRONTAL_SETPOINTS: dict[str, float] = {
    "energy": 80.0,
    "heat": 20.0,
    "waste": 10.0,
    "integrity": 85.0,
    "stability": 80.0,
}

# How quickly each layer's running model adapts to new observations.
# 0 = fully trust prior; 1 = fully trust observation.
_L1_LEARNING_RATE: float = 0.25   # limbic — fast affect-driven updating
_L2_LEARNING_RATE: float = 0.08   # prefrontal — slow prior consolidation

# Base precision at each level (before thalamic gating)
_L1_BASE_PRECISION: float = 1.2
_L2_BASE_PRECISION: float = 1.6

# How steeply prediction error at L2 translates into a metabolic heat cost.
# Kept small — we want error to shape behavior, not kill the organism.
_ERROR_HEAT_SCALE: float = 0.003

# Maximum hierarchical error penalty added to EFE in a DECIDE step
_MAX_HIER_EFE_PENALTY: float = 8.0

# Weight for how strongly top-down predictions from L2 "pull" L1's model
_TOP_DOWN_PULL: float = 0.15


# ------------------------------------------------------------------ #
# Data types                                                          #
# ------------------------------------------------------------------ #

@dataclass
class LayerBelief:
    """The belief state of a single hierarchical layer."""

    level: int                                    # 0=brainstem, 1=limbic, 2=prefrontal
    predictions: dict[str, float]                 # what this layer predicts for level below
    errors: dict[str, float] = field(default_factory=dict)  # current prediction errors
    precision: float = 1.0                        # current precision (gate weight)


@dataclass
class HierarchySignal:
    """Output of one hierarchy update step.

    Consumed by the pulse loop and _decide():

    top_down_precision
        Per-vital precision adjustments computed from L2 top-down predictions.
        These are merged with PrecisionEngine weights in _decide().

    hierarchical_error
        Scalar EFE penalty reflecting how badly the hierarchy mis-predicted
        this tick's vitals.  Added to the EFE score of each proposal.

    heat_cost
        Metabolic heat generated by running the hierarchy this tick.

    layer_errors
        Dict of per-vital errors at each layer, for logging/debugging.
    """

    top_down_precision: dict[str, float]
    hierarchical_error: float
    heat_cost: float
    layer_errors: dict[int, dict[str, float]] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# Main class                                                          #
# ------------------------------------------------------------------ #

class PredictiveHierarchy:
    """Three-level predictive coding hierarchy.

    Maintains living belief states at all three levels and propagates
    errors/predictions on every call to ``update()``.

    Parameters
    ----------
    homeostasis:
        Optional ``HomeostasisAdapter``.  When provided, the L2
        extrapolation uses adapted (experience-driven) setpoints rather
        than the compile-time ``_PREFRONTAL_SETPOINTS`` constants.  This
        lets the hierarchy's top-down prior drift slowly in sync with the
        organism's long-run vital observations.
    """

    def __init__(
        self,
        homeostasis: "HomeostasisAdapter | None" = None,
    ) -> None:
        self._homeostasis = homeostasis
        # L0 belief — brainstem: predictions = current raw vitals (no model)
        self._l0 = LayerBelief(
            level=0,
            predictions=dict.fromkeys(_VITALS, 50.0),
            precision=1.0,
        )
        # L1 belief — limbic: predictions = running model of what L0 should show
        self._l1 = LayerBelief(
            level=1,
            predictions=dict.fromkeys(_VITALS, 50.0),
            precision=_L1_BASE_PRECISION,
        )
        # L2 belief — prefrontal: predictions = desired setpoints + trend model
        self._l2 = LayerBelief(
            level=2,
            predictions=dict(_PREFRONTAL_SETPOINTS),
            precision=_L2_BASE_PRECISION,
        )

        # L2 trend model: rolling average of recent vital deltas (for extrapolation)
        self._trend_history: deque[dict[str, float]] = deque(maxlen=6)
        self._prev_vitals: dict[str, float] | None = None

    # ------------------------------------------------------------------ #
    # Main interface                                                       #
    # ------------------------------------------------------------------ #

    def update(
        self,
        state: MetabolicState,
        limbic_signal: LimbicSignal | None = None,
        l1_precision: float | None = None,
        l2_precision: float | None = None,
    ) -> HierarchySignal:
        """Propagate one tick of hierarchical predictive coding.

        Parameters
        ----------
        state:
            Current metabolic state (read-only here; costs applied externally).
        limbic_signal:
            Optional LimbicSignal from the current tick.  Used to inform
            the L1 precision (threat = higher precision at limbic layer).
        l1_precision:
            Optional override for L1 precision (from ThalamusGate).
        l2_precision:
            Optional override for L2 precision (from ThalamusGate).

        Returns
        -------
        HierarchySignal
        """
        raw_vitals = self._extract_vitals(state)

        # Update trend history
        if self._prev_vitals is not None:
            delta = {v: raw_vitals[v] - self._prev_vitals[v] for v in _VITALS}
            self._trend_history.append(delta)
        self._prev_vitals = dict(raw_vitals)

        # --- Override precisions from thalamic gating if provided
        if l1_precision is not None:
            self._l1.precision = l1_precision
        if l2_precision is not None:
            self._l2.precision = l2_precision

        # --- Amygdala threat boosts L1 precision
        if limbic_signal is not None and limbic_signal.threat_level > 0.3:
            threat_boost = limbic_signal.threat_level * 0.6
            self._l1.precision = min(4.0, self._l1.precision + threat_boost)

        # ============================================================
        # Bottom-up pass: errors propagate upward
        # ============================================================

        # L0 → L1: brainstem sends raw vitals; limbic computes errors
        l1_errors = self._compute_errors(
            observed=raw_vitals,
            predicted=self._l1.predictions,
            precision=self._l1.precision,
        )
        self._l1.errors = l1_errors

        # L1 updates its model toward observations (fast learning rate)
        for v in _VITALS:
            self._l1.predictions[v] += _L1_LEARNING_RATE * l1_errors.get(v, 0.0)

        # Weighted error sent upward to L2 — precision-gated
        l1_to_l2 = {
            v: self._l1.precision * l1_errors.get(v, 0.0)
            for v in _VITALS
        }

        # L2 receives weighted error from L1; updates its priors (slow)
        l2_errors = self._compute_errors(
            observed={v: self._l1.predictions[v] for v in _VITALS},
            predicted=self._l2.predictions,
            precision=self._l2.precision,
        )
        self._l2.errors = l2_errors

        for v in _VITALS:
            self._l2.predictions[v] += _L2_LEARNING_RATE * l2_errors.get(v, 0.0)

        # ============================================================
        # Top-down pass: predictions propagate downward
        # ============================================================

        # L2 generates extrapolated predictions for what L1 *should* see
        l2_trend_pred = self._extrapolate_l2_predictions()

        # L1 is partially pulled toward L2's predictions (top-down prior)
        for v in _VITALS:
            self._l1.predictions[v] = (
                (1.0 - _TOP_DOWN_PULL) * self._l1.predictions[v]
                + _TOP_DOWN_PULL * l2_trend_pred[v]
            )

        # ============================================================
        # Compute output signal
        # ============================================================

        # Top-down precision: per-vital boost where L2 prediction strongly
        # deviates from setpoint (prefrontal "cares more" about those dims)
        top_down_precision = self._compute_top_down_precision()

        # Hierarchical error scalar for EFE penalty
        l2_total_error = sum(abs(e) for e in l2_errors.values())
        hier_efe_penalty = min(
            _MAX_HIER_EFE_PENALTY,
            l2_total_error * self._l2.precision * 0.05,
        )

        # Heat cost: proportional to weighted error magnitude
        heat_cost = l2_total_error * _ERROR_HEAT_SCALE * self._l2.precision

        return HierarchySignal(
            top_down_precision=top_down_precision,
            hierarchical_error=hier_efe_penalty,
            heat_cost=heat_cost,
            layer_errors={1: dict(l1_errors), 2: dict(l2_errors)},
        )

    # ------------------------------------------------------------------ #
    # Layer accessors                                                      #
    # ------------------------------------------------------------------ #

    def l2_predictions(self) -> dict[str, float]:
        """Return current prefrontal-layer predictions (read-only copy)."""
        return dict(self._l2.predictions)

    def l1_errors(self) -> dict[str, float]:
        """Return current limbic-layer prediction errors (read-only copy)."""
        return dict(self._l1.errors)

    def status(self) -> dict:
        """Return a summary dict for logging."""
        return {
            "l1_precision": self._l1.precision,
            "l2_precision": self._l2.precision,
            "l1_errors": {v: round(e, 3) for v, e in self._l1.errors.items()},
            "l2_predictions": {v: round(p, 2) for v, p in self._l2.predictions.items()},
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_vitals(state: MetabolicState) -> dict[str, float]:
        return {
            "energy": state.energy,
            "heat": state.heat,
            "waste": state.waste,
            "integrity": state.integrity,
            "stability": state.stability,
        }

    @staticmethod
    def _compute_errors(
        observed: dict[str, float],
        predicted: dict[str, float],
        precision: float,
    ) -> dict[str, float]:
        """Compute precision-scaled prediction errors (observed − predicted)."""
        return {
            v: precision * (observed.get(v, predicted.get(v, 0.0)) - predicted.get(v, 0.0))
            for v in _VITALS
        }

    def _extrapolate_l2_predictions(self) -> dict[str, float]:
        """Extrapolate L2 predictions using the smoothed trend.

        If there is no trend history, returns current L2 predictions.
        Otherwise extrapolates one step forward using the exponentially-
        smoothed mean delta, biased toward the homeostatic target.

        When a ``HomeostasisAdapter`` is attached, the pull target uses
        adapted (experience-driven) setpoints instead of the compile-time
        constants, allowing the hierarchy's top-down prior to drift slowly
        in line with the organism's long-run vital observations.
        """
        # Resolve the setpoints to pull toward (adapted or original)
        pull_targets = (
            self._homeostasis.adapted_setpoints()
            if self._homeostasis is not None
            else dict(_PREFRONTAL_SETPOINTS)
        )

        if not self._trend_history:
            return dict(self._l2.predictions)

        # Exponentially-weighted mean delta (more recent = more weight)
        n = len(self._trend_history)
        weights = [0.9 ** (n - 1 - i) for i in range(n)]
        total_w = sum(weights)
        smoothed: dict[str, float] = dict.fromkeys(_VITALS, 0.0)
        for w, delta in zip(weights, self._trend_history):
            for v in _VITALS:
                smoothed[v] += w * delta.get(v, 0.0)
        smoothed = {v: smoothed[v] / total_w for v in _VITALS}

        # One-step extrapolation from current L2 predictions
        extrapolated: dict[str, float] = {}
        for v in _VITALS:
            raw = self._l2.predictions[v] + smoothed[v]
            # Soft homeostatic pull: nudge toward adapted setpoint.
            # pull_targets always contains all vitals (guaranteed by
            # HomeostasisAdapter.adapted_setpoints() contract).
            sp = pull_targets[v]
            pull = 0.05 * (sp - raw)
            extrapolated[v] = raw + pull

        return extrapolated

    def _compute_top_down_precision(self) -> dict[str, float]:
        """Derive per-vital precision boosts from L2 deviation from setpoints.

        Where the prefrontal layer's beliefs are furthest from the desired
        setpoint, attention (precision) should be amplified — the organism
        is most surprised about those dimensions.
        """
        result: dict[str, float] = {}
        for v in _VITALS:
            pred = self._l2.predictions.get(v, _PREFRONTAL_SETPOINTS[v])
            sp = _PREFRONTAL_SETPOINTS[v]
            rng = max(1.0, abs(sp)) if sp != 0 else 50.0
            # Normalised deviation 0-1
            dev = min(1.0, abs(pred - sp) / rng)
            # Precision boost: 0 at setpoint, up to +1.5 at maximum deviation
            result[v] = 1.0 + dev * 1.5 * self._l2.precision
        return result
