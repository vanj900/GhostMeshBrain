"""Thalamic precision routing gate — Phase 3.

The thalamus is the brain's signal-routing hub.  In GhostMesh it acts as
a dynamic gate between hierarchical layers, controlling what fraction of
the prediction-error signal passes upward and how strongly top-down
predictions are enforced downward.

Biological analogue
-------------------
The real thalamus modulates cortical gain — under focused attention it
*amplifies* specific sensory channels; under sedation or dissociation it
*suppresses* them.  In ADHD this gate is dysregulated (impulsive broadcast
or complete inattention).  In PTSD, hyper-salience of threat channels
co-exists with suppression of social/regulatory channels.

Implementation
--------------
ThalamusGate computes a ``GateReport`` each tick based on:

- Current metabolic free energy (global stress)
- Amygdala threat level (survival urgency)
- Affective valence (positive vs. negative)
- PrecisionEngine regime (dormant / sweet_spot / overload)

The gate produces:

l1_precision
    Precision scalar for the limbic layer (scales bottom-up errors from L0).
    High under threat; suppressed under overload-dissociation.

l2_precision
    Precision scalar for the prefrontal layer (scales bottom-up errors from L1).
    Highest in sweet-spot; reduced under threat (prefrontal offline when scared)
    and under overload.

channel_weights
    Per-vital routing coefficients in [0, 1].  Under threat, survival channels
    (energy, heat, integrity) are opened; exploratory channels (stability, waste
    in non-critical zone) are narrowed.  Under dissociation, all channels damp.

Usage
-----
    gate = ThalamusGate()

    # Each heartbeat (after limbic, before hierarchy.update):
    gate_report = gate.route(state, limbic_signal, precision_regime)

    # Pass precision scalars into hierarchy.update():
    hierarchy.update(state, limbic_signal,
                     l1_precision=gate_report.l1_precision,
                     l2_precision=gate_report.l2_precision)

    # Pass channel_weights into _decide() for per-vital precision weighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.limbic import LimbicSignal


# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #

# Free-energy thresholds
_FE_DORMANT_MAX: float = 8.0      # below this: low-arousal / dormant
_FE_SWEET_MIN: float = 8.0
_FE_SWEET_MAX: float = 45.0       # sweet-spot band
_FE_OVERLOAD_MIN: float = 45.0    # above this: overload / potential dissociation
_FE_DISSOCIATION_MIN: float = 70.0  # above this: dissociation gate dampens all channels

# Threat threshold at which the gate shifts from balanced to survival-focused
_THREAT_HIGH: float = 0.5

# Survival vs. exploratory vital groups
_SURVIVAL_VITALS: frozenset[str] = frozenset({"energy", "heat", "integrity"})
_EXPLORATORY_VITALS: frozenset[str] = frozenset({"stability", "waste"})

# Base precision scalars by regime
_L1_PRECISION_DORMANT: float = 0.7
_L1_PRECISION_SWEET: float = 1.2
_L1_PRECISION_OVERLOAD: float = 1.5   # limbic stays hot under overload
_L1_PRECISION_DISSOCIATION: float = 0.9

_L2_PRECISION_DORMANT: float = 0.6
_L2_PRECISION_SWEET: float = 1.4     # prefrontal is most active in sweet-spot
_L2_PRECISION_OVERLOAD: float = 0.9  # prefrontal goes offline a bit under stress
_L2_PRECISION_DISSOCIATION: float = 0.5

# Survival channel boost under threat
_SURVIVAL_BOOST: float = 0.5     # additive boost to survival vital weights
# Exploratory channel suppression under threat
_EXPLORE_SUPPRESS: float = 0.35  # multiplicative dampener on exploratory vitals


# ------------------------------------------------------------------ #
# Data types                                                          #
# ------------------------------------------------------------------ #

@dataclass
class GateReport:
    """Output of a single ThalamusGate routing step."""

    l1_precision: float                    # Limbic layer precision scalar
    l2_precision: float                    # Prefrontal layer precision scalar
    channel_weights: dict[str, float]      # Per-vital routing weights in [0,1]
    regime: str                            # "dormant" | "sweet_spot" | "overload" | "dissociation"
    threat_level: float                    # Amygdala threat at this tick
    affect: float                          # Affective valence at this tick
    # Whether the gate suppressed exploratory channels this tick
    exploratory_suppressed: bool = False


# ------------------------------------------------------------------ #
# Main class                                                          #
# ------------------------------------------------------------------ #

class ThalamusGate:
    """Dynamic precision router between hierarchical processing layers.

    Each call to ``route()`` inspects the metabolic state and threat signals
    and returns a ``GateReport`` with layer-specific precision scalars and
    per-vital channel weights for the current tick.
    """

    def __init__(self) -> None:
        self._last_report: GateReport | None = None

    # ------------------------------------------------------------------ #
    # Main interface                                                       #
    # ------------------------------------------------------------------ #

    def route(
        self,
        state: MetabolicState,
        limbic_signal: LimbicSignal | None = None,
        precision_regime: str = "sweet_spot",
    ) -> GateReport:
        """Compute routing gate parameters for the current tick.

        Parameters
        ----------
        state:
            Current metabolic state (read-only).
        limbic_signal:
            Optional limbic signal carrying threat level and affect.
        precision_regime:
            Current PrecisionEngine regime string ("dormant", "sweet_spot",
            "overload").

        Returns
        -------
        GateReport
        """
        fe = state.free_energy_estimate()
        affect = state.affect
        threat = limbic_signal.threat_level if limbic_signal else 0.0

        # Determine regime
        if fe >= _FE_DISSOCIATION_MIN:
            regime = "dissociation"
        elif fe >= _FE_OVERLOAD_MIN or precision_regime == "overload":
            regime = "overload"
        elif fe <= _FE_DORMANT_MAX or precision_regime == "dormant":
            regime = "dormant"
        else:
            regime = "sweet_spot"

        # Base precision scalars
        l1_prec = self._l1_precision_for_regime(regime)
        l2_prec = self._l2_precision_for_regime(regime)

        # Threat modulation: high threat boosts L1 (survival-focused attention),
        # suppresses L2 (prefrontal overridden by fight/flight).
        threat_range = 1.0 - _THREAT_HIGH
        if threat >= _THREAT_HIGH and threat_range > 0.0:
            threat_excess = min(1.0, (threat - _THREAT_HIGH) / threat_range)
            l1_prec = min(4.0, l1_prec + threat_excess * 1.2)
            l2_prec = max(0.3, l2_prec - threat_excess * 0.4)

        # Positive affect mildly boosts L2 (prefrontal more active when content)
        if affect > 0.3:
            l2_prec = min(3.0, l2_prec + (affect - 0.3) * 0.4)

        # Per-vital channel weights
        channel_weights, exploratory_suppressed = self._compute_channel_weights(
            regime=regime,
            threat=threat,
            affect=affect,
        )

        report = GateReport(
            l1_precision=round(l1_prec, 4),
            l2_precision=round(l2_prec, 4),
            channel_weights=channel_weights,
            regime=regime,
            threat_level=threat,
            affect=affect,
            exploratory_suppressed=exploratory_suppressed,
        )
        self._last_report = report
        return report

    @property
    def last_report(self) -> GateReport | None:
        """Most recent gate report (None before first call to route())."""
        return self._last_report

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _l1_precision_for_regime(regime: str) -> float:
        return {
            "dormant": _L1_PRECISION_DORMANT,
            "sweet_spot": _L1_PRECISION_SWEET,
            "overload": _L1_PRECISION_OVERLOAD,
            "dissociation": _L1_PRECISION_DISSOCIATION,
        }.get(regime, _L1_PRECISION_SWEET)

    @staticmethod
    def _l2_precision_for_regime(regime: str) -> float:
        return {
            "dormant": _L2_PRECISION_DORMANT,
            "sweet_spot": _L2_PRECISION_SWEET,
            "overload": _L2_PRECISION_OVERLOAD,
            "dissociation": _L2_PRECISION_DISSOCIATION,
        }.get(regime, _L2_PRECISION_SWEET)

    @staticmethod
    def _compute_channel_weights(
        regime: str,
        threat: float,
        affect: float,
    ) -> tuple[dict[str, float], bool]:
        """Compute per-vital routing weights.

        Returns (channel_weights, exploratory_suppressed).
        """
        # Start at balanced base
        weights: dict[str, float] = {
            "energy": 0.8,
            "heat": 0.8,
            "waste": 0.6,
            "integrity": 0.8,
            "stability": 0.6,
        }

        exploratory_suppressed = False

        if regime == "dormant":
            # All channels slightly open — organism is at rest
            weights = {k: v * 0.75 for k, v in weights.items()}

        elif regime == "sweet_spot":
            # Full open; positive affect boosts exploratory channels
            if affect > 0.2:
                explore_boost = min(0.3, (affect - 0.2) * 0.5)
                for v in _EXPLORATORY_VITALS:
                    weights[v] = min(1.0, weights[v] + explore_boost)

        elif regime in ("overload", "dissociation"):
            # Survival channels stay open; exploratory channels damp
            for v in _SURVIVAL_VITALS:
                weights[v] = min(1.0, weights[v] * 1.1)
            for v in _EXPLORATORY_VITALS:
                weights[v] = max(0.1, weights[v] * 0.4)
            exploratory_suppressed = True

        # Threat override: open survival channels harder
        if threat >= _THREAT_HIGH:
            threat_ratio = min(1.0, (threat - _THREAT_HIGH) / (1.0 - _THREAT_HIGH))
            for v in _SURVIVAL_VITALS:
                weights[v] = min(1.0, weights[v] + threat_ratio * _SURVIVAL_BOOST)
            for v in _EXPLORATORY_VITALS:
                weights[v] = max(0.05, weights[v] * (1.0 - threat_ratio * _EXPLORE_SUPPRESS))
            exploratory_suppressed = True

        # Clamp all weights
        weights = {k: max(0.0, min(1.0, v)) for k, v in weights.items()}
        return weights, exploratory_suppressed
