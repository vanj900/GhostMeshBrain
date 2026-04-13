"""Active inference engine — action proposal, EFE computation, selection.

Core operations
---------------
active_inference_step(state, proposals)
    Given the current MetabolicState and a list of action proposals, predict
    the metabolic deltas for each and select the one with lowest Expected Free
    Energy (EFE).

compute_efe(state, delta, precision_weights)
    Compute scalar EFE for a single predicted delta.  Lower is better.
    EFE = accuracy term (precision-weighted prediction error on survival variables)
        + complexity term (KL divergence proxy — cost of shifting priors)

compute_inference_cost(proposals, precision_weights, compute_load)
    Return the real metabolic cost of running the inference process itself:
    evaluating proposals and updating beliefs.  Charged proportionally to
    planning depth (n proposals), prior-shift magnitude (KL proxy), and the
    sharpness of precision weighting.

    Calibrated so that at normal load a 5-proposal DECIDE step costs
    ~0.15 energy + ~0.03 heat — roughly 1.25× the passive decay rate.
    This is enough to shape behavior without causing rapid death.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from thermodynamic_agency.core.metabolic import MetabolicState


@dataclass
class ActionProposal:
    """A candidate action together with its predicted metabolic impact."""

    name: str
    description: str
    predicted_delta: dict[str, float]   # keys: energy, heat, waste, integrity, stability
    cost_energy: float = 1.0            # base energy cost to execute
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.metadata = self.metadata or {}


@dataclass
class InferenceCost:
    """Real metabolic cost paid to run the inference process itself.

    Separate from the cost of the *selected* action (cost_energy).  This
    reflects the thermodynamic price of evaluating proposals and updating
    belief priors — i.e., the work of minimising variational free energy.
    """

    energy_cost: float
    heat_cost: float
    kl_complexity: float    # KL divergence proxy (prior-shift magnitude)
    precision_used: float   # mean precision weight applied during scoring


@dataclass
class InferenceResult:
    """Output of active_inference_step."""

    selected: ActionProposal
    efe_scores: dict[str, float]        # name → EFE value (lower = better)
    reasoning: str = ""
    inference_cost: InferenceCost | None = None   # metabolic cost of this inference pass


# ---------------------------------------------------------------------- #
# EFE computation                                                          #
# ---------------------------------------------------------------------- #

# Default precision (inverse variance) weights — can be overridden by
# PrecisionEngine.  Higher values mean the organism "cares more" about
# surprise in that dimension.
_PRECISION: dict[str, float] = {
    "energy": 2.0,       # critical — starvation kills fast
    "heat": 1.5,         # high — overheating kills
    "waste": 1.0,        # moderate — constipation is slow
    "integrity": 1.8,    # high — coherence is near-existential
    "stability": 1.4,    # high — entropy is existential
}

# "Desired" / setpoint values the organism wants to maintain
_SETPOINT: dict[str, float] = {
    "energy": 80.0,
    "heat": 20.0,
    "waste": 10.0,
    "integrity": 85.0,
    "stability": 80.0,
}


def compute_efe(
    state: MetabolicState,
    predicted_delta: dict[str, float],
    precision_weights: dict[str, float] | None = None,
) -> float:
    """Compute Expected Free Energy for a predicted metabolic delta.

    EFE = accuracy + complexity

    accuracy  = Σ precision_i * (predicted_post_i − setpoint_i)²
                (prediction error — how far from desired state)
    complexity = Σ |Δ_i| * 0.05
                (cost of prior shift — how much beliefs must change)

    Both terms are grounded in the Friston FEP formulation:
    minimising EFE = minimising expected surprise + complexity cost.

    Parameters
    ----------
    state:
        Current metabolic state (pre-action).
    predicted_delta:
        Mapping of vital-name → Δvalue expected after the action.
    precision_weights:
        Optional override for per-vital precision.  Defaults to _PRECISION.

    Returns
    -------
    float
        EFE score (non-negative; lower is better).
    """
    precision = precision_weights if precision_weights is not None else _PRECISION

    current = {
        "energy": state.energy,
        "heat": state.heat,
        "waste": state.waste,
        "integrity": state.integrity,
        "stability": state.stability,
    }

    # Accuracy term — precision-weighted squared prediction error
    accuracy = 0.0
    for vital, setpoint in _SETPOINT.items():
        delta = predicted_delta.get(vital, 0.0)
        post = current[vital] + delta
        p = precision.get(vital, 1.0)
        accuracy += p * (post - setpoint) ** 2

    # Complexity term — penalise large prior shifts (metabolic effort of updating)
    complexity = sum(abs(v) for v in predicted_delta.values()) * 0.05
    efe = accuracy + complexity

    # Death proximity penalty — amplifies EFE near lethal thresholds
    from thermodynamic_agency.core.metabolic import (
        ENERGY_DEATH_THRESHOLD,
        THERMAL_DEATH_THRESHOLD,
        INTEGRITY_DEATH_THRESHOLD,
        STABILITY_DEATH_THRESHOLD,
    )
    post_energy = current["energy"] + predicted_delta.get("energy", 0.0)
    post_heat = current["heat"] + predicted_delta.get("heat", 0.0)
    post_integrity = current["integrity"] + predicted_delta.get("integrity", 0.0)
    post_stability = current["stability"] + predicted_delta.get("stability", 0.0)

    death_margin = (
        max(0.0, 5.0 - (post_energy - ENERGY_DEATH_THRESHOLD))
        + max(0.0, 5.0 - (THERMAL_DEATH_THRESHOLD - post_heat))
        + max(0.0, 5.0 - (post_integrity - INTEGRITY_DEATH_THRESHOLD))
        + max(0.0, 5.0 - (post_stability - STABILITY_DEATH_THRESHOLD))
    )
    efe += death_margin * 50.0

    return efe


# ---------------------------------------------------------------------- #
# Inference cost                                                           #
# ---------------------------------------------------------------------- #

# GHOST_COMPUTE_LOAD default — overridden by env var in pulse.py
_DEFAULT_COMPUTE_LOAD: float = 1.0


def compute_inference_cost(
    proposals: list[ActionProposal],
    precision_weights: dict[str, float] | None = None,
    compute_load: float = _DEFAULT_COMPUTE_LOAD,
) -> InferenceCost:
    """Calculate the real metabolic cost of running active inference.

    This is the cost of the *cognitive work* of evaluating proposals and
    updating precision-weighted beliefs — not the cost of the selected action.

    Calibration (compute_load=1.0, 5 proposals, typical deltas):
        energy_cost ≈ 0.14–0.18   (1.2–1.5× passive decay of 0.12)
        heat_cost   ≈ 0.02–0.04

    At double compute_load (heavy LLM use): costs double.
    At 1 proposal (trivial decision): costs drop to ~0.07 energy.

    Components
    ----------
    base          : fixed overhead of running one inference cycle
    kl_complexity : KL divergence proxy — mean |Δ| across proposals,
                    scaled small so a single large delta ≠ instant death
    precision_tax : sharper attention (higher mean precision) costs more,
                    because sharpening prediction-error weighting is work
    depth_cost    : more proposals = deeper planning = more computation
    """
    precision = precision_weights if precision_weights is not None else _PRECISION
    mean_precision = sum(precision.values()) / len(precision)

    # KL complexity proxy: mean total |Δ| across all proposals
    if proposals:
        mean_delta_magnitude = sum(
            sum(abs(v) for v in p.predicted_delta.values())
            for p in proposals
        ) / len(proposals)
    else:
        mean_delta_magnitude = 0.0

    n = len(proposals)

    # Energy cost: base + KL proxy + per-proposal depth cost
    # Calibrated: 5 proposals, mean_delta≈22 → ~0.15 energy at load=1
    kl_cost = mean_delta_magnitude * 0.003
    depth_cost = n * 0.004
    base_energy = 0.05
    energy_cost = compute_load * (base_energy + kl_cost + depth_cost)

    # Heat cost: reflects precision sharpening work
    # Calibrated: mean_precision≈1.5, 5 props → ~0.03 heat at load=1
    precision_tax = mean_precision * 0.008
    heat_cost = compute_load * (precision_tax + n * 0.002)

    return InferenceCost(
        energy_cost=energy_cost,
        heat_cost=heat_cost,
        kl_complexity=mean_delta_magnitude,
        precision_used=mean_precision,
    )


# ---------------------------------------------------------------------- #
# Active inference step                                                    #
# ---------------------------------------------------------------------- #


def active_inference_step(
    state: MetabolicState,
    proposals: list[ActionProposal],
    precision_weights: dict[str, float] | None = None,
    compute_load: float = _DEFAULT_COMPUTE_LOAD,
) -> InferenceResult:
    """Select the action with the lowest EFE from a list of proposals.

    Charges the real metabolic cost of running inference to ``state``
    immediately.  The energy and heat cost of cognitive work is debited
    before the selected action is executed — mortal computation pays first.

    Parameters
    ----------
    state:
        Current MetabolicState — will have inference cost applied.
    proposals:
        3-5 candidate actions with predicted metabolic deltas.
    precision_weights:
        Optional per-vital precision overrides from PrecisionEngine.
    compute_load:
        Scalar computational burden (scales inference cost).

    Returns
    -------
    InferenceResult
        The selected action, per-proposal EFE scores, and the metabolic
        cost that was charged to state.
    """
    if not proposals:
        raise ValueError("active_inference_step requires at least one proposal")

    # Compute and immediately charge the cost of running inference itself
    cost = compute_inference_cost(proposals, precision_weights, compute_load)
    state.apply_action_feedback(
        delta_energy=-cost.energy_cost,
        delta_heat=cost.heat_cost,
    )

    scores: dict[str, float] = {}
    for p in proposals:
        # Adjust predicted delta for the energy cost of the action itself
        delta = dict(p.predicted_delta)
        delta["energy"] = delta.get("energy", 0.0) - p.cost_energy
        scores[p.name] = compute_efe(state, delta, precision_weights)

    best_name = min(scores, key=lambda k: scores[k])
    selected = next(p for p in proposals if p.name == best_name)

    reasoning_parts = [
        f"{p.name}: EFE={scores[p.name]:.2f}" for p in proposals
    ]
    reasoning = (
        f"Min-EFE selection — {', '.join(reasoning_parts)} "
        f"[inference cost: E={cost.energy_cost:.3f} H={cost.heat_cost:.3f} "
        f"KL={cost.kl_complexity:.1f} prec={cost.precision_used:.2f}]"
    )

    return InferenceResult(
        selected=selected,
        efe_scores=scores,
        reasoning=reasoning,
        inference_cost=cost,
    )


# ---------------------------------------------------------------------- #
# Built-in proposal generator (used when no LLM is available)            #
# ---------------------------------------------------------------------- #


def generate_default_proposals(state: MetabolicState) -> list[ActionProposal]:
    """Generate a small set of default proposals based on current vital signs.

    In a full deployment these are generated by an LLM (Ollama etc.).  This
    fallback ensures the inference loop always has something to work with.
    """
    proposals = [
        ActionProposal(
            name="idle",
            description="Do nothing; conserve energy and let vitals passively settle.",
            predicted_delta={"energy": -0.05, "heat": -0.5, "waste": 0.1},
            cost_energy=0.05,
        ),
        ActionProposal(
            name="reflect",
            description="Run a short self-reflection pass to improve integrity.",
            predicted_delta={
                "energy": -1.0,
                "heat": 0.5,
                "integrity": 3.0,
                "stability": 1.0,
            },
            cost_energy=1.0,
        ),
        ActionProposal(
            name="compress_context",
            description="Summarise recent context to reduce heat and waste.",
            predicted_delta={
                "energy": -2.0,
                "heat": -15.0,
                "waste": -20.0,
                "integrity": 1.0,
            },
            cost_energy=2.0,
        ),
        ActionProposal(
            name="forage_resources",
            description="Execute lightweight tool calls to replenish energy.",
            predicted_delta={"energy": 15.0, "heat": 2.0, "waste": 3.0},
            cost_energy=3.0,
        ),
        ActionProposal(
            name="dream",
            description="Enter dream/consolidation state; high integrity payoff.",
            predicted_delta={
                "energy": -3.0,
                "heat": -5.0,
                "waste": -10.0,
                "integrity": 8.0,
                "stability": 5.0,
            },
            cost_energy=3.0,
        ),
    ]
    return proposals

