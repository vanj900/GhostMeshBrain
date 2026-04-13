"""Active inference engine — action proposal, EFE computation, selection.

Core operations
---------------
active_inference_step(state, proposals)
    Given the current MetabolicState and a list of action proposals, predict
    the metabolic deltas for each and select the one with lowest Expected Free
    Energy (EFE).

compute_efe(state, delta)
    Compute scalar EFE for a single predicted delta.  Lower is better.
    EFE = precision-weighted prediction error on survival variables
          + expected complexity cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
class InferenceResult:
    """Output of active_inference_step."""

    selected: ActionProposal
    efe_scores: dict[str, float]        # name → EFE value (lower = better)
    reasoning: str = ""


# ---------------------------------------------------------------------- #
# EFE computation                                                          #
# ---------------------------------------------------------------------- #

# Precision (inverse variance) weights for each vital — higher means the
# organism "cares more" about surprise in that dimension.
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


def compute_efe(state: MetabolicState, predicted_delta: dict[str, float]) -> float:
    """Compute Expected Free Energy for a predicted metabolic delta.

    EFE = Σ precision_i * (predicted_post_i - setpoint_i)² + complexity_cost

    Lower EFE → action is preferred by the min-surprise drive.

    Parameters
    ----------
    state:
        Current metabolic state (pre-action).
    predicted_delta:
        Mapping of vital-name → Δvalue expected after the action.

    Returns
    -------
    float
        EFE score (non-negative; lower is better).
    """
    current = {
        "energy": state.energy,
        "heat": state.heat,
        "waste": state.waste,
        "integrity": state.integrity,
        "stability": state.stability,
    }

    efe = 0.0
    for vital, setpoint in _SETPOINT.items():
        delta = predicted_delta.get(vital, 0.0)
        post = current[vital] + delta
        precision = _PRECISION.get(vital, 1.0)
        efe += precision * (post - setpoint) ** 2

    # Complexity cost: penalise large absolute deltas (metabolic effort)
    complexity = sum(abs(v) for v in predicted_delta.values()) * 0.05
    efe += complexity

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
# Active inference step                                                    #
# ---------------------------------------------------------------------- #


def active_inference_step(
    state: MetabolicState,
    proposals: list[ActionProposal],
) -> InferenceResult:
    """Select the action with the lowest EFE from a list of proposals.

    Parameters
    ----------
    state:
        Current MetabolicState.
    proposals:
        3-5 candidate actions with predicted metabolic deltas.

    Returns
    -------
    InferenceResult
        The selected action and per-proposal EFE scores.
    """
    if not proposals:
        raise ValueError("active_inference_step requires at least one proposal")

    scores: dict[str, float] = {}
    for p in proposals:
        # Adjust predicted delta for the energy cost of the action itself
        delta = dict(p.predicted_delta)
        delta["energy"] = delta.get("energy", 0.0) - p.cost_energy
        scores[p.name] = compute_efe(state, delta)

    best_name = min(scores, key=lambda k: scores[k])
    selected = next(p for p in proposals if p.name == best_name)

    reasoning_parts = [
        f"{p.name}: EFE={scores[p.name]:.2f}" for p in proposals
    ]
    reasoning = "Min-EFE selection — " + ", ".join(reasoning_parts)

    return InferenceResult(
        selected=selected,
        efe_scores=scores,
        reasoning=reasoning,
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
