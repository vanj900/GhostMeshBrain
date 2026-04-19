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

ForwardModel (Layer 3 — Cerebellum prediction)
    Maintains a rolling buffer of recent vital deltas and predicts the next
    N vital states.  A ``prediction_error_term()`` method compares its last
    forecasts to actual outcomes and returns a complexity penalty that is
    added to EFE, forcing the organism to get better at anticipating its own
    death risks.
"""

from __future__ import annotations

import math
from collections import deque
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
class EFEComponents:
    """Breakdown of the multi-step EFE for the selected action.

    Captured after active_inference_step() for phase-transition analysis.
    """

    accuracy: float = 0.0     # precision-weighted prediction-error term
    complexity: float = 0.0   # prior-shift (KL) complexity term
    risk: float = 0.0         # death-proximity risk penalty
    wear: float = 0.0         # allostatic-load wear penalty


@dataclass
class InferenceResult:
    """Output of active_inference_step."""

    selected: ActionProposal
    efe_scores: dict[str, float]        # name → EFE value (lower = better)
    reasoning: str = ""
    inference_cost: InferenceCost | None = None   # metabolic cost of this inference pass
    efe_components: EFEComponents | None = None   # breakdown of best-action EFE


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
# Multi-step EFE (Phase 1 — anticipatory free-energy minimisation)        #
# ---------------------------------------------------------------------- #

# Rollout horizon and discount factor
_HORIZON: int = 5
_GAMMA: float = 0.92

# Per-vital risk-penalty strength: applied when a vital is within its
# safety margin of the death threshold.  Higher λ = stronger deterrence.
_RISK_LAMBDA: dict[str, float] = {
    "energy": 3.0,
    "heat": 2.5,
    "waste": 1.5,
    "integrity": 2.8,
    "stability": 2.2,
}

# Safety margins: how far inside the death threshold to start penalising.
# e.g. energy margin=20 means the penalty starts when energy < 20.
_SAFETY_MARGIN: dict[str, float] = {
    "energy": 20.0,
    "heat": 15.0,
    "integrity": 20.0,
    "stability": 20.0,
}


def _decay_vitals_one_step(
    vitals: dict[str, float],
    allostatic_load: float = 0.0,
) -> dict[str, float]:
    """Apply one tick of passive decay to a vitals snapshot dict.

    Mirrors the decay equations in ``MetabolicState.tick()`` without the
    exception-raising, hormone proxies, or arousal gate.  Suitable for
    lightweight forward rollouts in EFE computation.
    """
    al_ratio = allostatic_load / 100.0
    waste = vitals["waste"] + 0.018 + 0.02 * al_ratio
    heat = vitals["heat"] + 0.1 * (1.0 + vitals["waste"] / 50.0) + 0.08 * al_ratio
    integrity = vitals["integrity"] * (1.0 - (heat / 120.0) * 0.01)
    stability = vitals["stability"] - 0.05
    energy = vitals["energy"] - 0.12
    return {
        "energy": max(energy, -1.0),
        "heat": min(heat, 110.0),
        "waste": max(waste, 0.0),
        "integrity": max(integrity, 0.0),
        "stability": max(stability, -1.0),
    }


def _accuracy_term(
    vitals: dict[str, float],
    precision: dict[str, float],
    setpoints: dict[str, float] | None = None,
) -> float:
    """Precision-weighted squared deviation from setpoints (accuracy term).

    Parameters
    ----------
    vitals:
        Current (simulated) vital values.
    precision:
        Per-vital precision (inverse variance) weights.
    setpoints:
        Optional adapted setpoints from ``HomeostasisAdapter``.  When
        provided these replace the module-level ``_SETPOINT`` constants,
        allowing the organism's notion of "normal" to drift slowly over
        time.  Defaults to ``_SETPOINT`` when ``None``.
    """
    sp = setpoints if setpoints is not None else _SETPOINT
    total = 0.0
    for vital, setpoint in sp.items():
        p = precision.get(vital, 1.0)
        total += p * (vitals.get(vital, setpoint) - setpoint) ** 2
    return total


def _risk_term(vitals: dict[str, float]) -> float:
    """Smooth threshold-margin penalty — rises quadratically inside safety margin.

    Unlike the hard binary death check, this creates a continuous gradient that
    repels the organism from dangerous regions even when not yet in crisis.
    """
    from thermodynamic_agency.core.metabolic import (
        ENERGY_DEATH_THRESHOLD,
        THERMAL_DEATH_THRESHOLD,
        INTEGRITY_DEATH_THRESHOLD,
        STABILITY_DEATH_THRESHOLD,
    )

    d_energy = vitals["energy"] - ENERGY_DEATH_THRESHOLD
    d_heat = THERMAL_DEATH_THRESHOLD - vitals["heat"]
    d_integrity = vitals["integrity"] - INTEGRITY_DEATH_THRESHOLD
    d_stability = vitals["stability"] - STABILITY_DEATH_THRESHOLD

    m_energy = _SAFETY_MARGIN.get("energy", 20.0)
    m_heat = _SAFETY_MARGIN.get("heat", 15.0)
    m_integrity = _SAFETY_MARGIN.get("integrity", 20.0)
    m_stability = _SAFETY_MARGIN.get("stability", 20.0)

    return (
        _RISK_LAMBDA["energy"] * max(0.0, m_energy - d_energy) ** 2
        + _RISK_LAMBDA["heat"] * max(0.0, m_heat - d_heat) ** 2
        + _RISK_LAMBDA["integrity"] * max(0.0, m_integrity - d_integrity) ** 2
        + _RISK_LAMBDA["stability"] * max(0.0, m_stability - d_stability) ** 2
    )


def _wear_term(allostatic_load: float) -> float:
    """Latent damage penalty from accumulated allostatic load (Phase 2 link)."""
    return 5.0 * (allostatic_load / 100.0) ** 2


def compute_multistep_efe(
    state: MetabolicState,
    predicted_delta: dict[str, float],
    precision_weights: dict[str, float] | None = None,
    horizon: int = _HORIZON,
    gamma: float = _GAMMA,
    setpoints: dict[str, float] | None = None,
    return_components: bool = False,
) -> "float | tuple[float, EFEComponents]":
    """Compute multi-step Expected Free Energy via a short forward rollout.

    Instead of scoring only the immediate post-action state, this rolls
    forward ``horizon`` ticks of passive dynamics and accumulates the
    discounted sum of accuracy + complexity + risk + wear.

    EFE(a) = Σ_{t=1}^{H} γ^{t-1} * (accuracy_t + complexity_t + risk_t + wear_t)

    Parameters
    ----------
    state:
        Current metabolic state (read-only; not mutated).
    predicted_delta:
        Mapping of vital-name → Δvalue expected immediately after the action.
    precision_weights:
        Optional per-vital precision overrides.  Defaults to ``_PRECISION``.
    horizon:
        Number of future ticks to simulate (default 5).
    gamma:
        Temporal discount factor (default 0.92; recent ticks weighted more).
    setpoints:
        Optional adapted setpoints from ``HomeostasisAdapter``.  When
        provided, the organism's long-run vital expectations are used in the
        accuracy term instead of the compile-time ``_SETPOINT`` constants.
        Defaults to ``_SETPOINT`` when ``None``.
    return_components:
        When ``True``, return ``(total_efe, EFEComponents)`` instead of just
        the scalar total.  Used by the CollapseProbe for phase-transition
        analysis.

    Returns
    -------
    float or tuple[float, EFEComponents]
        Multi-step EFE score (non-negative; lower is better).
        When *return_components* is ``True``, also returns a breakdown of
        the accumulated accuracy, complexity, risk, and wear terms.
    """
    precision = precision_weights if precision_weights is not None else _PRECISION
    allostatic_load = getattr(state, "allostatic_load", 0.0)

    # Start from post-action state
    vitals: dict[str, float] = {
        "energy": state.energy,
        "heat": state.heat,
        "waste": state.waste,
        "integrity": state.integrity,
        "stability": state.stability,
    }
    for k, v in predicted_delta.items():
        if k in vitals:
            vitals[k] = vitals[k] + v

    # Amortise the immediate actuation cost evenly across rollout ticks
    complexity_per_tick = sum(abs(v) for v in predicted_delta.values()) * 0.05 / max(1, horizon)

    total = 0.0
    total_accuracy = 0.0
    total_complexity = 0.0
    total_risk = 0.0
    total_wear = 0.0
    discount = 1.0
    for _ in range(1, horizon + 1):
        vitals = _decay_vitals_one_step(vitals, allostatic_load=allostatic_load)
        accuracy = _accuracy_term(vitals, precision, setpoints=setpoints)
        risk = _risk_term(vitals)
        wear = _wear_term(allostatic_load)
        total += discount * (accuracy + complexity_per_tick + risk + wear)
        total_accuracy += discount * accuracy
        total_complexity += discount * complexity_per_tick
        total_risk += discount * risk
        total_wear += discount * wear
        discount *= gamma

    if return_components:
        return total, EFEComponents(
            accuracy=total_accuracy,
            complexity=total_complexity,
            risk=total_risk,
            wear=total_wear,
        )
    return total

# GHOST_COMPUTE_LOAD default — overridden by env var in pulse.py
_DEFAULT_COMPUTE_LOAD: float = 1.0

# Inference cost calibration constants.
# These are tuned so that a 5-proposal DECIDE step at compute_load=1.0 costs
# ~0.13–0.15 energy and ~0.02–0.03 heat — roughly 1.1–1.25× the passive
# metabolic decay rate (0.12 E/tick), making cognitive work meaningfully
# expensive without causing rapid death.
_COST_KL_SCALE: float = 0.003       # energy cost per unit of mean delta magnitude
_COST_DEPTH_ENERGY: float = 0.004   # energy cost per proposal evaluated
_COST_BASE_ENERGY: float = 0.05     # fixed energy overhead per inference pass
_COST_PRECISION_HEAT: float = 0.008 # heat cost per unit of mean precision
_COST_DEPTH_HEAT: float = 0.002     # heat cost per proposal evaluated


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
    kl_cost = mean_delta_magnitude * _COST_KL_SCALE
    depth_cost = n * _COST_DEPTH_ENERGY
    energy_cost = compute_load * (_COST_BASE_ENERGY + kl_cost + depth_cost)

    # Heat cost: reflects precision sharpening work
    # Calibrated: mean_precision≈1.5, 5 props → ~0.03 heat at load=1
    precision_tax = mean_precision * _COST_PRECISION_HEAT
    heat_cost = compute_load * (precision_tax + n * _COST_DEPTH_HEAT)

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
    reward_discount: float = 0.0,
    setpoints: dict[str, float] | None = None,
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
    reward_discount:
        Fractional discount applied to all EFE scores (0–0.2).  Supplied by
        NucleusAccumbens when affect is positive — makes all policies
        relatively cheaper, biasing toward exploratory behaviour.
    setpoints:
        Optional adapted setpoints from ``HomeostasisAdapter``.  Passed
        through to ``compute_multistep_efe`` so that the organism's slowly
        drifting vital expectations are reflected in EFE scoring.

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
    best_delta: dict[str, float] = {}
    for p in proposals:
        # Adjust predicted delta for the energy cost of the action itself
        delta = dict(p.predicted_delta)
        delta["energy"] = delta.get("energy", 0.0) - p.cost_energy
        raw_efe = compute_multistep_efe(state, delta, precision_weights, setpoints=setpoints)
        # Apply reward discount from nucleus accumbens (positive affect bonus)
        scores[p.name] = raw_efe * (1.0 - max(0.0, min(0.2, reward_discount)))
        best_delta[p.name] = delta  # keep delta for component extraction

    best_name = min(scores, key=lambda k: scores[k])
    selected = next(p for p in proposals if p.name == best_name)

    # Capture EFE component breakdown for the selected action (used by CollapseProbe)
    _sel_delta = best_delta[best_name]
    # total_efe is already in scores[best_name]; we recompute only to get the breakdown
    _total_efe, efe_comps = compute_multistep_efe(  # type: ignore[misc]
        state, _sel_delta, precision_weights, setpoints=setpoints, return_components=True
    )

    reasoning_parts = [
        f"{p.name}: EFE={scores[p.name]:.2f}" for p in proposals
    ]
    discount_note = f" reward_discount={reward_discount:.3f}" if reward_discount > 0 else ""
    reasoning = (
        f"Min-EFE selection — {', '.join(reasoning_parts)} "
        f"[inference cost: E={cost.energy_cost:.3f} H={cost.heat_cost:.3f} "
        f"KL={cost.kl_complexity:.1f} prec={cost.precision_used:.2f}{discount_note}]"
    )

    return InferenceResult(
        selected=selected,
        efe_scores=scores,
        reasoning=reasoning,
        inference_cost=cost,
        efe_components=efe_comps,
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


# ---------------------------------------------------------------------- #
# Forward Model (Layer 3 — Cerebellum prediction)                        #
# ---------------------------------------------------------------------- #

# Number of ticks of vital history to keep for delta smoothing
_FORWARD_HISTORY: int = 8

# Number of future vital states to predict
_FORWARD_STEPS: int = 3

# Prediction error penalty weight in EFE — scales the complexity term added
# when forecasts were wrong.  Kept small so it shapes but doesn't dominate.
_PREDICTION_ERROR_WEIGHT: float = 0.04

# Vital names tracked by the forward model
_VITAL_NAMES: tuple[str, ...] = ("energy", "heat", "waste", "integrity", "stability")


@dataclass
class ForwardPrediction:
    """A set of N-step vital state predictions."""

    steps: list[dict[str, float]]   # each element is {vital: predicted_value}


class ForwardModel:
    """Simple cerebellum-style forward model for vital state prediction.

    Maintains a rolling buffer of recent vital readings and uses exponential
    smoothing of observed deltas to predict the next ``n_steps`` vital states.
    After each tick, ``update()`` is called with the actual new vitals and the
    prediction error against the last forecast is accumulated.

    The prediction error drives a complexity penalty on subsequent EFE scores
    via ``prediction_error_term()``.  When the organism consistently mis-
    forecasts its own vital trajectory it pays a higher cognitive cost —
    incentivising accurate internal models of its own metabolism.

    Usage
    -----
        fm = ForwardModel()

        # Each tick:
        pred_error = fm.prediction_error_term()  # penalty for last tick's miss
        # ... add pred_error to EFE during DECIDE ...
        fm.update(state)                         # record actual vitals

        # After a DECIDE step, get the next-tick forecast:
        forecast = fm.predict(state)
    """

    def __init__(
        self,
        history_size: int = _FORWARD_HISTORY,
        n_steps: int = _FORWARD_STEPS,
    ) -> None:
        self._history: deque[dict[str, float]] = deque(maxlen=history_size)
        self._n_steps = n_steps
        self._last_prediction: ForwardPrediction | None = None
        self._last_error: float = 0.0

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def update(self, state: MetabolicState) -> None:
        """Record the current vital state and compute prediction error.

        Call this once per tick, *after* the tick has completed, to update
        the model with the actual observed vitals.

        Parameters
        ----------
        state:
            Current (post-tick) metabolic state.
        """
        vitals = self._vitals(state)

        # Compute prediction error against last forecast (if we have one)
        if self._last_prediction and self._last_prediction.steps:
            predicted = self._last_prediction.steps[0]
            squared_errors = [
                (predicted.get(v, vitals[v]) - vitals[v]) ** 2
                for v in _VITAL_NAMES
            ]
            self._last_error = sum(squared_errors) / len(squared_errors)
        else:
            self._last_error = 0.0

        self._history.append(vitals)

    def predict(self, state: MetabolicState) -> ForwardPrediction:
        """Forecast the next ``n_steps`` vital states.

        Uses exponential smoothing of observed tick-to-tick deltas.  When the
        history buffer is empty, returns the current state unchanged.

        Parameters
        ----------
        state:
            Current metabolic state (used as the starting point).

        Returns
        -------
        ForwardPrediction
            Predicted vital-state dicts for the next n_steps ticks.
        """
        current = self._vitals(state)

        if len(self._history) < 2:
            # Not enough history: predict current state persists
            self._last_prediction = ForwardPrediction(
                steps=[dict(current) for _ in range(self._n_steps)]
            )
            return self._last_prediction

        # Exponentially-smoothed mean delta
        smoothed_delta = self._smoothed_delta()

        steps: list[dict[str, float]] = []
        prev = dict(current)
        for _ in range(self._n_steps):
            nxt = {v: prev[v] + smoothed_delta.get(v, 0.0) for v in _VITAL_NAMES}
            steps.append(nxt)
            prev = nxt

        self._last_prediction = ForwardPrediction(steps=steps)
        return self._last_prediction

    def prediction_error_term(self) -> float:
        """Return the EFE complexity penalty from last tick's prediction error.

        Should be called during a DECIDE step before calling ``update()``,
        so the penalty reflects the model's most recent forecast miss.

        Returns
        -------
        float
            Non-negative penalty to add to EFE scores.
        """
        return self._last_error * _PREDICTION_ERROR_WEIGHT

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _vitals(state: MetabolicState) -> dict[str, float]:
        return {
            "energy": state.energy,
            "heat": state.heat,
            "waste": state.waste,
            "integrity": state.integrity,
            "stability": state.stability,
        }

    def _smoothed_delta(self) -> dict[str, float]:
        """Compute exponentially-smoothed tick-to-tick deltas over history.

        Single forward pass builds per-step raw deltas; a separate backward
        pass applies exponential weights — avoids repeated dict lookups per vital.
        """
        history = list(self._history)
        if len(history) < 2:
            return {v: 0.0 for v in _VITAL_NAMES}

        # Pre-compute raw deltas in one forward pass
        raw_deltas: list[dict[str, float]] = [
            {v: history[i][v] - history[i - 1][v] for v in _VITAL_NAMES}
            for i in range(1, len(history))
        ]

        alpha = 0.3  # smoothing factor (higher = more recent weight)
        smoothed: dict[str, float] = {v: 0.0 for v in _VITAL_NAMES}
        weight_sum = 0.0
        w = 1.0

        # Iterate most-recent first (reversed raw_deltas)
        for delta in reversed(raw_deltas):
            for v in _VITAL_NAMES:
                smoothed[v] += w * delta[v]
            weight_sum += w
            w *= (1.0 - alpha)

        if weight_sum > 0:
            smoothed = {v: smoothed[v] / weight_sum for v in _VITAL_NAMES}

        return smoothed

