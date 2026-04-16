"""CounterfactualEngine — depth-first forward simulation for fear-based avoidance.

For each candidate ActionProposal the engine runs a lightweight DFS simulation:

  Depth 0:   Apply the action's predicted_delta (+ cost) to a copy of current
             vitals.  Real state is never touched.
  Depth 1…N: Apply one tick of passive decay per step, mirroring
             ``MetabolicState.tick()`` without exception raising.

Pruning rules
-------------
Hard prune  (depth ≤ ``hard_prune_depth``, default 2)
    Any vital crosses a death threshold → the branch is discarded immediately,
    flagged as a "lethal_zone", and ``terminal_risk`` is set to ``1.0``.
    This mimics the biological flinch / fear reflex — instant avoidance of
    anything that kills within two heartbeats, with no further deliberation.

Soft high-surprise  (depth > hard_prune_depth, or death at depth > 2)
    A vital breaches a safety margin but is not yet lethal → the branch
    continues but the depth is recorded in ``high_surprise_depths``.
    ``terminal_risk`` for surviving branches is proportional to the fraction of
    simulated steps spent inside a safety margin, weighted toward recent danger.

Metabolic cost
--------------
The engine charges a small energy + heat cost for each depth step *actually
simulated*.  Hard-pruned branches are extremely cheap — the organism pays
almost nothing to avoid them, mirroring the fact that biological fear is
metabolically inexpensive compared to deliberate planning.

Usage (inside ``_decide()`` in ``pulse.py``)
--------------------------------------------
    traces = counterfactual_engine.run_batch(state, safe_proposals)
    cf_energy, cf_heat = counterfactual_engine.compute_metabolic_cost(traces)
    state.apply_action_feedback(delta_energy=-cf_energy, delta_heat=cf_heat)

    # Adjust EFE scores by terminal risk before final selection
    for trace in traces:
        efe_scores[trace.proposal_name] += trace.terminal_risk * CF_RISK_WEIGHT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from thermodynamic_agency.core.metabolic import (
    MetabolicState,
    ENERGY_DEATH_THRESHOLD,
    THERMAL_DEATH_THRESHOLD,
    INTEGRITY_DEATH_THRESHOLD,
    STABILITY_DEATH_THRESHOLD,
)
from thermodynamic_agency.cognition.inference import (
    ActionProposal,
    _decay_vitals_one_step,
)

if TYPE_CHECKING:
    pass


# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

# Default look-ahead depth.
# 10 ticks = one full "breath" of the organism, long enough to catch
# slow-burn risks (stability drift, integrity erosion from accumulated heat)
# that the existing 5-step multi-step EFE misses, while remaining cheap
# enough to run for 3–5 proposals each DECIDE tick.
_DEFAULT_HORIZON: int = 10

# Temporal discount factor — matches compute_multistep_efe for consistency.
_GAMMA: float = 0.92

# Hard-prune threshold: lethal outcomes at depth ≤ this are "fear" reactions
# (immediate avoidance, terminal_risk = 1.0).
_DEFAULT_HARD_PRUNE_DEPTH: int = 2

# Safety margins — inside these distances from each death threshold the
# organism enters the "high-surprise" zone (anxiety signal).
# Mirrors _SAFETY_MARGIN in inference.py.
_SAFETY_MARGIN: dict[str, float] = {
    "energy":    20.0,   # flag when energy < 20            (death at 0)
    "heat":      15.0,   # flag when (100 − heat) < 15      (death at 100)
    "integrity": 20.0,   # flag when integrity < 30         (death at 10)
    "stability": 20.0,   # flag when stability < 20         (death at 0)
}

# Metabolic cost per depth step actually simulated.
# 10-step clean trace: ~0.010 E + ~0.005 H per proposal.
# Hard-pruned at depth 2:  ~0.002 E + ~0.001 H.  Fear is cheap.
_COST_ENERGY_PER_STEP: float = 0.001
_COST_HEAT_PER_STEP: float = 0.0005

# EFE penalty weight for terminal_risk=1.0 (certain imminent death).
# At 250.0, a lethal proposal's EFE is boosted by 250, dwarfing typical
# scores of 50–200 and making it effectively unselectable.
CF_RISK_WEIGHT: float = 250.0


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _extract_vitals(state: MetabolicState) -> dict[str, float]:
    return {
        "energy":    state.energy,
        "heat":      state.heat,
        "waste":     state.waste,
        "integrity": state.integrity,
        "stability": state.stability,
    }


def _apply_delta(
    vitals: dict[str, float],
    delta: dict[str, float],
) -> dict[str, float]:
    """Return a new vitals dict with ``delta`` applied, clamped to physical bounds."""
    result = dict(vitals)
    for k, v in delta.items():
        if k in result:
            result[k] = result[k] + v
    result["energy"]    = max(result["energy"],    -1.0)
    result["heat"]      = min(result["heat"],      110.0)
    result["integrity"] = max(result["integrity"],   0.0)
    result["stability"] = max(result["stability"],  -1.0)
    result["waste"]     = max(result["waste"],       0.0)
    return result


def _is_lethal(vitals: dict[str, float]) -> bool:
    """True if any vital has crossed a death threshold."""
    return (
        vitals["energy"]    <= ENERGY_DEATH_THRESHOLD
        or vitals["heat"]      >= THERMAL_DEATH_THRESHOLD
        or vitals["integrity"] <= INTEGRITY_DEATH_THRESHOLD
        or vitals["stability"] <= STABILITY_DEATH_THRESHOLD
    )


def _in_safety_margin(vitals: dict[str, float]) -> bool:
    """True if any vital is inside its safety margin (danger zone, not yet dead)."""
    return (
        (vitals["energy"]    - ENERGY_DEATH_THRESHOLD)    < _SAFETY_MARGIN["energy"]
        or (THERMAL_DEATH_THRESHOLD - vitals["heat"])        < _SAFETY_MARGIN["heat"]
        or (vitals["integrity"] - INTEGRITY_DEATH_THRESHOLD) < _SAFETY_MARGIN["integrity"]
        or (vitals["stability"] - STABILITY_DEATH_THRESHOLD) < _SAFETY_MARGIN["stability"]
    )


def _sim_free_energy(vitals: dict[str, float]) -> float:
    """Compute free-energy estimate from a plain vitals dict.

    Mirrors ``MetabolicState.free_energy_estimate()`` exactly — weights and
    setpoints must be kept in sync if the metabolic formulae change.
    """
    deviations = (
        2.0 * max(0.0, 80.0 - vitals["energy"])    / 80.0
        + 1.5 * max(0.0, vitals["heat"]   - 20.0)  / 80.0
        + 1.0 * max(0.0, vitals["waste"]  - 10.0)  / 90.0
        + 1.8 * max(0.0, 85.0 - vitals["integrity"]) / 85.0
        + 1.4 * max(0.0, 80.0 - vitals["stability"]) / 80.0
    )
    return deviations / 7.7 * 100.0


def _deep_lethal_risk(
    pruned_at_depth: int,
    horizon: int,
    hard_prune_depth: int,
) -> float:
    """Terminal risk for paths that die after the hard-prune zone.

    Scales from ~0.90 (died just after hard-prune zone) down to a minimum of
    0.50 (died near the end of the horizon).  Any lethal trajectory always
    carries at least 0.50 risk to ensure it is penalised meaningfully.
    """
    denominator = max(1, horizon - hard_prune_depth)
    depth_fraction = (pruned_at_depth - hard_prune_depth) / denominator
    return max(0.50, 0.90 * (1.0 - 0.5 * depth_fraction))


def _survivor_risk(
    high_surprise_depths: list[int],
    horizon: int,
) -> float:
    """Terminal risk for paths that survived the full horizon.

    Uses the geometric mean of:
    - coverage fraction   (how many steps were in a safety margin)
    - recency weighting   (later danger counts more — it is imminent)

    Returns a value in [0.0, 0.80].  (0.80 is intentionally below the 0.90
    floor for deep-lethal paths — surviving is always better than dying.)
    """
    if not high_surprise_depths or horizon <= 0:
        return 0.0
    coverage = len(high_surprise_depths) / horizon
    recency  = sum(d / horizon for d in high_surprise_depths)
    if coverage == 0.0 or recency == 0.0:
        return 0.0
    return min(0.80, (coverage * recency) ** 0.5)


# ------------------------------------------------------------------ #
# Data types                                                           #
# ------------------------------------------------------------------ #

@dataclass
class CounterfactualTrace:
    """Result of simulating one ActionProposal forward N steps.

    Attributes
    ----------
    proposal_name:
        Name of the proposal this trace corresponds to.
    survived:
        True if the simulation ran to the full horizon without any vital
        crossing a death threshold.
    terminal_risk:
        Float 0.0–1.0.
        - 1.0  = lethal within ``hard_prune_depth`` (biological fear reflex).
        - 0.9  = lethal deeper in the horizon.
        - 0–0.8 = proportional to safety-margin exposure for surviving paths.
        - 0.0  = no safety-margin breach; entirely safe trajectory.
    pruned_at_depth:
        Depth at which death occurred, or None if the full horizon was
        simulated.
    high_surprise_depths:
        List of 1-indexed depths where at least one vital was inside a safety
        margin.  Empty for clean, safe trajectories.
    vitals_trajectory:
        Per-tick vital snapshots for the steps actually simulated.  Empty for
        hard-pruned branches (there is nothing useful to log).
    cumulative_vfe:
        Discounted sum of simulated free-energy estimates across the
        trajectory.  ``float('inf')`` for hard-pruned branches.
    depth_simulated:
        Actual number of steps simulated (≤ horizon).  This is what
        ``compute_metabolic_cost`` uses to charge the organism.
    """

    proposal_name: str
    survived: bool
    terminal_risk: float
    pruned_at_depth: int | None
    high_surprise_depths: list[int] = field(default_factory=list)
    vitals_trajectory: list[dict[str, float]] = field(default_factory=list)
    cumulative_vfe: float = 0.0
    depth_simulated: int = 0


# ------------------------------------------------------------------ #
# Engine                                                               #
# ------------------------------------------------------------------ #

class CounterfactualEngine:
    """Depth-first counterfactual simulation with fear-based pruning.

    Parameters
    ----------
    horizon:
        Number of passive-decay ticks to simulate per proposal (default 10).
    gamma:
        Temporal discount factor for cumulative VFE (default 0.92).
    hard_prune_depth:
        Lethal outcomes at or before this depth trigger an immediate hard
        prune (terminal_risk = 1.0).  Default is 2, matching the user's
        recommendation: if a branch kills the organism within 2 steps it is
        treated as an unconditional "fear zone" — no further deliberation.
    """

    def __init__(
        self,
        horizon: int = _DEFAULT_HORIZON,
        gamma: float = _GAMMA,
        hard_prune_depth: int = _DEFAULT_HARD_PRUNE_DEPTH,
    ) -> None:
        self.horizon = horizon
        self.gamma = gamma
        self.hard_prune_depth = hard_prune_depth

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def simulate(
        self,
        state: MetabolicState,
        proposal: ActionProposal,
    ) -> CounterfactualTrace:
        """Simulate one proposal forward in time.

        ``state`` is **never mutated**.  A plain vitals dict is cloned and
        advanced using the same passive-decay equations as the real tick loop,
        without the exception-raising, hormone proxies, or arousal gate.

        Parameters
        ----------
        state:
            Current metabolic state (read-only).
        proposal:
            Candidate action to evaluate.

        Returns
        -------
        CounterfactualTrace
        """
        # Post-action vitals: apply the predicted delta (and subtract the
        # action's own energy cost) from a snapshot of the current state.
        delta = dict(proposal.predicted_delta)
        delta["energy"] = delta.get("energy", 0.0) - proposal.cost_energy
        vitals = _apply_delta(_extract_vitals(state), delta)

        trajectory: list[dict[str, float]] = []
        high_surprise_depths: list[int] = []
        cumulative_vfe = 0.0
        discount = 1.0

        for depth in range(1, self.horizon + 1):
            vitals = _decay_vitals_one_step(vitals, allostatic_load=state.allostatic_load)

            # ---- Hard prune (fear reflex) --------------------------------
            if _is_lethal(vitals) and depth <= self.hard_prune_depth:
                return CounterfactualTrace(
                    proposal_name=proposal.name,
                    survived=False,
                    terminal_risk=1.0,
                    pruned_at_depth=depth,
                    high_surprise_depths=[depth],
                    vitals_trajectory=[],        # no trajectory for fear-pruned branches
                    cumulative_vfe=float("inf"),
                    depth_simulated=depth,
                )

            # ---- Deep lethal (past hard-prune zone) ----------------------
            if _is_lethal(vitals):
                # Still record what we simulated up to this point
                trajectory.append(dict(vitals))
                high_surprise_depths.append(depth)
                return CounterfactualTrace(
                    proposal_name=proposal.name,
                    survived=False,
                    terminal_risk=_deep_lethal_risk(
                        depth, self.horizon, self.hard_prune_depth
                    ),
                    pruned_at_depth=depth,
                    high_surprise_depths=high_surprise_depths,
                    vitals_trajectory=trajectory,
                    cumulative_vfe=cumulative_vfe,
                    depth_simulated=depth,
                )

            # ---- Soft high-surprise check --------------------------------
            if _in_safety_margin(vitals):
                high_surprise_depths.append(depth)

            # ---- Accumulate discounted VFE -------------------------------
            cumulative_vfe += discount * _sim_free_energy(vitals)
            discount *= self.gamma

            trajectory.append(dict(vitals))

        # Survived the full horizon
        return CounterfactualTrace(
            proposal_name=proposal.name,
            survived=True,
            terminal_risk=_survivor_risk(high_surprise_depths, self.horizon),
            pruned_at_depth=None,
            high_surprise_depths=high_surprise_depths,
            vitals_trajectory=trajectory,
            cumulative_vfe=cumulative_vfe,
            depth_simulated=self.horizon,
        )

    def run_batch(
        self,
        state: MetabolicState,
        proposals: list[ActionProposal],
    ) -> list[CounterfactualTrace]:
        """Simulate all proposals depth-first and return one trace per proposal.

        Proposals are evaluated sequentially to full depth (or early prune),
        mirroring a depth-first traversal of the proposal forest.  This is
        intentional: a hard-pruned branch costs almost nothing (metabolically
        cheap fear), so there is no efficiency reason to interleave evaluation
        breadth-first.

        ``state`` is **never mutated**.

        Parameters
        ----------
        state:
            Current metabolic state (read-only).
        proposals:
            Candidate actions to evaluate.

        Returns
        -------
        list[CounterfactualTrace]
            One trace per proposal, in the same order as ``proposals``.
        """
        return [self.simulate(state, p) for p in proposals]

    def compute_metabolic_cost(
        self,
        traces: list[CounterfactualTrace],
    ) -> tuple[float, float]:
        """Compute the total metabolic cost of running this batch of simulations.

        Cost scales with the total number of steps *actually simulated* across
        all traces.  Hard-pruned branches (≤ 2 steps) contribute tiny amounts
        — the organism barely pays to rule out lethal proposals, reflecting
        that biological fear responses are metabolically cheap.

        Returns
        -------
        tuple[float, float]
            ``(energy_cost, heat_cost)``
        """
        total_steps = sum(t.depth_simulated for t in traces)
        return (
            total_steps * _COST_ENERGY_PER_STEP,
            total_steps * _COST_HEAT_PER_STEP,
        )
