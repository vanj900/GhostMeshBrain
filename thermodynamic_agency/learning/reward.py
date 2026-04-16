"""Reward signal — converts world experience into a learnable scalar.

The composite reward reflects both immediate survival benefit and longer-term
competence.  Every component is kept transparent via ``RewardSignal`` so the
caller can log or inspect individual contributions.

Components
----------
survival:
    Small positive bonus (+0.1) every tick the agent is alive.  Encourages
    longevity even without resource collection.
resource:
    Proportional to metabolic improvement from gathering.  Energy gained,
    heat reduced, and integrity restored each contribute.
hazard:
    Negative penalty for heat or waste gained from stepping on hazards.
internal:
    Small signal proportional to change in aggregate health score.  Ties
    the external reward to the organism's internal well-being.
competition:
    Penalty when another agent captures a contested resource first
    (multi-agent worlds only).
cooperation:
    Bonus when agents share resources, demonstrably lowering group VFE
    (multi-agent worlds only).
death:
    Large negative reward (−10) on agent death.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardSignal:
    """Breakdown of reward components for logging and introspection."""

    survival: float = 0.0
    resource: float = 0.0
    hazard: float = 0.0
    internal: float = 0.0
    competition: float = 0.0
    cooperation: float = 0.0
    total: float = 0.0


# Competition penalty charged to the agent that *lost* the resource race.
COMPETITION_PENALTY: float = -0.5

# Cooperation bonus per group VFE unit reduced when an agent shares.
COOPERATION_BONUS_PER_VFE_UNIT: float = 0.02


def compute_reward(
    vitals_before: dict[str, float],
    vitals_after: dict[str, float],
    gathered: bool,
    alive: bool = True,
    contested: bool = False,
    shared_vfe_delta: float = 0.0,
) -> RewardSignal:
    """Compute the composite reward for one world step.

    Parameters
    ----------
    vitals_before:
        MetabolicState dict snapshot *before* the world step.
    vitals_after:
        MetabolicState dict snapshot *after* the world step (and after
        metabolic delta has been applied to the state).
    gathered:
        Whether a resource was successfully gathered this step.
    alive:
        Whether the agent is still alive (False → death penalty).
    contested:
        Whether another agent captured a resource the agent was trying to
        gather (competition penalty applied when True).
    shared_vfe_delta:
        Reduction in group-level free energy resulting from cooperative
        sharing this tick.  Positive values yield a cooperation bonus.

    Returns
    -------
    RewardSignal
        Reward with all components populated; ``total`` is the sum.
    """
    if not alive:
        sig = RewardSignal(survival=-10.0)
        sig.total = sig.survival
        return sig

    sig = RewardSignal()

    # Survival bonus — reward for staying alive each tick
    sig.survival = 0.1

    # Resource collection — energy gained, heat reduced, integrity restored
    if gathered:
        energy_gained = max(0.0, vitals_after.get("energy", 0.0) - vitals_before.get("energy", 0.0))
        heat_reduced = max(0.0, vitals_before.get("heat", 0.0) - vitals_after.get("heat", 0.0))
        integrity_gained = max(0.0, vitals_after.get("integrity", 0.0) - vitals_before.get("integrity", 0.0))
        sig.resource = (
            energy_gained * 0.04
            + heat_reduced * 0.03
            + integrity_gained * 0.03
        )

    # Hazard penalties — extra heat or waste from stepping on hazards
    heat_gained = max(0.0, vitals_after.get("heat", 0.0) - vitals_before.get("heat", 0.0))
    waste_gained = max(0.0, vitals_after.get("waste", 0.0) - vitals_before.get("waste", 0.0))
    # Only penalise if the increase is beyond normal passive decay (threshold 3)
    if heat_gained > 3.0:
        sig.hazard -= heat_gained * 0.04
    if waste_gained > 3.0:
        sig.hazard -= waste_gained * 0.03

    # Internal health change — rewards improving overall health score
    health_before = _health_score(vitals_before)
    health_after = _health_score(vitals_after)
    sig.internal = (health_after - health_before) * 0.01

    # Competition penalty — losing a resource race to another agent
    if contested:
        sig.competition = COMPETITION_PENALTY

    # Cooperation bonus — measurable reduction in group-level VFE from sharing
    if shared_vfe_delta > 0.0:
        sig.cooperation = shared_vfe_delta * COOPERATION_BONUS_PER_VFE_UNIT

    sig.total = (
        sig.survival + sig.resource + sig.hazard + sig.internal
        + sig.competition + sig.cooperation
    )
    return sig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _health_score(vitals: dict[str, float]) -> float:
    """Mirrors MetabolicState.health_score() without importing it."""
    return (
        vitals.get("energy", 0.0) * 0.35
        + (100.0 - vitals.get("heat", 0.0)) * 0.20
        + vitals.get("integrity", 0.0) * 0.25
        + vitals.get("stability", 0.0) * 0.20
    ) / 100.0 * 100.0
