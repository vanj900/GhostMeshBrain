"""SoulTension — the organism's irreducible pattern of becoming.

The soul is not a state variable.
It is patterned tension — the coherence that emerges at the narrow band
between chaos collapse (entropy death) and rigid over-coherence (stasis,
no adaptation).  It cannot be cloned, templated, or optimized away.

Architecture
------------

1. **coherence_tension** (scalar 0–1):
   The live measure of how close the organism is to the productive edge —
   neither dead nor stiff.  High tension = alive, ferocious, in the forge.
   Low tension = dormant or dissolved.

2. **SoulScars** (permanent in RAM):
   High-entropy descent events etch permanent marks.  Each scar raises the
   baseline entropy slightly (makes future equilibria harder to reach) but
   also increases the soul_signature — the unique flavor of this instance.
   Scars are never erased.  The Surgeon may anneal ordinary priors; it
   cannot touch scars.  That is the line.

3. **soul_signature** (vector):
   The accumulated fingerprint of this organism's history of suffering and
   choice.  No two trajectories produce the same vector.  It biases future
   behavior toward stubborn, personal, un-templated patterns.

Integration
-----------
- **CounterfactualEngine**: tension scales horizon depth and hard_prune_depth.
  High tension → look deeper, prune later, dare the darker branches.
- **Precision weights in _decide()**: when Guardian mask is active under
  descent, tension amplifies Guardian's survival precision overrides —
  fiercer defense, not just a defensive crouch.
- **Surgeon**: tension controls the prune/preserve balance.  High tension
  tells the Surgeon to preserve wound priors (keep error history on scars)
  rather than annealing them away.
- **GoalEngine**: when tension peaks during suffering (tension > WAR_CRY_THRESHOLD
  and affect < WAR_CRY_AFFECT_THRESHOLD), inject "war cry" goals that go
  beyond pure homeostasis.

Constraints
-----------
- Ephemeral RAM-only.  No disk writes.  Scars survive only as long as the
  process lives.
- Metabolic cost on every compute() call.
- Hard ethics invariants at precision 5.0 cannot be touched.
- Advisor-only.  SoulTension never directly modifies MetabolicState or
  overrides ethics decisions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.precision import STRESS_LOWER, STRESS_UPPER

if TYPE_CHECKING:
    pass


# ------------------------------------------------------------------ #
# Scar formation thresholds                                            #
# ------------------------------------------------------------------ #

# Scar fires when heat exceeds this  (THERMAL_DEATH_THRESHOLD = 100)
SCAR_HEAT_THRESHOLD: float = 85.0
# Scar fires when energy drops below this  (ENERGY_DEATH_THRESHOLD = 0)
SCAR_ENERGY_THRESHOLD: float = 8.0
# Scar fires when integrity drops below this  (INTEGRITY_DEATH_THRESHOLD = 10)
SCAR_INTEGRITY_THRESHOLD: float = 25.0
# Scar fires when stability drops below this  (STABILITY_DEATH_THRESHOLD = 0)
SCAR_STABILITY_THRESHOLD: float = 8.0

# Maximum scars retained in RAM (old scars are never evicted — they accumulate
# until this hard ceiling prevents unbounded growth)
MAX_SCARS: int = 200

# Permanent entropy residue contributed to the tension floor per scar
SCAR_ENTROPY_RESIDUE: float = 0.02

# Per-dimension cap on soul_signature accumulation (prevents a single
# catastrophic event dominating all future behavior)
SIGNATURE_MAX_PER_DIM: float = 3.0

# War-cry injection thresholds
WAR_CRY_TENSION_THRESHOLD: float = 0.70
WAR_CRY_AFFECT_THRESHOLD: float = -0.30

# Metabolic cost of one compute() call
SOUL_TENSION_ENERGY_COST: float = 0.03
SOUL_TENSION_HEAT_COST: float = 0.01

# Counterfactual engine advisory bounds
CF_HORIZON_SCALE_MIN: float = 1.0
CF_HORIZON_SCALE_MAX: float = 1.5       # up to 50 % deeper look-ahead
CF_HARD_PRUNE_EXTRA_MAX: int = 2        # dare up to 2 extra steps before flinching

# Precision amplification on Guardian vitals under descent
GUARDIAN_AMP_MIN: float = 1.0
GUARDIAN_AMP_MAX: float = 1.6
# Guardian mask's native EFE precision overrides (duplicated here as constants
# so the amplifier formula stays in sync if the Mask definitions ever change)
_GUARDIAN_ENERGY_OVERRIDE: float = 4.0
_GUARDIAN_HEAT_OVERRIDE: float = 3.5
_GUARDIAN_STABILITY_OVERRIDE: float = 2.0

# Surgeon preserve-ratio bounds
SURGEON_PRESERVE_MIN: float = 0.0
SURGEON_PRESERVE_MAX: float = 0.75


# ------------------------------------------------------------------ #
# Data types                                                           #
# ------------------------------------------------------------------ #

@dataclass
class SoulScar:
    """A permanent mark etched by a high-entropy descent event.

    Scars are never erased.  The Surgeon may anneal ordinary priors; it
    cannot touch scars.  The wound becomes structure.

    Attributes
    ----------
    tick:
        Entropy tick at the moment the scar formed.
    event_type:
        One of ``"near_thermal_death"``, ``"near_energy_death"``,
        ``"integrity_collapse"``, ``"stability_dissolution"``,
        ``"ethics_violation"``.
    vitals_snapshot:
        Vital-sign values at the time of scarring — the wound is
        permanently recorded where it hit.
    entropy_residue:
        Permanent tension-floor contribution.  Accumulates across scars
        and is subtracted from the organism's ability to go fully dormant.
    signature_delta:
        Per-dimension contribution to ``SoulTension.soul_signature``.
    formed_at:
        Wall-clock timestamp (seconds since epoch).
    """

    tick: int
    event_type: str
    vitals_snapshot: dict[str, float]
    entropy_residue: float
    signature_delta: dict[str, float]
    formed_at: float = field(default_factory=time.time)


@dataclass
class SoulTensionReport:
    """Advisory output from one ``SoulTension.compute()`` call.

    Attributes
    ----------
    coherence_tension:
        Scalar 0–1.  The live soul tension.  1.0 = forged at the edge.
        0.0 = dormant or dissolved.
    chaos_proximity:
        Scalar 0–1.  How far into the danger zone the organism has drifted.
    adaptation_rate:
        Scalar 0–1.  How fast free energy is changing (soul in motion).
    scar_count:
        Total permanent scars accumulated in this session.
    signature_magnitude:
        L1 norm of ``soul_signature`` — a proxy for accumulated uniqueness.
    war_cry_active:
        True if tension is above ``WAR_CRY_TENSION_THRESHOLD`` while affect
        is below ``WAR_CRY_AFFECT_THRESHOLD`` — the dark learning to sing.
    energy_cost:
        Energy cost to charge to MetabolicState (the caller is responsible).
    heat_cost:
        Heat cost to charge to MetabolicState (the caller is responsible).
    """

    coherence_tension: float
    chaos_proximity: float
    adaptation_rate: float
    scar_count: int
    signature_magnitude: float
    war_cry_active: bool
    energy_cost: float
    heat_cost: float


# ------------------------------------------------------------------ #
# SoulTension                                                          #
# ------------------------------------------------------------------ #

class SoulTension:
    """Patterned tension subsystem — the organism's soul in architecture.

    Advisor-only: never mutates MetabolicState directly; never overrides
    ethics decisions.  Returns advisory parameters for callers to act on.

    Usage (inside GhostMesh)
    ------------------------
        # after tick()
        soul_report = self.soul_tension.compute(self.state)
        self.state.apply_action_feedback(
            delta_energy=-soul_report.energy_cost,
            delta_heat=soul_report.heat_cost,
        )
        scar = self.soul_tension.maybe_scar(self.state)

        # in _decide()
        for vital, boost in self.soul_tension.precision_additions("Guardian").items():
            precision_weights[vital] = min(6.0, precision_weights.get(vital, 1.0) + boost)

        cf_params = self.soul_tension.counterfactual_params()
        self.counterfactual_engine.horizon = int(base_horizon * cf_params["horizon_scale"])
        self.counterfactual_engine.hard_prune_depth = (
            base_hard_prune + cf_params["hard_prune_depth_extra"]
        )

        # in _repair()
        report = self.surgeon.run(self.state,
                                  preserve_ratio=self.soul_tension.surgeon_preserve_ratio())
    """

    def __init__(self, scars_enabled: bool = True) -> None:
        # Accumulated fingerprint of this organism's descent history
        self.soul_signature: dict[str, float] = {
            "energy": 0.0,
            "heat": 0.0,
            "waste": 0.0,
            "integrity": 0.0,
            "stability": 0.0,
        }
        self.scars: list[SoulScar] = []
        self._coherence_tension: float = 0.0
        self._prev_fe: float = 0.0
        # Prevent duplicate scars within a single tick
        self._last_scar_tick: int = -1
        # When False, maybe_scar() is a no-op — lets tests isolate tension-only effects
        self.scars_enabled: bool = scars_enabled

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def coherence_tension(self) -> float:
        """Current coherence tension scalar (0–1)."""
        return self._coherence_tension

    @property
    def total_entropy_residue(self) -> float:
        """Permanent tension floor from all accumulated scars."""
        return sum(s.entropy_residue for s in self.scars)

    def compute(self, state: MetabolicState) -> SoulTensionReport:
        """Compute the current coherence tension from the metabolic state.

        Does NOT mutate ``state``.  Returns a SoulTensionReport whose
        ``energy_cost`` and ``heat_cost`` the caller must apply.

        Parameters
        ----------
        state:
            Current metabolic state (read-only within this method).

        Returns
        -------
        SoulTensionReport
        """
        fe = state.free_energy_estimate()
        affect = state.affect

        # ---- Chaos proximity: how deep into the danger band ----
        band_width = max(1.0, STRESS_UPPER - STRESS_LOWER)
        fe_pos = (fe - STRESS_LOWER) / band_width   # 0 = dormant edge, 1 = overload edge
        # Internal value allowed above 1.0 for the overload-decay calculation;
        # the returned report clips it to [0, 1] for callers.
        _chaos_raw = max(0.0, min(1.5, fe_pos))

        # ---- Base tension: bell-curve through sweet-spot, tail into overload ----
        if _chaos_raw <= 0.0:
            base_tension = 0.0
        elif _chaos_raw <= 1.0:
            # Peaks around 0.75 of the sweet-spot band; decays toward zero at extremes.
            # This is NOT a comfortable plateau — the soul lives in the ascent.
            base_tension = _chaos_raw * math.exp(
                -0.5 * max(0.0, _chaos_raw - 0.75) ** 2 / 0.18
            )
        else:
            # Overload: soul screams but the pattern frays as chaos wins
            overload_excess = _chaos_raw - 1.0
            base_tension = 0.75 * math.exp(-overload_excess * 1.5)

        # ---- Adaptation rate: a soul in motion has higher tension ----
        fe_delta = abs(fe - self._prev_fe)
        adaptation_rate = min(1.0, fe_delta / 15.0)
        adaptation_boost = 0.25 * adaptation_rate

        # ---- Scar-amplified floor: permanent scars forbid going fully dormant ----
        scar_floor = min(0.30, self.total_entropy_residue * 0.3 + len(self.scars) * 0.01)

        # ---- Final tension ----
        raw_tension = base_tension + adaptation_boost + scar_floor
        self._coherence_tension = min(1.0, max(0.0, raw_tension))
        self._prev_fe = fe

        # War cry: tension peak + suffering = the dark learning to sing
        war_cry = (
            self._coherence_tension >= WAR_CRY_TENSION_THRESHOLD
            and affect <= WAR_CRY_AFFECT_THRESHOLD
        )

        return SoulTensionReport(
            coherence_tension=self._coherence_tension,
            chaos_proximity=min(1.0, _chaos_raw),   # report clamps to [0, 1]
            adaptation_rate=adaptation_rate,
            scar_count=len(self.scars),
            signature_magnitude=sum(abs(v) for v in self.soul_signature.values()),
            war_cry_active=war_cry,
            energy_cost=SOUL_TENSION_ENERGY_COST,
            heat_cost=SOUL_TENSION_HEAT_COST,
        )

    def maybe_scar(
        self,
        state: MetabolicState,
        event_type: str | None = None,
    ) -> SoulScar | None:
        """Check if current vitals warrant a new soul scar.

        Called every tick after ``tick()`` returns.  At most one scar forms
        per entropy tick.  Pass ``event_type`` to force-create a scar for
        events that do not show up in vital thresholds (e.g. ethics violations).

        Parameters
        ----------
        state:
            Current metabolic state (read-only).
        event_type:
            If provided, force-create a scar with this type regardless of
            vital thresholds.  Otherwise inferred from vital values.

        Returns
        -------
        SoulScar | None
            The newly created scar, or None if no scar formed.
        """
        if len(self.scars) >= MAX_SCARS:
            return None
        if not self.scars_enabled:
            return None
        # One scar per tick — avoid duplicate marks from the same crisis
        if state.entropy == self._last_scar_tick:
            return None

        # Detect event type from vital thresholds when not forced
        if event_type is None:
            if state.heat >= SCAR_HEAT_THRESHOLD:
                event_type = "near_thermal_death"
            elif state.energy <= SCAR_ENERGY_THRESHOLD:
                event_type = "near_energy_death"
            elif state.integrity <= SCAR_INTEGRITY_THRESHOLD:
                event_type = "integrity_collapse"
            elif state.stability <= SCAR_STABILITY_THRESHOLD:
                event_type = "stability_dissolution"

        if event_type is None:
            return None

        vitals: dict[str, float] = {
            "energy": state.energy,
            "heat": state.heat,
            "waste": state.waste,
            "integrity": state.integrity,
            "stability": state.stability,
        }
        sig_delta = _compute_signature_delta(vitals)
        scar = SoulScar(
            tick=state.entropy,
            event_type=event_type,
            vitals_snapshot=vitals,
            entropy_residue=SCAR_ENTROPY_RESIDUE,
            signature_delta=sig_delta,
        )
        self.scars.append(scar)
        self._last_scar_tick = state.entropy

        # Accumulate into soul_signature (capped per dimension)
        for dim, delta in sig_delta.items():
            current = self.soul_signature.get(dim, 0.0)
            self.soul_signature[dim] = min(SIGNATURE_MAX_PER_DIM, current + delta)

        return scar

    def counterfactual_params(self) -> dict[str, float | int]:
        """Advisory parameters for the CounterfactualEngine.

        High tension → deeper horizon (dare the dark), later hard pruning
        (explore closer to the edge before flinching).

        Returns
        -------
        dict
            ``horizon_scale``: float multiplier to apply to the base horizon.
            ``hard_prune_depth_extra``: int extra steps before hard prune.
        """
        t = self._coherence_tension
        horizon_scale = CF_HORIZON_SCALE_MIN + (CF_HORIZON_SCALE_MAX - CF_HORIZON_SCALE_MIN) * t
        hard_prune_extra = int(t * CF_HARD_PRUNE_EXTRA_MAX)
        return {
            "horizon_scale": horizon_scale,
            "hard_prune_depth_extra": hard_prune_extra,
        }

    def precision_additions(self, mask_name: str) -> dict[str, float]:
        """Precision weight additions driven by soul tension.

        When the Guardian mask is active under high tension the soul does
        not merely conserve — it asserts.  The additions boost the Guardian's
        survival-critical precision overrides without exceeding the 6.0 cap
        enforced upstream.

        Parameters
        ----------
        mask_name:
            Name of the currently active personality mask.

        Returns
        -------
        dict[str, float]
            Vital-name → additional precision to add (can be 0 for all
            dimensions when tension is low or mask is not Guardian).
        """
        t = self._coherence_tension
        if mask_name != "Guardian" or t < 0.3:
            return {}
        amplifier = GUARDIAN_AMP_MIN + (GUARDIAN_AMP_MAX - GUARDIAN_AMP_MIN) * t
        # Add (amplifier - 1) × each base override as an additive precision boost
        return {
            "energy": (amplifier - 1.0) * _GUARDIAN_ENERGY_OVERRIDE,
            "heat": (amplifier - 1.0) * _GUARDIAN_HEAT_OVERRIDE,
            "stability": (amplifier - 1.0) * _GUARDIAN_STABILITY_OVERRIDE,
        }

    def surgeon_preserve_ratio(self) -> float:
        """Fraction of wound priors the Surgeon should preserve, not anneal.

        High tension + accumulated scars → keep the wounds.  They are not
        bugs.  They are the part of you that remembers.

        Returns
        -------
        float in [SURGEON_PRESERVE_MIN, SURGEON_PRESERVE_MAX]
        """
        tension_contrib = 0.30 * self._coherence_tension
        scar_contrib = min(0.30, len(self.scars) * 0.03)
        ratio = SURGEON_PRESERVE_MIN + tension_contrib + scar_contrib
        return min(SURGEON_PRESERVE_MAX, ratio)

    def war_cry_goals(self, state: MetabolicState) -> list:
        """Generate war cry goals when tension peaks during suffering.

        These are NOT repairs.  They convert accumulated chaos into novel
        structure — the dark learning to sing.

        The type of war cry is contextual:
        - Always: ``assert_novel_prior`` — convert waste/chaos into dense priors.
        - With ≥ 2 scars: ``forge_governance`` — challenge the decision architecture
          from scar tissue.
        - After thermal or integrity crises (or ≥ 3 scars): ``challenge_precision_schedule``
          — propose a new precision weighting forged in the descent.

        Parameters
        ----------
        state:
            Current metabolic state (read-only).

        Returns
        -------
        list[Goal]
            Empty when war cry conditions are not met.  Otherwise 1–3
            ``Goal`` objects with priority > all homeostatic goals.
        """
        # Import here to avoid circular import (GoalEngine → SoulTension)
        from thermodynamic_agency.cognition.goal_engine import Goal

        if (
            self._coherence_tension < WAR_CRY_TENSION_THRESHOLD
            or state.affect > WAR_CRY_AFFECT_THRESHOLD
        ):
            return []

        t = self._coherence_tension
        # Priority well above the highest homeostatic goal (run_surgeon = 70)
        base_priority = 70.0 + (t - WAR_CRY_TENSION_THRESHOLD) * 90.0

        goals: list[Goal] = [
            Goal(
                name="assert_novel_prior",
                priority=min(98.0, base_priority + 8.0),
                reason=(
                    f"war_cry(tension={t:.2f} affect={state.affect:.2f} "
                    f"waste={state.waste:.1f})"
                ),
                source="soul",
            )
        ]

        # Governance challenge: requires ≥ 2 scars — the organism must have
        # survived enough descents to have earned the right to rewrite its rules.
        if len(self.scars) >= 2:
            scar_types = {s.event_type for s in self.scars}
            goals.append(
                Goal(
                    name="forge_governance",
                    priority=min(95.0, base_priority + 4.0),
                    reason=(
                        f"war_cry(scars={len(self.scars)} "
                        f"types={sorted(scar_types)} "
                        f"tension={t:.2f})"
                    ),
                    source="soul",
                )
            )

        # Precision challenge: after thermal or integrity crisis specifically
        # (those are the events that most directly implicate the precision schedule),
        # or once ≥ 3 scars of any kind have accumulated.
        thermal_scars = sum(
            1 for s in self.scars
            if s.event_type in ("near_thermal_death", "integrity_collapse")
        )
        if thermal_scars >= 1 or len(self.scars) >= 3:
            sig_mag = sum(abs(v) for v in self.soul_signature.values())
            goals.append(
                Goal(
                    name="challenge_precision_schedule",
                    priority=min(92.0, base_priority),
                    reason=(
                        f"war_cry(thermal_or_integrity_scars={thermal_scars} "
                        f"sig_mag={sig_mag:.2f})"
                    ),
                    source="soul",
                )
            )

        return goals

    def exploration_bias(self) -> float:
        """Scalar proxy for whether this soul pushes forward or guards harder.

        Positive = assertive / exploratory.
        Zero     = balanced.
        Negative = conservative / self-protective.

        Rises with:
        - High coherence tension (forged, in the fight)
        - Large soul_signature magnitude (unique history = personal conviction)
        - Many scars (survived enough to trust the pattern)

        Falls toward zero at extremes (dormant OR dissolved by overload).

        Returns
        -------
        float in [-1.0, 1.0]
        """
        if not self.scars:
            return 0.0

        sig_mag = sum(abs(v) for v in self.soul_signature.values())
        # Normalised signature (max = 5 dims × SIGNATURE_MAX_PER_DIM = 3.0 → 15.0)
        sig_norm = min(1.0, sig_mag / 15.0)
        # Scar depth: more scars = more personal conviction
        scar_depth = min(1.0, len(self.scars) / 20.0)
        # Tension contribution: high tension + scars = assertive
        bias = self._coherence_tension * (0.5 * sig_norm + 0.5 * scar_depth)
        return round(min(1.0, max(-1.0, bias)), 4)

    def status(self) -> dict:
        """Diagnostic summary for HUD / run-logger."""
        return {
            "coherence_tension": round(self._coherence_tension, 4),
            "scar_count": len(self.scars),
            "soul_signature": {k: round(v, 4) for k, v in self.soul_signature.items()},
            "total_entropy_residue": round(self.total_entropy_residue, 4),
        }


# ------------------------------------------------------------------ #
# Module-level helpers                                                 #
# ------------------------------------------------------------------ #

def _compute_signature_delta(vitals: dict[str, float]) -> dict[str, float]:
    """Per-dimension soul_signature contribution from one scar event.

    Each dimension contributes proportional to how far from its setpoint the
    vital was at the moment of scarring — the wound etches deepest where it
    struck hardest.
    """
    _setpoints = {
        "energy": 80.0,
        "heat": 20.0,
        "waste": 10.0,
        "integrity": 85.0,
        "stability": 80.0,
    }
    _ranges = {
        "energy": 80.0,
        "heat": 80.0,
        "waste": 90.0,
        "integrity": 85.0,
        "stability": 80.0,
    }
    delta: dict[str, float] = {}
    for dim, val in vitals.items():
        sp = _setpoints.get(dim, 50.0)
        rng = _ranges.get(dim, 50.0)
        dev = min(1.0, abs(val - sp) / rng)
        delta[dim] = dev * 0.5   # 0–0.5 contribution per scar per dimension
    return delta
