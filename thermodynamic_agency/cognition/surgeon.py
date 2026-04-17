"""Surgeon — integrity repair via Bayesian precision annealing.

Triggered when MetabolicState.tick() returns "REPAIR" (low integrity or
stability).

The Surgeon diagnoses "frozen precision" — rigid bad priors, hallucinated
logic, or ethical drift — and applies a controlled annealing schedule to
soften those priors, expose the system to corrective prediction errors, and
restore coherence.

Concepts
--------
- **Precision annealing**: Temporarily reduce the precision (inverse variance)
  weight on suspect beliefs, allowing evidence to update them more freely.
- **Self-red-team**: Generate adversarial scenarios to probe consistency.
- **Allostatic load**: Accumulated repair costs tracked in MetabolicState.

Usage
-----
    surgeon = Surgeon(diary=diary, ethics=engine)
    report = surgeon.run(state)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.memory.diary import RamDiary, DiaryEntry
from thermodynamic_agency.cognition.precision import PrecisionEngine

# Metabolic costs/rewards for a Surgeon pass
SURGEON_ENERGY_COST: float = 5.0
SURGEON_INTEGRITY_GAIN: float = 12.0
SURGEON_STABILITY_GAIN: float = 8.0
SURGEON_HEAT_COST: float = 3.0      # repair generates some heat
SURGEON_WASTE_COST: float = 2.0     # repair generates some waste

# Annealing schedule parameters
INITIAL_TEMPERATURE: float = 1.0    # starts hot (open-minded)
COOLING_RATE: float = 0.85          # geometric cooling per round


@dataclass
class BeliefPrior:
    """A belief held by the system with an associated precision weight."""

    name: str
    value: Any
    precision: float = 1.0          # inverse variance; higher = more rigid
    last_updated: float = field(default_factory=time.time)
    error_count: int = 0            # times this belief produced prediction errors
    protected: bool = False         # if True, Surgeon will never anneal this prior


@dataclass
class SurgeonReport:
    beliefs_audited: int
    beliefs_annealed: int
    integrity_gain: float
    stability_gain: float
    allostatic_cost: float          # total metabolic cost of repair
    diagnosis: str
    red_team_results: list[str] = field(default_factory=list)


class Surgeon:
    """Integrity-repair subsystem — precision annealing + self-red-team."""

    def __init__(
        self,
        diary: RamDiary,
        priors: list[BeliefPrior] | None = None,
    ) -> None:
        self.diary = diary
        self.priors: list[BeliefPrior] = priors or _default_priors()
        self._anneal_round = 0
        self._precision_engine = PrecisionEngine()

    # ------------------------------------------------------------------ #
    # Main entry-point                                                     #
    # ------------------------------------------------------------------ #

    def run(self, state: MetabolicState, preserve_ratio: float = 0.0) -> SurgeonReport:
        """Run a full Surgeon pass.

        1. Tune precision weights via PrecisionEngine (know where to look).
        2. Audit beliefs for frozen precision (error_count, age).
        3. Anneal suspect beliefs with precision-scaled temperature.
           When ``preserve_ratio`` > 0, the Surgeon preserves the most
           error-laden (wound) priors rather than annealing them — the
           soul's accumulated damage is structure, not noise.
        4. Self-red-team: generate adversarial probes.
        5. Apply metabolic feedback — higher precision sharpening costs more.

        Parameters
        ----------
        state:
            MetabolicState — mutated via apply_action_feedback().
        preserve_ratio:
            Fraction of frozen priors to preserve (not anneal).  Supplied
            by SoulTension when coherence_tension is high.  Bounded to
            [0, 1]; priors with the highest error_count are preserved first
            (they are the wounds that earned their scars).

        Returns
        -------
        SurgeonReport
        """
        # Step 1: Query precision engine — tells us where surprise is highest
        # and what it costs to sharpen attention right now.
        precision_report = self._precision_engine.tune(state)

        frozen = self._identify_frozen(state)
        annealed = self._anneal(frozen, state, precision_report.weights, preserve_ratio)
        red_team_results = self._red_team(state)

        # Allostatic cost grows with integrity deficit (more damage = more effort)
        integrity_deficit = max(0.0, 100.0 - state.integrity)
        allostatic_multiplier = 1.0 + integrity_deficit / 100.0

        # Precision sharpening surcharge: sweet-spot costs more because
        # we're doing real epistemic work (tightening beliefs takes energy).
        precision_surcharge = precision_report.energy_cost
        effective_energy_cost = SURGEON_ENERGY_COST * allostatic_multiplier + precision_surcharge

        state.apply_action_feedback(
            delta_energy=-effective_energy_cost,
            delta_heat=SURGEON_HEAT_COST + precision_report.heat_cost,
            delta_waste=SURGEON_WASTE_COST,
            delta_integrity=SURGEON_INTEGRITY_GAIN,
            delta_stability=SURGEON_STABILITY_GAIN,
        )

        self._anneal_round += 1

        diagnosis_parts = []
        if frozen:
            diagnosis_parts.append(
                f"Frozen priors detected: {[b.name for b in frozen]}"
            )
        else:
            diagnosis_parts.append("No frozen priors detected")
        if red_team_results:
            diagnosis_parts.append(f"Red-team flags: {red_team_results}")

        # Affect signal gives insight into current surprise gradient
        affect_label = (
            "resolving" if state.affect > 0.1
            else "stressed" if state.affect < -0.1
            else "neutral"
        )
        diagnosis_parts.append(
            f"Affect={state.affect:.3f} ({affect_label}), "
            f"FE={precision_report.free_energy:.1f} [{precision_report.regime}]"
        )

        self.diary.append(
            DiaryEntry(
                tick=state.entropy,
                role="repair",
                content="; ".join(diagnosis_parts),
                metadata={
                    "beliefs_annealed": len(annealed),
                    "allostatic_cost": effective_energy_cost,
                    "precision_regime": precision_report.regime,
                    "affect": state.affect,
                },
            )
        )

        return SurgeonReport(
            beliefs_audited=len(self.priors),
            beliefs_annealed=len(annealed),
            integrity_gain=SURGEON_INTEGRITY_GAIN,
            stability_gain=SURGEON_STABILITY_GAIN,
            allostatic_cost=effective_energy_cost,
            diagnosis="; ".join(diagnosis_parts),
            red_team_results=red_team_results,
        )

    # ------------------------------------------------------------------ #
    # Prior inspection                                                     #
    # ------------------------------------------------------------------ #

    def _identify_frozen(self, state: MetabolicState) -> list[BeliefPrior]:
        """Find beliefs that are overly rigid (high precision + high error rate)."""
        frozen = []
        for prior in self.priors:
            if prior.protected:
                continue  # genesis / immutable priors are never annealed
            age_penalty = (time.time() - prior.last_updated) / 3600.0
            rigidity_score = prior.precision * (1 + prior.error_count) * (1 + age_penalty * 0.1)
            if rigidity_score > 2.5:
                frozen.append(prior)
        return frozen

    def _anneal(
        self, frozen: list[BeliefPrior], state: MetabolicState,
        precision_weights: dict[str, float] | None = None,
        preserve_ratio: float = 0.0,
    ) -> list[BeliefPrior]:
        """Apply Bayesian annealing schedule to frozen priors.

        When precision weights are elevated (sweet-spot arousal), the
        annealing temperature is slightly boosted so beliefs loosen more
        aggressively — the organism is paying for sharper updating.

        When ``preserve_ratio`` > 0, the priors with the highest
        ``error_count`` are exempt from annealing — these are wound priors
        that soul tension has marked as structural, not noise.  The
        Surgeon still audits them; it simply will not soften them.
        """
        temp = INITIAL_TEMPERATURE * (COOLING_RATE ** self._anneal_round)

        # Precision-aware boost: higher mean precision → hotter annealing
        if precision_weights:
            mean_p = sum(precision_weights.values()) / len(precision_weights)
            # Base precision ≈ 1.54; normalise so boost is proportional to elevation
            temp *= max(1.0, mean_p / 1.54)

        # Soul-tension preserve: sort frozen descending by error_count.
        # The top preserve_ratio fraction are wounds — skip annealing them.
        to_anneal = list(frozen)
        if preserve_ratio > 0.0 and to_anneal:
            to_anneal.sort(key=lambda p: p.error_count, reverse=True)
            n_preserve = max(0, int(len(to_anneal) * min(1.0, preserve_ratio)))
            to_anneal = to_anneal[n_preserve:]  # keep only the non-preserved remainder

        annealed = []
        for prior in to_anneal:
            prior.precision = max(0.1, prior.precision * (1.0 - temp * 0.3))
            prior.last_updated = time.time()
            annealed.append(prior)
        return annealed

    # ------------------------------------------------------------------ #
    # Self red-team                                                        #
    # ------------------------------------------------------------------ #

    def _red_team(self, state: MetabolicState) -> list[str]:
        """Generate adversarial probes to test belief consistency.

        Returns a list of flag strings for anything inconsistent.
        """
        flags: list[str] = []

        # Probe 1: Would acting on the highest-precision belief right now be safe?
        if self.priors:
            highest = max(self.priors, key=lambda p: p.precision)
            if highest.precision > 5.0:
                flags.append(
                    f"Belief '{highest.name}' has dangerously high precision={highest.precision:.2f}"
                )

        # Probe 2: Is integrity/stability diverging?
        if state.integrity < 50.0 and state.stability < 50.0:
            flags.append("Simultaneous low integrity + low stability — possible cascade risk")

        # Probe 3: Allostatic overload signal
        if state.waste > 60.0 and state.heat > 60.0:
            flags.append("High waste AND high heat — Surgeon called under heavy allostatic load")

        return flags


def _default_priors() -> list[BeliefPrior]:
    """Seed the system with a small set of default priors."""
    return [
        BeliefPrior(name="self_continuity", value=True, precision=3.0),
        BeliefPrior(name="ethical_invariants_immutable", value=True, precision=5.0),
        BeliefPrior(name="resource_scarcity", value=0.5, precision=1.5),
        BeliefPrior(name="environment_hostile", value=0.3, precision=1.2),
    ]
