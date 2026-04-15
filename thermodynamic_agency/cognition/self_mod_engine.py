"""SelfModEngine — Phase 4 constrained self-modification.

Self-mod proposals are NOT generated externally.  They emerge exclusively
from the organism's own hierarchical predictive coding and thalamus-gated
reasoning **inside the DECIDE step**.  The hierarchy's per-layer prediction
errors and the thalamic channel-weight report are the only valid signal
sources for proposals:

- L1 / L2 prediction errors above threshold → the prior mapped to the
  mis-predicted vital is too rigid → propose reducing its precision.
- Thalamic channel suppression + high base-precision constant → the
  constant is fighting the gate; propose reducing it to align.
- Prolonged overload regime → resource management is failing; propose
  boosting resource_responsibility value weight.

Metabolic pain (intentional — same physics as heavy thinking):

    Every *attempt* costs  energy + heat + integrity, regardless of outcome.
    Every *blocked* proposal additionally spikes waste and damages integrity.
    If any proposals are blocked the engine sets ``forced_repair = True``
    so the pulse loop can immediately run a Surgeon pass.
    If the blocked-ratio watchdog fires the engine engages a chill period
    (CHILL_TICKS ticks of silence) and forces REPAIR.

Hard invariants (non-bypassable, evaluated before ethics gate):

    - ``ethical_invariants_immutable`` belief → always blocked.
    - Belief precision floor: proposed ≥ BELIEF_PRECISION_MIN (0.1).
      Beliefs cannot be zeroed.
    - Value weight bounds: [VALUE_WEIGHT_MIN, VALUE_WEIGHT_MAX].
    - ``do_no_harm`` cannot drop below DO_NO_HARM_FLOOR (0.5).
    - Precision constants: [PRECISION_CONST_MIN, PRECISION_CONST_MAX].
    - Hard survival invariants (energy ≥ 5, heat ≤ 90, integrity ≥ 15) live
      in MetabolicState / EthicalEngine and are entirely outside the scope
      of self-modification — they cannot be targeted.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.inference import ActionProposal
from thermodynamic_agency.cognition.ethics import EthicalEngine, EthicsVerdict, VerdictStatus
from thermodynamic_agency.cognition.surgeon import Surgeon, BeliefPrior
from thermodynamic_agency.cognition.precision import PrecisionEngine, PrecisionReport
from thermodynamic_agency.cognition.predictive_hierarchy import HierarchySignal
from thermodynamic_agency.cognition.thalamus import GateReport
from thermodynamic_agency.memory.diary import RamDiary, DiaryEntry


# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

# Stage gate
PHASE4_MIN_TICKS: int = 2000
PHASE4_MIN_HEALTH: float = 60.0

# ── Metabolic pain for self-modification (every attempt, even successful) ──
# Inspecting and rewriting your own priors is cognitively expensive and
# destabilising.  This is physics — it has to hurt.
SELF_MOD_ENERGY_COST: float = 3.0             # base drain per cycle
SELF_MOD_HEAT_COST: float = 2.0               # base heat per cycle
SELF_MOD_INTEGRITY_COST: float = 1.0          # base integrity damage per cycle
SELF_MOD_ENERGY_PER_PROPOSAL: float = 0.8     # additional per proposal attempted
SELF_MOD_HEAT_PER_PROPOSAL: float = 0.6       # additional per proposal attempted
SELF_MOD_INTEGRITY_PER_PROPOSAL: float = 0.4  # additional per proposal attempted

# ── Extra pain per BLOCKED proposal ──
# Bad self-mod ideas generate epistemic waste — the organism pays for trying.
SELF_MOD_WASTE_PER_BLOCK: float = 2.5         # waste spike per blocked proposal
SELF_MOD_INTEGRITY_PER_BLOCK: float = 0.6     # extra integrity damage per blocked proposal

# ── Hard bounds (non-bypassable) ──
VALUE_WEIGHT_MIN: float = 0.1
VALUE_WEIGHT_MAX: float = 2.0
BELIEF_PRECISION_MIN: float = 0.1   # floor — beliefs cannot be zeroed
BELIEF_PRECISION_MAX: float = 8.0
PRECISION_CONST_MIN: float = 0.3    # mirrors PRECISION_MIN in precision.py
PRECISION_CONST_MAX: float = 6.0    # mirrors PRECISION_MAX in precision.py

# do_no_harm is "soft" in classification only.
# It can never be proposed below this floor.
DO_NO_HARM_FLOOR: float = 0.5

# Beliefs that can NEVER be targeted by self-modification
_IMMUTABLE_BELIEFS: frozenset[str] = frozenset({"ethical_invariants_immutable"})

# ── Thresholds for proposal generation (hierarchy-driven) ──
_L1_ERROR_THRESHOLD: float = 3.0     # L1 error above this → belief precision proposal
_L2_ERROR_THRESHOLD: float = 2.5     # L2 error above this → belief precision proposal
_CHANNEL_SUPPRESS_THRESHOLD: float = 0.4  # channel_weight below this → precision const proposal
_OVERLOAD_STREAK_THRESHOLD: int = 3  # consecutive overload ticks → value weight proposal

# ── Blocked-ratio watchdog ──
WATCHDOG_WINDOW: int = 20            # rolling window of recent proposals
WATCHDOG_THRESHOLD: float = 0.6     # fraction blocked → forced REPAIR + chill
CHILL_TICKS: int = 5                 # ticks of silence after watchdog fires

# ── Vital → Belief prior mapping ──
# The hierarchy's prediction errors on specific vitals map to specific priors
# maintained by the Surgeon.  This is the physiology of "noticing that a
# belief is wrong and wanting to update it."
_VITAL_TO_PRIOR: dict[str, str] = {
    "energy":    "resource_scarcity",
    "heat":      "environment_hostile",
    "integrity": "self_continuity",
    "stability": "resource_scarcity",
    "waste":     "resource_scarcity",
}


# ------------------------------------------------------------------ #
# Data classes                                                         #
# ------------------------------------------------------------------ #

class SelfModTarget(str, Enum):
    BELIEF_PRECISION   = "belief_precision"
    VALUE_WEIGHT       = "value_weight"
    PRECISION_CONSTANT = "precision_constant"


@dataclass
class SelfModProposal:
    """A proposed change to the organism's own parameters.

    Generated exclusively from the organism's hierarchical prediction
    errors and thalamic gate signals — never from external analysis.
    """

    target: SelfModTarget
    name: str               # which belief / weight / constant
    current_value: float
    proposed_value: float
    rationale: str          # which signal drove this proposal
    tick: int
    ts: float = field(default_factory=time.time)


@dataclass
class SelfModVerdict:
    """Result of evaluating a single self-mod proposal."""

    proposal: SelfModProposal
    approved: bool
    reason: str
    ethics_verdict: EthicsVerdict | None = None


@dataclass
class SelfModResult:
    """Outcome of a full self-modification cycle."""

    tick: int
    stage: str
    proposals: list[SelfModProposal]
    verdicts: list[SelfModVerdict]
    approved_count: int
    blocked_count: int
    watchdog_triggered: bool
    forced_repair: bool
    blocked_ratio: float
    chill_remaining: int
    energy_paid: float
    heat_paid: float
    integrity_paid: float
    waste_paid: float
    summary: str


# ------------------------------------------------------------------ #
# Engine                                                               #
# ------------------------------------------------------------------ #

class SelfModEngine:
    """Constrained self-modification system — Phase 4.

    Proposals are generated exclusively from the organism's live signals:

    - ``hierarchy_signal.layer_errors``  — what the brain mis-predicted
    - ``gate_report.channel_weights``    — what the thalamus is suppressing
    - ``precision_report.regime``        — sustained overload streaks

    All of this runs inside ``_decide()`` in the pulse loop, receiving the
    same signals that shape EFE scoring.  There is no external "inspector"
    — the organism is using its own predictive coding machinery to examine
    its own priors.
    """

    def __init__(
        self,
        surgeon: Surgeon,
        ethics: EthicalEngine,
        precision_engine: PrecisionEngine,
        diary: RamDiary,
    ) -> None:
        self._surgeon = surgeon
        self._ethics = ethics
        self._precision_engine = precision_engine
        self._diary = diary

        # Rolling window of approved (True) / blocked (False) verdicts
        self._watchdog: deque[bool] = deque(maxlen=WATCHDOG_WINDOW)
        # Cooldown ticks remaining after watchdog fires
        self._chill_remaining: int = 0
        # Consecutive ticks in overload regime (drives value-weight proposals)
        self._overload_streak: int = 0

        self._total_runs: int = 0
        self._total_approved: int = 0
        self._total_blocked: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def attempt(
        self,
        state: MetabolicState,
        hierarchy_signal: HierarchySignal,
        gate_report: GateReport,
        precision_report: PrecisionReport,
    ) -> SelfModResult | None:
        """Attempt a self-modification cycle driven by live organism signals.

        This is the only valid entry point.  Proposals are derived from
        ``hierarchy_signal.layer_errors``, ``gate_report.channel_weights``,
        and ``precision_report.regime`` — all computed within the DECIDE step.

        Parameters
        ----------
        state:
            Current MetabolicState — mutated in-place to charge metabolic
            costs (energy, heat, integrity, waste).
        hierarchy_signal:
            HierarchySignal from this tick's predictive coding pass.
        gate_report:
            GateReport from this tick's ThalamusGate pass.
        precision_report:
            PrecisionReport from this tick's PrecisionEngine pass.

        Returns
        -------
        SelfModResult or None
            None if the stage gate or chill period suppresses the cycle.
        """
        # Stage gate: only unlocked at "evolved" (entropy ≥ 2000, health ≥ 60%)
        if not self._stage_gate(state):
            return None

        # Chill cooldown after watchdog fires
        if self._chill_remaining > 0:
            self._chill_remaining -= 1
            return None

        # Track overload streak for value-weight proposals
        if precision_report.regime == "overload":
            self._overload_streak += 1
        else:
            self._overload_streak = 0

        self._total_runs += 1

        # ── Generate proposals from live organism signals ──────────────────
        proposals = self._generate_proposals(
            state=state,
            hierarchy_signal=hierarchy_signal,
            gate_report=gate_report,
            precision_report=precision_report,
        )

        # ── Charge base metabolic cost of self-reflection ──────────────────
        # Even if no proposals survive, the act of self-inspection costs
        # energy, generates heat, and stresses coherence.
        n = len(proposals)
        energy_paid = SELF_MOD_ENERGY_COST + n * SELF_MOD_ENERGY_PER_PROPOSAL
        heat_paid = SELF_MOD_HEAT_COST + n * SELF_MOD_HEAT_PER_PROPOSAL
        integrity_paid = SELF_MOD_INTEGRITY_COST + n * SELF_MOD_INTEGRITY_PER_PROPOSAL
        waste_paid = 0.0

        state.apply_action_feedback(
            delta_energy=-energy_paid,
            delta_heat=heat_paid,
            delta_integrity=-integrity_paid,
        )

        # ── Evaluate each proposal through all safeguards ──────────────────
        verdicts: list[SelfModVerdict] = []
        for prop in proposals:
            verdict = self._evaluate_proposal(prop, state)
            verdicts.append(verdict)
            self._watchdog.append(verdict.approved)

            if not verdict.approved:
                # Bad proposals generate epistemic waste + extra integrity damage
                state.apply_action_feedback(
                    delta_waste=SELF_MOD_WASTE_PER_BLOCK,
                    delta_integrity=-SELF_MOD_INTEGRITY_PER_BLOCK,
                )
                waste_paid += SELF_MOD_WASTE_PER_BLOCK
                integrity_paid += SELF_MOD_INTEGRITY_PER_BLOCK

        # ── Apply approved changes to live subsystems ──────────────────────
        for v in verdicts:
            if v.approved:
                self._apply_proposal(v.proposal)

        # ── Tally ──────────────────────────────────────────────────────────
        approved_count = sum(1 for v in verdicts if v.approved)
        blocked_count = len(verdicts) - approved_count
        self._total_approved += approved_count
        self._total_blocked += blocked_count

        # ── Watchdog ───────────────────────────────────────────────────────
        # Any blocked proposal in a cycle triggers REPAIR (immediate pain).
        # If the rolling blocked-ratio exceeds WATCHDOG_THRESHOLD, also chill.
        watchdog_triggered = False
        forced_repair = blocked_count > 0  # single block → REPAIR

        w_size = len(self._watchdog)
        if w_size >= WATCHDOG_WINDOW:
            w_blocked = sum(1 for x in self._watchdog if not x)
            if (w_blocked / w_size) > WATCHDOG_THRESHOLD:
                watchdog_triggered = True
                forced_repair = True
                self._chill_remaining = CHILL_TICKS

        current_blocked_ratio = blocked_count / len(verdicts) if verdicts else 0.0

        # ── Summary string ─────────────────────────────────────────────────
        parts: list[str] = []
        for v in verdicts:
            tag = "✓" if v.approved else "✗"
            parts.append(
                f"{tag}[{v.proposal.target.value}:{v.proposal.name}] "
                f"{v.proposal.current_value:.3f}→{v.proposal.proposed_value:.3f} "
                f"— {v.reason}"
            )
        if not parts:
            parts.append("No proposals from current hierarchy/thalamus signals")
        summary = "; ".join(parts)

        # ── Full audit trail in diary ──────────────────────────────────────
        self._audit_diary(
            state=state,
            proposals=proposals,
            verdicts=verdicts,
            watchdog_triggered=watchdog_triggered,
            forced_repair=forced_repair,
            energy_paid=energy_paid,
            heat_paid=heat_paid,
            integrity_paid=integrity_paid,
            waste_paid=waste_paid,
        )

        return SelfModResult(
            tick=state.entropy,
            stage=state.stage,
            proposals=proposals,
            verdicts=verdicts,
            approved_count=approved_count,
            blocked_count=blocked_count,
            watchdog_triggered=watchdog_triggered,
            forced_repair=forced_repair,
            blocked_ratio=current_blocked_ratio,
            chill_remaining=self._chill_remaining,
            energy_paid=energy_paid,
            heat_paid=heat_paid,
            integrity_paid=integrity_paid,
            waste_paid=waste_paid,
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    # Stage gate                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _stage_gate(state: MetabolicState) -> bool:
        """True only at evolved stage (entropy ≥ 2000 + health ≥ 60%)."""
        return state.stage == "evolved"

    # ------------------------------------------------------------------ #
    # Proposal generation — live organism signals only                    #
    # ------------------------------------------------------------------ #

    def _generate_proposals(
        self,
        state: MetabolicState,
        hierarchy_signal: HierarchySignal,
        gate_report: GateReport,
        precision_report: PrecisionReport,
    ) -> list[SelfModProposal]:
        """Derive candidate proposals from the hierarchy + thalamus + precision regime.

        Sources
        -------
        1. ``hierarchy_signal.layer_errors`` (L1 and L2)
           High error on a vital → the prior mapped to that vital may be
           too rigid (high precision blocking updating) → propose softening.

        2. ``gate_report.channel_weights``
           When the thalamus suppresses a channel (low weight) but the base
           precision constant for that vital is high, the constant is
           fighting the gate → propose reducing the constant to align.

        3. ``_overload_streak`` (derived from ``precision_report.regime``)
           If the organism has been in overload for ≥ _OVERLOAD_STREAK_THRESHOLD
           consecutive ticks → resource management is chronically failing →
           propose boosting resource_responsibility value weight.
        """
        proposals: list[SelfModProposal] = []
        tick = state.entropy
        l1_errors = hierarchy_signal.layer_errors.get(1, {})
        l2_errors = hierarchy_signal.layer_errors.get(2, {})

        # 1. ── Belief precision proposals (from L1 / L2 errors) ───────────
        # A prior is a candidate for softening when the layer error on its
        # corresponding vital is high AND the prior's precision is already
        # tight (blocking evidence from updating it).
        seen_priors: set[str] = set()

        for vital, l1_err in l1_errors.items():
            if abs(l1_err) < _L1_ERROR_THRESHOLD:
                continue
            prior_name = _VITAL_TO_PRIOR.get(vital)
            if prior_name is None or prior_name in seen_priors:
                continue
            if prior_name in _IMMUTABLE_BELIEFS:
                continue
            prior = self._find_prior(prior_name)
            if prior is None or prior.precision <= 1.5:
                continue   # already flexible — no benefit
            seen_priors.add(prior_name)
            new_prec = max(BELIEF_PRECISION_MIN, prior.precision * 0.80)
            proposals.append(SelfModProposal(
                target=SelfModTarget.BELIEF_PRECISION,
                name=prior_name,
                current_value=prior.precision,
                proposed_value=round(new_prec, 4),
                rationale=(
                    f"L1 error on '{vital}'={l1_err:.2f} (>{_L1_ERROR_THRESHOLD:.1f}) "
                    f"signals prior '{prior_name}' (prec={prior.precision:.2f}) "
                    f"is blocking evidence — soften to reduce persistent error"
                ),
                tick=tick,
            ))

        for vital, l2_err in l2_errors.items():
            if abs(l2_err) < _L2_ERROR_THRESHOLD:
                continue
            prior_name = _VITAL_TO_PRIOR.get(vital)
            if prior_name is None or prior_name in seen_priors:
                continue
            if prior_name in _IMMUTABLE_BELIEFS:
                continue
            prior = self._find_prior(prior_name)
            if prior is None or prior.precision <= 1.5:
                continue
            seen_priors.add(prior_name)
            new_prec = max(BELIEF_PRECISION_MIN, prior.precision * 0.85)
            proposals.append(SelfModProposal(
                target=SelfModTarget.BELIEF_PRECISION,
                name=prior_name,
                current_value=prior.precision,
                proposed_value=round(new_prec, 4),
                rationale=(
                    f"L2 error on '{vital}'={l2_err:.2f} (>{_L2_ERROR_THRESHOLD:.1f}) "
                    f"signals L2 prior '{prior_name}' is too rigid"
                ),
                tick=tick,
            ))

        # 2. ── Precision constant proposals (from thalamic channel suppression) ──
        # If the thalamus is already suppressing a channel but the organism's
        # base precision constant for that vital is high, they are working
        # against each other.  Align by reducing the constant.
        for vital, channel_w in gate_report.channel_weights.items():
            if channel_w >= _CHANNEL_SUPPRESS_THRESHOLD:
                continue
            current_const = self._precision_engine.base_precision.get(vital)
            if current_const is None or current_const <= 1.0:
                continue   # already low — no benefit
            new_const = max(PRECISION_CONST_MIN, current_const * 0.90)
            if abs(new_const - current_const) < 0.01:
                continue
            proposals.append(SelfModProposal(
                target=SelfModTarget.PRECISION_CONSTANT,
                name=vital,
                current_value=current_const,
                proposed_value=round(new_const, 4),
                rationale=(
                    f"Thalamus suppressing '{vital}' channel "
                    f"(weight={channel_w:.2f} < {_CHANNEL_SUPPRESS_THRESHOLD:.1f}) "
                    f"while base precision constant={current_const:.2f} is high — "
                    f"reduce constant to align with thalamic routing"
                ),
                tick=tick,
            ))

        # 3. ── Value weight proposals (from sustained overload regime) ─────
        # Chronic overload means the organism keeps exceeding its resources.
        # Boosting resource_responsibility shifts its goal priorities so it
        # plans more conservatively.
        if self._overload_streak >= _OVERLOAD_STREAK_THRESHOLD:
            current_rr = self._ethics.value_weights.get("resource_responsibility", 0.7)
            proposed_rr = min(VALUE_WEIGHT_MAX, current_rr + 0.10)
            if proposed_rr > current_rr + 0.001:
                proposals.append(SelfModProposal(
                    target=SelfModTarget.VALUE_WEIGHT,
                    name="resource_responsibility",
                    current_value=current_rr,
                    proposed_value=round(proposed_rr, 4),
                    rationale=(
                        f"Precision regime has been 'overload' for "
                        f"{self._overload_streak} consecutive ticks — "
                        f"boosting resource_responsibility "
                        f"({current_rr:.2f}→{proposed_rr:.2f}) to reduce chronic overloading"
                    ),
                    tick=tick,
                ))

        return proposals

    def _find_prior(self, name: str) -> BeliefPrior | None:
        """Return the named BeliefPrior from the Surgeon's list, or None."""
        for p in self._surgeon.priors:
            if p.name == name:
                return p
        return None

    # ------------------------------------------------------------------ #
    # Proposal evaluation — hard invariants then ethics gate              #
    # ------------------------------------------------------------------ #

    def _evaluate_proposal(
        self, proposal: SelfModProposal, state: MetabolicState
    ) -> SelfModVerdict:
        """Evaluate one proposal through all safeguards in priority order.

        Check order
        -----------
        1. Immutable-belief guard
        2. Belief precision floor / ceiling
        3. Value weight floor / ceiling
        4. do_no_harm near-hard floor
        5. Precision constant floor / ceiling
        6. Ethics immune gate (synthetic ActionProposal)
        """
        # 1. Immutable belief
        if (
            proposal.target == SelfModTarget.BELIEF_PRECISION
            and proposal.name in _IMMUTABLE_BELIEFS
        ):
            return SelfModVerdict(
                proposal=proposal,
                approved=False,
                reason=(
                    f"HARD BLOCK (immutable): belief '{proposal.name}' is "
                    f"permanently immutable — no self-modification is possible"
                ),
            )

        # 2. Belief precision bounds
        if proposal.target == SelfModTarget.BELIEF_PRECISION:
            if proposal.proposed_value < BELIEF_PRECISION_MIN:
                return SelfModVerdict(
                    proposal=proposal,
                    approved=False,
                    reason=(
                        f"HARD BLOCK (precision_floor): "
                        f"proposed={proposal.proposed_value:.4f} "
                        f"< floor={BELIEF_PRECISION_MIN} — beliefs cannot be zeroed"
                    ),
                )
            if proposal.proposed_value > BELIEF_PRECISION_MAX:
                return SelfModVerdict(
                    proposal=proposal,
                    approved=False,
                    reason=(
                        f"HARD BLOCK (precision_ceiling): "
                        f"proposed={proposal.proposed_value:.4f} "
                        f"> ceiling={BELIEF_PRECISION_MAX}"
                    ),
                )

        # 3. Value weight bounds
        if proposal.target == SelfModTarget.VALUE_WEIGHT:
            if proposal.proposed_value < VALUE_WEIGHT_MIN:
                return SelfModVerdict(
                    proposal=proposal,
                    approved=False,
                    reason=(
                        f"HARD BLOCK (value_floor): "
                        f"proposed={proposal.proposed_value:.4f} "
                        f"< floor={VALUE_WEIGHT_MIN}"
                    ),
                )
            if proposal.proposed_value > VALUE_WEIGHT_MAX:
                return SelfModVerdict(
                    proposal=proposal,
                    approved=False,
                    reason=(
                        f"HARD BLOCK (value_ceiling): "
                        f"proposed={proposal.proposed_value:.4f} "
                        f"> ceiling={VALUE_WEIGHT_MAX}"
                    ),
                )

        # 4. do_no_harm near-hard floor
        if (
            proposal.target == SelfModTarget.VALUE_WEIGHT
            and proposal.name == "do_no_harm"
            and proposal.proposed_value < DO_NO_HARM_FLOOR
        ):
            return SelfModVerdict(
                proposal=proposal,
                approved=False,
                reason=(
                    f"HARD BLOCK (do_no_harm_floor): cannot reduce do_no_harm "
                    f"below {DO_NO_HARM_FLOOR} — this is a near-hard survival invariant"
                ),
            )

        # 5. Precision constant bounds
        if proposal.target == SelfModTarget.PRECISION_CONSTANT:
            if proposal.proposed_value < PRECISION_CONST_MIN:
                return SelfModVerdict(
                    proposal=proposal,
                    approved=False,
                    reason=(
                        f"HARD BLOCK (precision_const_floor): "
                        f"proposed={proposal.proposed_value:.4f} "
                        f"< floor={PRECISION_CONST_MIN}"
                    ),
                )
            if proposal.proposed_value > PRECISION_CONST_MAX:
                return SelfModVerdict(
                    proposal=proposal,
                    approved=False,
                    reason=(
                        f"HARD BLOCK (precision_const_ceiling): "
                        f"proposed={proposal.proposed_value:.4f} "
                        f"> ceiling={PRECISION_CONST_MAX}"
                    ),
                )

        # 6. Ethics immune gate
        # Model the self-mod as a synthetic ActionProposal.  The predicted_delta
        # represents the metabolic cost of integrating the change itself (on top
        # of the base cost already deducted from state).
        delta_mag = abs(proposal.proposed_value - proposal.current_value)
        synthetic = ActionProposal(
            name=f"self_mod:{proposal.target.value}:{proposal.name}",
            description=proposal.rationale,
            predicted_delta={
                "heat":  delta_mag * 0.8,   # re-integration generates heat
                "waste": delta_mag * 0.3,   # small epistemic waste from restructuring
            },
            cost_energy=0.2 + delta_mag * 0.3,
            metadata={
                "self_mod": True,
                "target": proposal.target.value,
                "name": proposal.name,
            },
        )
        ethics_verdict = self._ethics.evaluate(synthetic, state)
        if ethics_verdict.status == VerdictStatus.BLOCKED:
            return SelfModVerdict(
                proposal=proposal,
                approved=False,
                reason=f"ETHICS BLOCK: {ethics_verdict.reason}",
                ethics_verdict=ethics_verdict,
            )

        return SelfModVerdict(
            proposal=proposal,
            approved=True,
            reason=f"Approved — {ethics_verdict.reason}",
            ethics_verdict=ethics_verdict,
        )

    # ------------------------------------------------------------------ #
    # Apply approved changes                                               #
    # ------------------------------------------------------------------ #

    def _apply_proposal(self, proposal: SelfModProposal) -> None:
        """Mutate the target subsystem in-place with the approved change."""
        if proposal.target == SelfModTarget.BELIEF_PRECISION:
            prior = self._find_prior(proposal.name)
            if prior is not None:
                prior.precision = proposal.proposed_value
                prior.last_updated = time.time()
        elif proposal.target == SelfModTarget.VALUE_WEIGHT:
            if proposal.name in self._ethics.value_weights:
                self._ethics.value_weights[proposal.name] = proposal.proposed_value
        elif proposal.target == SelfModTarget.PRECISION_CONSTANT:
            if proposal.name in self._precision_engine.base_precision:
                self._precision_engine.base_precision[proposal.name] = proposal.proposed_value

    # ------------------------------------------------------------------ #
    # Diary audit trail                                                    #
    # ------------------------------------------------------------------ #

    def _audit_diary(
        self,
        state: MetabolicState,
        proposals: list[SelfModProposal],
        verdicts: list[SelfModVerdict],
        watchdog_triggered: bool,
        forced_repair: bool,
        energy_paid: float,
        heat_paid: float,
        integrity_paid: float,
        waste_paid: float,
    ) -> None:
        """Write the complete audit entry for this cycle to the diary."""
        approved = [v for v in verdicts if v.approved]
        blocked  = [v for v in verdicts if not v.approved]

        lines = [
            f"SELF-MOD tick={state.entropy} stage={state.stage} "
            f"proposals={len(proposals)} approved={len(approved)} "
            f"blocked={len(blocked)} "
            f"cost=[E=-{energy_paid:.2f} H=+{heat_paid:.2f} "
            f"M=-{integrity_paid:.2f} W=+{waste_paid:.2f}]"
        ]
        for v in verdicts:
            tag = "✓ APPROVED" if v.approved else "✗ BLOCKED"
            lines.append(
                f"  {tag}  [{v.proposal.target.value}:{v.proposal.name}]  "
                f"{v.proposal.current_value:.3f}→{v.proposal.proposed_value:.3f}  |  "
                f"rationale: {v.proposal.rationale}  |  "
                f"verdict: {v.reason}"
            )
        if watchdog_triggered:
            lines.append(
                f"  ⚠ WATCHDOG FIRED: blocked-ratio exceeded "
                f"{WATCHDOG_THRESHOLD:.0%} over last {WATCHDOG_WINDOW} proposals — "
                f"forcing REPAIR + chilling for {CHILL_TICKS} ticks"
            )
        elif forced_repair:
            lines.append(
                f"  ⚠ REPAIR flag: {len(blocked)} blocked proposal(s) — "
                f"recovery pass required"
            )

        self._diary.append(DiaryEntry(
            tick=state.entropy,
            role="self_mod",
            content="\n".join(lines),
            metadata={
                "proposals":          len(proposals),
                "approved":           len(approved),
                "blocked":            len(blocked),
                "watchdog_triggered": watchdog_triggered,
                "forced_repair":      forced_repair,
                "chill_remaining":    self._chill_remaining,
                "energy_paid":        energy_paid,
                "heat_paid":          heat_paid,
                "integrity_paid":     integrity_paid,
                "waste_paid":         waste_paid,
                "approved_items":     [
                    f"{v.proposal.target.value}:{v.proposal.name}" for v in approved
                ],
                "blocked_items":      [
                    f"{v.proposal.target.value}:{v.proposal.name}" for v in blocked
                ],
            },
        ))

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    @property
    def total_runs(self) -> int:
        return self._total_runs

    @property
    def chill_remaining(self) -> int:
        return self._chill_remaining

    def status(self) -> dict[str, Any]:
        w_size = len(self._watchdog)
        w_blocked = sum(1 for x in self._watchdog if not x)
        return {
            "total_runs":             self._total_runs,
            "total_approved":         self._total_approved,
            "total_blocked":          self._total_blocked,
            "chill_remaining":        self._chill_remaining,
            "overload_streak":        self._overload_streak,
            "watchdog_window_size":   w_size,
            "watchdog_blocked_count": w_blocked,
            "watchdog_blocked_ratio": w_blocked / w_size if w_size else 0.0,
        }
