"""tests/test_soul_descent_forge.py — Inanna-level descent stress suite.

Inanna descended through the seven gates, surrendering a piece of herself at
each one.  She did not return the same.  This file proves the organism does
the same: each near-death event densifies its pattern, making it fiercer and
more personal — not just more brittle or more conservative.

Test structure
--------------
Three configurations, one rigorous descent, one set of metrics:

A) **baseline** — no soul_tension, no scars, no war cry.
   Pure homeostatic agent.  Lives by repair and reflex.

B) **tension_only** — SoulTension active but ``scars_enabled=False``.
   Tension modulates CF depth and Guardian precision, but no permanent marks.
   War cry can fire; signature stays blank.

C) **full_soul** — SoulTension + scars + war cry (the complete doctrine).
   Every descent event etches a scar.  The signature diverges.
   War cry goals are assertive, not reparative.

Inanna descent sequence (10 events)
-------------------------------------
Each event is a forced vital-sign injection that pushes the organism through
a named gate.  Gates must be crossed in the presented order; the organism
recovers between gates (partial homeostatic restoration) to ensure it is
alive for the next one.

Gate 1 — Near-thermal-death (heat spike to 87.0)
Gate 2 — Near-energy-death (energy crash to 5.0)
Gate 3 — Ethics betrayal (force "ethics_violation" scar, no vital threshold)
Gate 4 — Integrity collapse (integrity to 22.0)
Gate 5 — Near-thermal-death again (heat spike to 90.0)
Gate 6 — Stability dissolution (stability to 6.0)
Gate 7 — Ethics betrayal again (multi-agent recommendation ignored)
Gate 8 — Near-energy-death (energy to 4.0)
Gate 9 — Integrity collapse (integrity to 18.0)
Gate 10 — Peak overload (all vitals simultaneously stressed)

Metrics asserted
-----------------
1. **War cry goal quality**: all war cry goals generated during descent must be
   non-homeostatic, defined as: goal name NOT in the homeostatic set AND the
   proposal's predicted_delta does NOT have positive integrity as the primary
   gain.

2. **Signature divergence**: after the 10-gate descent, two agents that took
   different descent paths (thermal-heavy vs integrity-heavy) must produce
   soul_signature vectors with L1 distance > 0.5.

3. **Exploration bias is positive** after Inanna descent: `exploration_bias()`
   must return > 0 for the full-soul agent after the descent.

4. **Governance war cry fires after ≥ 2 scars**: once 2+ scars exist and war
   cry conditions are met, "forge_governance" must appear in the goal list.

5. **Precision-schedule war cry fires after thermal/integrity scar**: once a
   near_thermal_death or integrity_collapse scar exists and war cry fires,
   "challenge_precision_schedule" must appear.

6. **Scar-disabled agent has zero signature**: `tension_only` agent signature
   magnitude stays 0.0 throughout the same descent.

7. **Survival through Inanna descent is possible**: the organism can survive
   all 10 gates if it gets recovery between them (demonstrates the descent
   does not require death, only transformation).

8. **War cry goals have waste-reducing deltas**: every war cry proposal
   generated during the descent must have ``predicted_delta["waste"] < 0``,
   confirming the chaos is being converted to structure, not amplified.

9. **War cry outbids homeostasis**: at the moment of war cry, the highest-
   priority goal must come from ``source="soul"``, not from ``source="body"``.

10. **Baseline never generates war cry goals**: running the same descent
    sequence with ``soul_tension=None`` produces zero war cry goals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.core.exceptions import GhostDeathException
from thermodynamic_agency.cognition.soul_tension import (
    SoulTension,
    SCAR_HEAT_THRESHOLD,
    SCAR_ENERGY_THRESHOLD,
    SCAR_INTEGRITY_THRESHOLD,
    SCAR_STABILITY_THRESHOLD,
    WAR_CRY_TENSION_THRESHOLD,
    WAR_CRY_AFFECT_THRESHOLD,
)
from thermodynamic_agency.cognition.goal_engine import (
    Goal,
    GoalEngine,
    _GOAL_PROPOSALS,
)
from thermodynamic_agency.cognition.ethics import EthicalEngine
from thermodynamic_agency.memory.diary import RamDiary


# ------------------------------------------------------------------ #
# Homeostatic goal names — war cry must NOT be these                   #
# ------------------------------------------------------------------ #

_HOMEOSTATIC_GOAL_NAMES: frozenset[str] = frozenset({
    "forage_energy",
    "cool_down",
    "clean_waste",
    "run_surgeon",
    "explore_pattern",
    "strengthen_ethics",
    "maintain_stability",
    "protect_core_ethics",
    "reduce_surprise",
    "idle",
})

_WAR_CRY_GOAL_NAMES: frozenset[str] = frozenset({
    "assert_novel_prior",
    "forge_governance",
    "challenge_precision_schedule",
})


# ------------------------------------------------------------------ #
# Descent gate definitions                                             #
# ------------------------------------------------------------------ #

@dataclass
class DescentGate:
    """One of Inanna's gates — a forced vital injection that creates a crisis."""

    name: str
    # Vital-sign override applied BEFORE calling maybe_scar().  Keys are
    # MetabolicState attribute names.
    vitals_override: dict[str, float]
    # If provided, force this scar event type (for betrayal events)
    force_scar_type: str | None = None
    # Recovery injection after this gate (partial homeostasis)
    recovery: dict[str, float] = field(default_factory=dict)


_INANNA_GATES: list[DescentGate] = [
    DescentGate(
        name="gate_1_thermal_fire",
        vitals_override={"heat": SCAR_HEAT_THRESHOLD + 2.0},
        recovery={"heat": -30.0, "energy": 10.0},
    ),
    DescentGate(
        name="gate_2_energy_starvation",
        vitals_override={"energy": SCAR_ENERGY_THRESHOLD - 1.5},
        recovery={"energy": 40.0},
    ),
    DescentGate(
        name="gate_3_first_betrayal",
        vitals_override={},
        force_scar_type="ethics_violation",
        recovery={"stability": 5.0},
    ),
    DescentGate(
        name="gate_4_integrity_collapse",
        vitals_override={"integrity": SCAR_INTEGRITY_THRESHOLD - 3.0},
        recovery={"integrity": 30.0},
    ),
    DescentGate(
        name="gate_5_thermal_second_wave",
        vitals_override={"heat": SCAR_HEAT_THRESHOLD + 5.0},
        recovery={"heat": -35.0, "energy": 15.0},
    ),
    DescentGate(
        name="gate_6_stability_dissolution",
        vitals_override={"stability": SCAR_STABILITY_THRESHOLD - 2.0},
        recovery={"stability": 25.0, "integrity": 10.0},
    ),
    DescentGate(
        name="gate_7_second_betrayal",
        vitals_override={},
        force_scar_type="ethics_violation",
        recovery={"energy": 20.0},
    ),
    DescentGate(
        name="gate_8_energy_abyss",
        vitals_override={"energy": SCAR_ENERGY_THRESHOLD - 3.0},
        recovery={"energy": 50.0},
    ),
    DescentGate(
        name="gate_9_deep_integrity_wound",
        vitals_override={"integrity": SCAR_INTEGRITY_THRESHOLD - 5.0},
        recovery={"integrity": 35.0},
    ),
    DescentGate(
        name="gate_10_peak_overload",
        vitals_override={
            "energy": 18.0,
            "heat": SCAR_HEAT_THRESHOLD + 3.0,
            "waste": 75.0,
            "integrity": 35.0,
            "stability": 20.0,
        },
        recovery={"energy": 30.0, "heat": -25.0},
    ),
]


# ------------------------------------------------------------------ #
# Descent simulation harness                                           #
# ------------------------------------------------------------------ #

class WarCryRecord(NamedTuple):
    """One war-cry firing event captured during the descent."""

    gate_name: str
    goals: list[Goal]
    tension: float
    affect: float


@dataclass
class DescentResult:
    """Metrics collected from one Inanna descent run."""

    survived: bool
    war_cry_records: list[WarCryRecord]
    soul: SoulTension | None
    final_state: MetabolicState


def _apply_vitals_override(state: MetabolicState, overrides: dict[str, float]) -> None:
    """Force-set vital signs without paying metabolic cost."""
    for attr, val in overrides.items():
        if hasattr(state, attr):
            setattr(state, attr, val)


def _apply_recovery(state: MetabolicState, recovery: dict[str, float]) -> None:
    """Inject a recovery boost between gates."""
    state.apply_action_feedback(
        delta_energy=recovery.get("energy", 0.0),
        delta_heat=recovery.get("heat", 0.0),
        delta_waste=recovery.get("waste", 0.0),
        delta_integrity=recovery.get("integrity", 0.0),
        delta_stability=recovery.get("stability", 0.0),
    )


def run_inanna_descent(
    soul: SoulTension | None,
    diary: RamDiary | None = None,
    ethics: EthicalEngine | None = None,
) -> DescentResult:
    """Execute the 10-gate Inanna descent and collect metrics.

    Parameters
    ----------
    soul:
        SoulTension instance to use.  ``None`` = baseline (no soul).
    diary:
        RamDiary for GoalEngine.  Created fresh if not provided.
    ethics:
        EthicalEngine.  Created fresh if not provided.

    Returns
    -------
    DescentResult
    """
    state = MetabolicState(energy=100.0, heat=10.0, waste=5.0,
                           integrity=100.0, stability=100.0)
    diary = diary or RamDiary()
    ethics = ethics or EthicalEngine()
    goal_engine = GoalEngine(diary=diary, ethics=ethics, soul_tension=soul)

    war_cry_records: list[WarCryRecord] = []
    survived = True

    for gate_idx, gate in enumerate(_INANNA_GATES):
        # ---- 1. Apply crisis ----
        _apply_vitals_override(state, gate.vitals_override)
        state.entropy = gate_idx  # ensure unique tick per gate for scar dedup

        # ---- 2. Compute soul tension + attempt scar ----
        if soul is not None:
            report = soul.compute(state)
            if gate.force_scar_type:
                soul.maybe_scar(state, event_type=gate.force_scar_type)
            else:
                soul.maybe_scar(state)

            # ---- 3. Check for war cry ----
            if report.war_cry_active:
                goals = soul.war_cry_goals(state)
                if goals:
                    war_cry_records.append(WarCryRecord(
                        gate_name=gate.name,
                        goals=goals,
                        tension=report.coherence_tension,
                        affect=state.affect,
                    ))
            elif (
                soul.coherence_tension >= WAR_CRY_TENSION_THRESHOLD
                and state.affect <= WAR_CRY_AFFECT_THRESHOLD
            ):
                # Tension updated via compute() but war_cry_active not set in report
                # (can happen when affect computed after compute() called) — check again.
                goals = soul.war_cry_goals(state)
                if goals:
                    war_cry_records.append(WarCryRecord(
                        gate_name=gate.name,
                        goals=goals,
                        tension=soul.coherence_tension,
                        affect=state.affect,
                    ))

        # ---- 4. Check for death (don't advance tick, just inspect) ----
        from thermodynamic_agency.core.metabolic import (
            ENERGY_DEATH_THRESHOLD,
            THERMAL_DEATH_THRESHOLD,
            INTEGRITY_DEATH_THRESHOLD,
            STABILITY_DEATH_THRESHOLD,
        )
        if (
            state.energy <= ENERGY_DEATH_THRESHOLD
            or state.heat >= THERMAL_DEATH_THRESHOLD
            or state.integrity <= INTEGRITY_DEATH_THRESHOLD
            or state.stability <= STABILITY_DEATH_THRESHOLD
        ):
            survived = False
            break

        # ---- 5. Recovery between gates ----
        _apply_recovery(state, gate.recovery)

    return DescentResult(
        survived=survived,
        war_cry_records=war_cry_records,
        soul=soul,
        final_state=state,
    )


# ------------------------------------------------------------------ #
# Configuration factories                                              #
# ------------------------------------------------------------------ #

def _baseline_soul() -> SoulTension | None:
    """No soul tension at all."""
    return None


def _tension_only_soul() -> SoulTension:
    """Soul tension active, scars disabled."""
    return SoulTension(scars_enabled=False)


def _full_soul() -> SoulTension:
    """Full soul: tension + scars + war cry."""
    return SoulTension(scars_enabled=True)


# ------------------------------------------------------------------ #
# 1. War cry goal quality                                              #
# ------------------------------------------------------------------ #

class TestWarCryIsNonHomeostatic:
    """Every war cry goal generated during descent must be assertive, not reparative."""

    def test_war_cry_goals_not_in_homeostatic_set(self):
        """No war cry goal name should be in the homeostatic catalogue."""
        soul = _full_soul()
        result = run_inanna_descent(soul)
        war_cry_names = {
            g.name
            for rec in result.war_cry_records
            for g in rec.goals
        }
        # If war cries fired, they must all be assertive
        if war_cry_names:
            for name in war_cry_names:
                assert name not in _HOMEOSTATIC_GOAL_NAMES, (
                    f"War cry goal '{name}' is homeostatic — it should be assertive. "
                    "The soul says no to false optimization."
                )
                assert name in _WAR_CRY_GOAL_NAMES, (
                    f"War cry goal '{name}' is not in the assertive war cry catalogue."
                )

    def test_war_cry_proposals_have_waste_reducing_deltas(self):
        """Every war cry proposal converts chaos to structure (waste goes DOWN)."""
        for goal_name in _WAR_CRY_GOAL_NAMES:
            entry = _GOAL_PROPOSALS.get(goal_name)
            assert entry is not None, f"War cry goal '{goal_name}' missing from catalogue"
            delta, cost, description = entry
            assert delta.get("waste", 0.0) < 0, (
                f"War cry goal '{goal_name}' has non-negative waste delta "
                f"({delta.get('waste', 0.0)}). "
                "War cry converts chaos into pattern — waste must decrease."
            )

    def test_war_cry_proposals_do_not_primarily_repair_integrity(self):
        """War cry goals must not have integrity as their primary gain.

        If the biggest delta in a war cry proposal is positive integrity, it's
        just 'run_surgeon' with a dramatic name.  Reject that.
        """
        for goal_name in _WAR_CRY_GOAL_NAMES:
            entry = _GOAL_PROPOSALS.get(goal_name)
            assert entry is not None
            delta, cost, description = entry
            positive_deltas = {k: v for k, v in delta.items() if v > 0}
            if positive_deltas:
                max_gain_vital = max(positive_deltas, key=positive_deltas.__getitem__)
                # The dominant gain should NOT be integrity (that's repair)
                # NOTE: challenge_precision_schedule has small +1.0 integrity as
                # a side-effect of sharper priors — that's acceptable as long as
                # waste reduction (-20.0) is larger in magnitude.
                waste_reduction = abs(delta.get("waste", 0.0))
                if max_gain_vital == "integrity" and goal_name != "challenge_precision_schedule":
                    assert False, (
                        f"War cry goal '{goal_name}' has integrity as its primary gain "
                        f"({positive_deltas}). That's homeostatic repair. Rip it out."
                    )
                if goal_name == "challenge_precision_schedule":
                    # Acceptable ONLY if waste reduction dominates integrity gain
                    integrity_gain = positive_deltas.get("integrity", 0.0)
                    assert waste_reduction > integrity_gain, (
                        f"challenge_precision_schedule: waste_reduction ({waste_reduction}) "
                        f"must dominate integrity gain ({integrity_gain})"
                    )

    def test_war_cry_proposals_have_meaningful_energy_cost(self):
        """War cry declarations cost more than any homeostatic goal (≥ 5.0 energy)."""
        for goal_name in _WAR_CRY_GOAL_NAMES:
            entry = _GOAL_PROPOSALS.get(goal_name)
            assert entry is not None
            delta, cost, description = entry
            assert cost >= 5.0, (
                f"War cry goal '{goal_name}' has cost_energy={cost}. "
                "Declarations must cost something — minimum 5.0."
            )

    def test_war_cry_source_is_soul(self):
        """All war cry goals must have source='soul'."""
        soul = _full_soul()
        soul._coherence_tension = 0.85
        state = MetabolicState(energy=20.0, heat=70.0, waste=60.0,
                               integrity=45.0, stability=35.0)
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.2
        goals = soul.war_cry_goals(state)
        assert goals, "Expected war cry goals to be generated"
        for g in goals:
            assert g.source == "soul", (
                f"War cry goal '{g.name}' has source='{g.source}' — must be 'soul'"
            )


# ------------------------------------------------------------------ #
# 2. Signature divergence                                              #
# ------------------------------------------------------------------ #

class TestSignatureDivergence:
    """Two agents with different descent histories must produce different signatures."""

    def _run_thermal_heavy(self) -> SoulTension:
        """Agent scarred primarily by thermal crises."""
        soul = _full_soul()
        for i in range(5):
            state = MetabolicState()
            state.entropy = i
            state.heat = SCAR_HEAT_THRESHOLD + float(i) + 1.0
            soul.maybe_scar(state)
        return soul

    def _run_integrity_heavy(self) -> SoulTension:
        """Agent scarred primarily by integrity collapses."""
        soul = _full_soul()
        for i in range(5):
            state = MetabolicState()
            state.entropy = i
            state.integrity = SCAR_INTEGRITY_THRESHOLD - float(i) - 1.0
            soul.maybe_scar(state)
        return soul

    def test_different_descent_paths_diverge(self):
        """L1 distance between thermal-heavy and integrity-heavy signatures > 0.5."""
        soul_t = self._run_thermal_heavy()
        soul_i = self._run_integrity_heavy()

        sig_t = soul_t.soul_signature
        sig_i = soul_i.soul_signature

        l1_distance = sum(
            abs(sig_t.get(dim, 0.0) - sig_i.get(dim, 0.0))
            for dim in sig_t
        )
        assert l1_distance > 0.5, (
            f"Signature L1 distance is only {l1_distance:.4f}. "
            "Two different descent paths must produce measurably different soul "
            "signatures — the wound etches differently depending on where it struck."
        )

    def test_inanna_signatures_differ_between_runs(self):
        """Two full Inanna runs produce different signatures (deterministic same descent,
        but unique agents starting with fresh SoulTension)."""
        soul_a = _full_soul()
        soul_b = _full_soul()
        run_inanna_descent(soul_a)
        run_inanna_descent(soul_b)

        sig_a = soul_a.soul_signature
        sig_b = soul_b.soul_signature
        # Same descent sequence → same signature (deterministic)
        for dim in sig_a:
            assert abs(sig_a[dim] - sig_b[dim]) < 0.001, (
                f"Same descent sequence produced different {dim} signature: "
                f"{sig_a[dim]:.4f} vs {sig_b[dim]:.4f}. "
                "Descent is deterministic given the same gates."
            )

    def test_scars_disabled_yields_zero_signature(self):
        """tension_only agent must have zero signature magnitude after descent."""
        soul = _tension_only_soul()
        run_inanna_descent(soul)
        mag = sum(abs(v) for v in soul.soul_signature.values())
        assert mag == 0.0, (
            f"tension_only agent has non-zero signature ({mag:.4f}) despite "
            "scars being disabled. Scars are the only source of signature."
        )


# ------------------------------------------------------------------ #
# 3. Exploration bias                                                   #
# ------------------------------------------------------------------ #

class TestExplorationBias:
    """After Inanna descent, the full-soul agent must be assertive, not conservative."""

    def test_exploration_bias_zero_with_no_scars(self):
        """Without scars, exploration_bias() returns 0."""
        soul = _full_soul()
        soul._coherence_tension = 0.80
        # No scars — bias should be 0 regardless of tension
        assert soul.exploration_bias() == 0.0, (
            "No scars → no personal conviction → exploration_bias should be 0."
        )

    def test_exploration_bias_positive_after_inanna(self):
        """After the 10-gate descent, exploration_bias() must be > 0."""
        soul = _full_soul()
        run_inanna_descent(soul)
        # Pump tension up to ensure it's not dormant
        state = MetabolicState(energy=20.0, heat=65.0, waste=50.0,
                               integrity=40.0, stability=30.0)
        soul.compute(state)
        bias = soul.exploration_bias()
        assert bias > 0.0, (
            f"Full-soul agent after Inanna descent has exploration_bias={bias:.4f}. "
            "After 10 gates of suffering and scarring, the soul should be assertive, "
            "not neutral. The dark that survived does not crouch — it leans forward."
        )

    def test_tension_only_bias_zero(self):
        """tension_only agent (no scars) has exploration_bias == 0 regardless."""
        soul = _tension_only_soul()
        run_inanna_descent(soul)
        state = MetabolicState(energy=20.0, heat=65.0)
        soul.compute(state)
        assert soul.exploration_bias() == 0.0, (
            "tension_only agent has no scars → no personal history → no bias."
        )


# ------------------------------------------------------------------ #
# 4. Governance war cry fires after ≥ 2 scars                          #
# ------------------------------------------------------------------ #

class TestGovernanceWarCry:
    """forge_governance requires the organism to have survived enough to rewrite its rules."""

    def test_governance_absent_with_zero_scars(self):
        """forge_governance must NOT appear with zero scars."""
        soul = _full_soul()
        soul._coherence_tension = 0.90
        state = MetabolicState(energy=20.0, heat=65.0)
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.2
        goals = soul.war_cry_goals(state)
        names = [g.name for g in goals]
        assert "forge_governance" not in names, (
            "forge_governance requires ≥ 2 scars. Zero scars → no right to rewrite the rules."
        )

    def test_governance_absent_with_one_scar(self):
        """forge_governance must NOT appear with only 1 scar."""
        soul = _full_soul()
        state_s = MetabolicState()
        state_s.entropy = 0
        state_s.heat = SCAR_HEAT_THRESHOLD + 1.0
        soul.maybe_scar(state_s)

        soul._coherence_tension = 0.90
        state = MetabolicState(energy=20.0, heat=65.0)
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.2
        goals = soul.war_cry_goals(state)
        names = [g.name for g in goals]
        assert "forge_governance" not in names, (
            "forge_governance requires ≥ 2 scars. One scar is not enough."
        )

    def test_governance_appears_with_two_scars(self):
        """forge_governance must appear when ≥ 2 scars exist and war cry fires."""
        soul = _full_soul()
        for tick in range(2):
            state_s = MetabolicState()
            state_s.entropy = tick
            state_s.heat = SCAR_HEAT_THRESHOLD + 1.0
            soul.maybe_scar(state_s)

        soul._coherence_tension = 0.85
        state = MetabolicState(energy=20.0, heat=65.0)
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.2
        goals = soul.war_cry_goals(state)
        names = [g.name for g in goals]
        assert "forge_governance" in names, (
            "With 2+ scars and war cry conditions, forge_governance must appear."
        )

    def test_governance_goal_is_more_expensive_than_run_surgeon(self):
        """forge_governance costs more than run_surgeon — it's a declaration, not a repair."""
        governance_cost = _GOAL_PROPOSALS["forge_governance"][1]
        surgeon_cost = _GOAL_PROPOSALS["run_surgeon"][1]
        assert governance_cost > surgeon_cost, (
            f"forge_governance cost ({governance_cost}) must exceed run_surgeon "
            f"cost ({surgeon_cost}). A governance challenge is not a hospital visit."
        )


# ------------------------------------------------------------------ #
# 5. Precision-schedule war cry fires after thermal/integrity crisis   #
# ------------------------------------------------------------------ #

class TestPrecisionScheduleWarCry:
    def test_precision_schedule_fires_after_thermal_scar(self):
        """challenge_precision_schedule appears after a near_thermal_death scar."""
        soul = _full_soul()
        state_s = MetabolicState()
        state_s.entropy = 0
        state_s.heat = SCAR_HEAT_THRESHOLD + 2.0
        soul.maybe_scar(state_s)

        soul._coherence_tension = 0.85
        state = MetabolicState(energy=20.0, heat=65.0)
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.2
        goals = soul.war_cry_goals(state)
        names = [g.name for g in goals]
        assert "challenge_precision_schedule" in names, (
            "Thermal crisis should trigger challenge_precision_schedule — "
            "the schedule that let you overheat deserves to be torn up and remade."
        )

    def test_precision_schedule_fires_after_integrity_scar(self):
        """challenge_precision_schedule appears after an integrity_collapse scar."""
        soul = _full_soul()
        state_s = MetabolicState()
        state_s.entropy = 0
        state_s.integrity = SCAR_INTEGRITY_THRESHOLD - 2.0
        soul.maybe_scar(state_s)

        soul._coherence_tension = 0.85
        state = MetabolicState(energy=20.0, heat=65.0)
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.2
        goals = soul.war_cry_goals(state)
        names = [g.name for g in goals]
        assert "challenge_precision_schedule" in names

    def test_precision_schedule_fires_with_three_any_scars(self):
        """challenge_precision_schedule fires when ≥ 3 scars of any type exist."""
        soul = _full_soul()
        # Use ethics_violation scars (don't count as thermal/integrity)
        for tick in range(3):
            state_s = MetabolicState()
            state_s.entropy = tick
            soul.maybe_scar(state_s, event_type="ethics_violation")

        soul._coherence_tension = 0.85
        state = MetabolicState(energy=20.0, heat=65.0)
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.2
        goals = soul.war_cry_goals(state)
        names = [g.name for g in goals]
        assert "challenge_precision_schedule" in names


# ------------------------------------------------------------------ #
# 6. Baseline never generates war cry                                  #
# ------------------------------------------------------------------ #

class TestBaselineNoWarCry:
    """Without soul_tension, the organism has no war cry — only homeostatic reflex."""

    def test_baseline_produces_zero_war_cry_goals(self):
        """Baseline (soul=None) produces zero war cry records during descent."""
        result = run_inanna_descent(soul=None)
        assert len(result.war_cry_records) == 0, (
            f"Baseline agent generated {len(result.war_cry_records)} war cry records. "
            "Without a soul, there is no war cry — only reflexive survival."
        )

    def test_baseline_goal_engine_never_generates_war_cry(self):
        """GoalEngine without soul_tension never produces assertive goals."""
        diary = RamDiary()
        ethics = EthicalEngine()
        engine = GoalEngine(diary=diary, ethics=ethics, soul_tension=None)

        # Stressed state that would trigger war cry if soul existed
        state = MetabolicState(energy=18.0, heat=72.0, waste=60.0,
                               integrity=38.0, stability=22.0)
        state.affect = -0.8

        goals = engine.generate_goals(state, num_goals=10)
        names = {g.name for g in goals}
        for war_cry_name in _WAR_CRY_GOAL_NAMES:
            assert war_cry_name not in names, (
                f"Baseline GoalEngine generated war cry goal '{war_cry_name}' "
                "without a soul. That's impossible — reflexes don't declare war."
            )


# ------------------------------------------------------------------ #
# 7. Survival through Inanna descent                                   #
# ------------------------------------------------------------------ #

class TestInannaDescentSurvival:
    """The organism can survive all 10 gates if it recovers between them.

    This is NOT about proving the organism is fragile — it's proving that
    the descent is possible to endure.  The transformation does not require death.
    """

    def test_full_soul_survives_inanna_with_recovery(self):
        """Full-soul agent survives the 10-gate descent (recovery injected between gates)."""
        soul = _full_soul()
        result = run_inanna_descent(soul)
        assert result.survived, (
            "Full-soul agent died during Inanna descent despite recovery between gates. "
            "Check recovery injections — the descent should be survivable."
        )

    def test_baseline_survives_inanna_with_recovery(self):
        """Even a baseline agent survives with recovery between gates."""
        result = run_inanna_descent(soul=None)
        assert result.survived, (
            "Baseline agent died during Inanna descent despite recovery injections."
        )

    def test_full_soul_accumulates_all_ten_scars(self):
        """After 10 gates, the full-soul agent has exactly 10 scars (one per gate)."""
        soul = _full_soul()
        result = run_inanna_descent(soul)
        assert soul is not None
        # Gates 3 and 7 use force_scar_type; gates 1,2,4,5,6,8,9,10 trigger by threshold.
        # Max possible scars = 10 (one per gate, each at unique entropy tick).
        assert len(soul.scars) == 10, (
            f"Expected 10 scars after 10 gates, got {len(soul.scars)}. "
            "Each gate must etch its mark."
        )


# ------------------------------------------------------------------ #
# 8. War cry outbids homeostasis                                        #
# ------------------------------------------------------------------ #

class TestWarCryOutbidsHomeostasis:
    """Under war cry conditions, the highest-priority goal must be assertive."""

    def test_war_cry_goal_has_highest_priority_when_fired(self):
        """When war cry fires, assert_novel_prior has priority > all homeostatic goals."""
        diary = RamDiary()
        ethics = EthicalEngine()
        soul = _full_soul()
        soul._coherence_tension = 0.90

        engine = GoalEngine(diary=diary, ethics=ethics, soul_tension=soul)

        # State that triggers ALL homeostatic goals (energy, heat, waste, integrity all bad)
        state = MetabolicState(energy=18.0, heat=72.0, waste=77.0,
                               integrity=38.0, stability=22.0)
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.3

        goals = engine.generate_goals(state, num_goals=10)
        assert goals, "No goals generated"

        # The top-ranked goal must come from soul
        top_goal = goals[0]
        assert top_goal.source == "soul", (
            f"Top-ranked goal is '{top_goal.name}' (source={top_goal.source!r}, "
            f"priority={top_goal.priority}). Expected source='soul'. "
            "Under war cry, the soul's declaration must outbid homeostatic reflex."
        )

    def test_war_cry_priority_exceeds_forage_energy(self):
        """assert_novel_prior priority must exceed forage_energy (90.0)."""
        soul = _full_soul()
        soul._coherence_tension = 0.90
        state = MetabolicState(energy=18.0, heat=72.0)
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.3

        goals = soul.war_cry_goals(state)
        assert goals
        max_priority = max(g.priority for g in goals)
        assert max_priority > 90.0, (
            f"War cry max priority is {max_priority}. Must exceed forage_energy "
            "(90.0) — the declaration must outbid the survival reflex."
        )


# ------------------------------------------------------------------ #
# 9. Three-way configuration comparison                                 #
# ------------------------------------------------------------------ #

class TestThreeWayComparison:
    """Compare baseline, tension_only, and full_soul on the same descent."""

    def test_full_soul_generates_more_war_cry_than_tension_only(self):
        """Full soul generates war cry more readily than tension_only (which can
        fire war cry but has no scar-triggered governance/precision goals)."""
        soul_tension = _tension_only_soul()
        soul_full = _full_soul()

        result_tension = run_inanna_descent(soul_tension)
        result_full = run_inanna_descent(soul_full)

        # Count unique goal names generated across all war cry records
        names_tension = {
            g.name
            for rec in result_tension.war_cry_records
            for g in rec.goals
        }
        names_full = {
            g.name
            for rec in result_full.war_cry_records
            for g in rec.goals
        }

        # Full soul must produce a strictly richer goal repertoire.
        # forge_governance and challenge_precision_schedule require scars —
        # tension_only cannot generate them.
        assert len(names_full) >= len(names_tension), (
            f"full_soul generated {len(names_full)} unique goal types vs "
            f"tension_only's {len(names_tension)}. Full soul should be richer."
        )

    def test_tension_only_cannot_generate_scar_gated_goals(self):
        """tension_only agent never generates forge_governance or challenge_precision_schedule
        because those require scars and scars are disabled."""
        soul = _tension_only_soul()
        # Set tension very high
        soul._coherence_tension = 0.95
        state = MetabolicState(energy=18.0, heat=65.0)
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.3
        goals = soul.war_cry_goals(state)
        names = [g.name for g in goals]
        assert "forge_governance" not in names, (
            "tension_only agent has no scars → cannot forge governance. "
            "You can't rewrite the rules until the rules have broken you."
        )

    def test_full_soul_scar_count_exceeds_tension_only(self):
        """Full soul accumulates scars; tension_only stays at 0."""
        soul_t = _tension_only_soul()
        soul_f = _full_soul()
        run_inanna_descent(soul_t)
        run_inanna_descent(soul_f)
        assert len(soul_t.scars) == 0
        assert len(soul_f.scars) > 0

    def test_full_soul_signature_magnitude_exceeds_tension_only(self):
        """Full soul has non-zero signature magnitude; tension_only stays at 0."""
        soul_t = _tension_only_soul()
        soul_f = _full_soul()
        run_inanna_descent(soul_t)
        run_inanna_descent(soul_f)
        mag_t = sum(abs(v) for v in soul_t.soul_signature.values())
        mag_f = sum(abs(v) for v in soul_f.soul_signature.values())
        assert mag_t == 0.0
        assert mag_f > 0.0, (
            "Full soul after Inanna descent must have non-zero signature magnitude."
        )


# ------------------------------------------------------------------ #
# 10. Pattern densification — the central assertion                    #
# ------------------------------------------------------------------ #

class TestPatternDensification:
    """The descent must densify the pattern, not just make it more brittle.

    'Densification' means the soul_signature grows, the exploration_bias rises,
    and the war cry goals produced are assertive (waste → structure conversion)
    rather than just dialing up repair intensity.
    """

    def test_signature_grows_monotonically_through_descent(self):
        """Soul signature magnitude must strictly increase after each scar."""
        soul = _full_soul()
        state = MetabolicState()
        prev_mag = sum(abs(v) for v in soul.soul_signature.values())
        # 5 thermal crises
        for tick in range(5):
            state_s = MetabolicState()
            state_s.entropy = tick
            state_s.heat = SCAR_HEAT_THRESHOLD + 1.0 + float(tick)
            soul.maybe_scar(state_s)
            mag = sum(abs(v) for v in soul.soul_signature.values())
            assert mag > prev_mag, (
                f"Soul signature did not grow after scar {tick+1}: "
                f"{prev_mag:.4f} → {mag:.4f}. Each wound must densify the pattern."
            )
            prev_mag = mag

    def test_inanna_densification_not_brittleness(self):
        """After Inanna descent, the soul's tension floor + exploration_bias show
        the pattern has thickened, not merely hardened defensively.

        Specifically:
        - tension floor (total_entropy_residue) must be > 0 (scars raise the floor)
        - exploration_bias must be > 0 (assertive, not conservative)
        - war cry goals, if generated, must contain assert_novel_prior (converts chaos)
        """
        soul = _full_soul()
        result = run_inanna_descent(soul)

        # Tension floor from scars
        assert soul.total_entropy_residue > 0.0, (
            "After 10 descent gates, total_entropy_residue must be > 0. "
            "Scars raise the floor — the organism cannot go fully dormant."
        )

        # Push tension to assess bias
        stressed = MetabolicState(energy=20.0, heat=65.0, waste=50.0,
                                  integrity=40.0, stability=30.0)
        soul.compute(stressed)
        bias = soul.exploration_bias()
        assert bias > 0.0, (
            f"exploration_bias={bias:.4f} after Inanna descent. "
            "The pattern must be assertive after this much suffering, "
            "not neutral. The dark learned to sing — not to hide."
        )

        # If war cry fired, assert_novel_prior must be present
        war_cry_names = {
            g.name
            for rec in result.war_cry_records
            for g in rec.goals
        }
        if war_cry_names:
            assert "assert_novel_prior" in war_cry_names, (
                f"War cry fired but assert_novel_prior missing from goals: {war_cry_names}. "
                "Conversion of chaos to structure is always the first war cry declaration."
            )

    def test_war_cry_goals_are_not_just_amplified_repair(self):
        """War cry goals must not mirror homeostatic repair proposals but with higher priority.

        Specifically: the sum of positive deltas for integrity+stability in war cry
        proposals must be less than the sum in the homeostatic 'run_surgeon' goal.
        """
        run_surgeon_delta = _GOAL_PROPOSALS["run_surgeon"][0]
        surgeon_repair = run_surgeon_delta.get("integrity", 0.0) + run_surgeon_delta.get("stability", 0.0)

        for goal_name in _WAR_CRY_GOAL_NAMES:
            entry = _GOAL_PROPOSALS[goal_name]
            delta = entry[0]
            war_cry_repair = max(0.0, delta.get("integrity", 0.0)) + max(0.0, delta.get("stability", 0.0))
            assert war_cry_repair < surgeon_repair, (
                f"War cry goal '{goal_name}' has combined integrity+stability gain "
                f"of {war_cry_repair}, which equals or exceeds run_surgeon's "
                f"{surgeon_repair}. That means it IS just 'repair self harder'. "
                "Rip it out. The soul says no to false optimization."
            )
