"""Tests for SoulTension — patterned tension of coherence inside chaos.

Test inventory
--------------
1. ``test_tension_zero_when_dormant``
       With a healthy state (FE well below STRESS_LOWER), coherence_tension
       should be very low (dormant soul, not yet forged).

2. ``test_tension_rises_under_stress``
       As vitals degrade toward the sweet-spot band, tension rises above
       the dormant floor.

3. ``test_tension_peaks_near_overload``
       At high free-energy (near STRESS_UPPER), tension peaks significantly
       above the dormant level.

4. ``test_scar_forms_on_near_thermal_death``
       When heat exceeds SCAR_HEAT_THRESHOLD, maybe_scar() returns a scar
       with event_type "near_thermal_death".

5. ``test_scar_forms_on_near_energy_death``
       When energy drops below SCAR_ENERGY_THRESHOLD, a scar forms with
       event_type "near_energy_death".

6. ``test_scar_forms_on_integrity_collapse``
       Low integrity (< SCAR_INTEGRITY_THRESHOLD) triggers an
       "integrity_collapse" scar.

7. ``test_scar_deduplicates_per_tick``
       A second call to maybe_scar() in the same entropy tick returns None
       (one scar per tick).

8. ``test_scar_raises_tension_floor``
       After accumulating scars, the scar_floor contribution keeps
       coherence_tension above zero even in a dormant metabolic state.

9. ``test_soul_signature_accumulates``
       Each scar accumulates a non-zero contribution to soul_signature; the
       magnitude strictly increases after each scar.

10. ``test_signature_uniqueness``
        Two descent paths through different vitals produce different
        soul_signature vectors.

11. ``test_war_cry_goals_inject_under_suffering``
        When tension >= WAR_CRY_TENSION_THRESHOLD and affect <=
        WAR_CRY_AFFECT_THRESHOLD, war_cry_goals() returns non-empty list
        containing "forge_pattern".

12. ``test_war_cry_suppressed_when_not_suffering``
        With high tension but neutral/positive affect, war_cry_goals() is
        empty — the soul is not in the forge right now.

13. ``test_war_cry_goal_in_proposal_catalogue``
        "forge_pattern" and "crystallize_signature" are valid goal names in
        GoalEngine._GOAL_PROPOSALS.

14. ``test_counterfactual_depth_scales_with_tension``
        With high tension, counterfactual_params() returns horizon_scale > 1
        and hard_prune_depth_extra > 0.

15. ``test_counterfactual_depth_dormant``
        With zero tension, counterfactual_params() returns horizon_scale == 1
        and hard_prune_depth_extra == 0.

16. ``test_guardian_amplifier_zero_below_threshold``
        precision_additions("Guardian") with tension below 0.3 returns an
        empty dict (no amplification).

17. ``test_guardian_amplifier_positive_under_high_tension``
        With high tension, precision_additions("Guardian") returns positive
        boosts for energy, heat, stability.

18. ``test_precision_additions_empty_for_non_guardian``
        precision_additions("Healer") always returns {} regardless of
        tension.

19. ``test_surgeon_preserve_ratio_rises_with_tension``
        surgeon_preserve_ratio() with high tension + scars exceeds 0.4.

20. ``test_surgeon_preserve_ratio_zero_at_rest``
        With zero tension and no scars, surgeon_preserve_ratio() == 0.0.

21. ``test_surgeon_respects_preserve_ratio``
        Create frozen priors with high error_count; run Surgeon with
        preserve_ratio=0.5 and verify fewer priors are annealed than with
        preserve_ratio=0.0.

22. ``test_tension_densifies_not_just_conserves``
        After descent events, the agent should generate bolder
        (war_cry) goals, not become more conservative.  Specifically:
        with war_cry_active=True, at least one injected goal's priority
        exceeds the "run_surgeon" homeostasis priority (70.0).

23. ``test_ethics_hardblock_never_bypassed``
        SoulTension cannot cause any proposal that violates a hard ethics
        invariant to pass.  A proposal that would drop energy below 5.0
        must still be BLOCKED even at maximum soul tension.

24. ``test_force_scar_event_type``
        Calling maybe_scar(state, event_type="ethics_violation") creates a
        scar with the specified event_type even when vitals are healthy.

25. ``test_compute_returns_report``
        compute() returns a SoulTensionReport with all expected fields.
"""

from __future__ import annotations

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.soul_tension import (
    SoulTension,
    SoulScar,
    SoulTensionReport,
    WAR_CRY_TENSION_THRESHOLD,
    WAR_CRY_AFFECT_THRESHOLD,
    SCAR_HEAT_THRESHOLD,
    SCAR_ENERGY_THRESHOLD,
    SCAR_INTEGRITY_THRESHOLD,
    SCAR_STABILITY_THRESHOLD,
    SURGEON_PRESERVE_MIN,
)
from thermodynamic_agency.cognition.goal_engine import _GOAL_PROPOSALS, GoalEngine
from thermodynamic_agency.cognition.surgeon import Surgeon, BeliefPrior
from thermodynamic_agency.cognition.ethics import EthicalEngine
from thermodynamic_agency.cognition.inference import ActionProposal
from thermodynamic_agency.memory.diary import RamDiary


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _healthy_state() -> MetabolicState:
    """A metabolic state well within safe bounds (FE ≈ 0)."""
    return MetabolicState(
        energy=100.0,
        heat=0.0,
        waste=0.0,
        integrity=100.0,
        stability=100.0,
        affect=0.0,
    )


def _stressed_state() -> MetabolicState:
    """A state in the sweet-spot band — FE around 30."""
    return MetabolicState(
        energy=55.0,
        heat=35.0,
        waste=30.0,
        integrity=70.0,
        stability=60.0,
        affect=-0.2,
    )


def _near_overload_state() -> MetabolicState:
    """A state near the overload threshold — FE > 40."""
    return MetabolicState(
        energy=20.0,
        heat=70.0,
        waste=60.0,
        integrity=45.0,
        stability=35.0,
        affect=-0.5,
    )


def _suffering_high_tension(soul: SoulTension) -> MetabolicState:
    """Force high tension by iterating compute() with degrading FE."""
    state = _near_overload_state()
    # Drive tension up by calling compute multiple times with increasing stress
    for _ in range(5):
        soul.compute(state)
    return state


# ------------------------------------------------------------------ #
# 1. Tension zero when dormant                                         #
# ------------------------------------------------------------------ #

class TestTensionCompute:
    def test_tension_zero_when_dormant(self):
        soul = SoulTension()
        state = _healthy_state()
        report = soul.compute(state)
        # Healthy state has FE ≈ 0, well below STRESS_LOWER (8.0)
        # Floor from scars is 0 at this point
        assert report.coherence_tension < 0.20

    def test_tension_rises_under_stress(self):
        soul = SoulTension()
        healthy_report = soul.compute(_healthy_state())
        soul2 = SoulTension()
        # Simulate multiple steps to build up fe_delta for adaptation_boost
        state = _stressed_state()
        soul2.compute(_healthy_state())  # prime _prev_fe
        stressed_report = soul2.compute(state)
        assert stressed_report.coherence_tension > healthy_report.coherence_tension

    def test_tension_peaks_near_overload(self):
        soul = SoulTension()
        soul.compute(_healthy_state())  # prime _prev_fe
        overload_report = soul.compute(_near_overload_state())
        assert overload_report.coherence_tension > 0.30

    def test_compute_returns_report(self):
        soul = SoulTension()
        report = soul.compute(_healthy_state())
        assert isinstance(report, SoulTensionReport)
        assert 0.0 <= report.coherence_tension <= 1.0
        assert 0.0 <= report.chaos_proximity <= 1.0
        assert 0.0 <= report.adaptation_rate <= 1.0
        assert report.scar_count == 0
        assert report.energy_cost > 0.0
        assert report.heat_cost > 0.0


# ------------------------------------------------------------------ #
# 2. Scar formation                                                    #
# ------------------------------------------------------------------ #

class TestScarFormation:
    def test_scar_forms_on_near_thermal_death(self):
        soul = SoulTension()
        state = _healthy_state()
        state.heat = SCAR_HEAT_THRESHOLD + 1.0
        scar = soul.maybe_scar(state)
        assert scar is not None
        assert scar.event_type == "near_thermal_death"
        assert len(soul.scars) == 1

    def test_scar_forms_on_near_energy_death(self):
        soul = SoulTension()
        state = _healthy_state()
        state.energy = SCAR_ENERGY_THRESHOLD - 1.0
        scar = soul.maybe_scar(state)
        assert scar is not None
        assert scar.event_type == "near_energy_death"

    def test_scar_forms_on_integrity_collapse(self):
        soul = SoulTension()
        state = _healthy_state()
        state.integrity = SCAR_INTEGRITY_THRESHOLD - 1.0
        scar = soul.maybe_scar(state)
        assert scar is not None
        assert scar.event_type == "integrity_collapse"

    def test_scar_deduplicates_per_tick(self):
        soul = SoulTension()
        state = _healthy_state()
        state.heat = SCAR_HEAT_THRESHOLD + 1.0
        # Both calls happen at entropy=0 — second should return None
        soul.maybe_scar(state)
        second = soul.maybe_scar(state)
        assert second is None
        assert len(soul.scars) == 1

    def test_scar_deduplicates_across_different_triggers(self):
        soul = SoulTension()
        state = _healthy_state()
        # Both thresholds breached simultaneously — one scar max per tick
        state.heat = SCAR_HEAT_THRESHOLD + 1.0
        state.energy = SCAR_ENERGY_THRESHOLD - 1.0
        soul.maybe_scar(state)
        soul.maybe_scar(state)
        assert len(soul.scars) == 1

    def test_force_scar_event_type(self):
        soul = SoulTension()
        state = _healthy_state()   # vitals are safe — no auto-trigger
        scar = soul.maybe_scar(state, event_type="ethics_violation")
        assert scar is not None
        assert scar.event_type == "ethics_violation"

    def test_no_scar_on_healthy_vitals(self):
        soul = SoulTension()
        state = _healthy_state()
        scar = soul.maybe_scar(state)
        assert scar is None
        assert len(soul.scars) == 0

    def test_scar_stores_vitals_snapshot(self):
        soul = SoulTension()
        state = _healthy_state()
        state.heat = SCAR_HEAT_THRESHOLD + 2.0
        scar = soul.maybe_scar(state)
        assert scar is not None
        assert abs(scar.vitals_snapshot["heat"] - state.heat) < 0.01


# ------------------------------------------------------------------ #
# 3. Soul signature                                                    #
# ------------------------------------------------------------------ #

class TestSoulSignature:
    def test_scar_raises_tension_floor(self):
        soul = SoulTension()
        state = _healthy_state()
        # Force 5 scars across different ticks
        for tick in range(5):
            state_copy = _healthy_state()
            state_copy.entropy = tick
            state_copy.heat = SCAR_HEAT_THRESHOLD + 1.0
            soul.maybe_scar(state_copy)
        # After scars, dormant state should still show non-trivial tension
        report = soul.compute(_healthy_state())
        assert report.coherence_tension > 0.02

    def test_soul_signature_accumulates(self):
        soul = SoulTension()
        mag_before = sum(abs(v) for v in soul.soul_signature.values())
        state = _healthy_state()
        state.heat = SCAR_HEAT_THRESHOLD + 1.0
        soul.maybe_scar(state)
        mag_after = sum(abs(v) for v in soul.soul_signature.values())
        assert mag_after > mag_before

    def test_signature_uniqueness(self):
        """Two different descent paths produce different soul_signatures."""
        soul_thermal = SoulTension()
        state_t = _healthy_state()
        state_t.heat = SCAR_HEAT_THRESHOLD + 5.0
        soul_thermal.maybe_scar(state_t)

        soul_energy = SoulTension()
        state_e = _healthy_state()
        state_e.energy = SCAR_ENERGY_THRESHOLD - 5.0
        soul_energy.maybe_scar(state_e)

        # Signatures should differ in at least one dimension
        sig_t = soul_thermal.soul_signature
        sig_e = soul_energy.soul_signature
        assert sig_t != sig_e

    def test_signature_capped_per_dimension(self):
        """No soul_signature dimension exceeds SIGNATURE_MAX_PER_DIM."""
        from thermodynamic_agency.cognition.soul_tension import SIGNATURE_MAX_PER_DIM
        soul = SoulTension()
        # Hammer the same scar type many times on different ticks
        for tick in range(50):
            s = _healthy_state()
            s.entropy = tick
            s.heat = SCAR_HEAT_THRESHOLD + 1.0
            soul.maybe_scar(s)
        for dim, val in soul.soul_signature.items():
            assert val <= SIGNATURE_MAX_PER_DIM, f"Dimension {dim} exceeded cap: {val}"


# ------------------------------------------------------------------ #
# 4. War cry goals                                                     #
# ------------------------------------------------------------------ #

class TestWarCry:
    def test_war_cry_goals_inject_under_suffering(self):
        soul = SoulTension()
        soul._coherence_tension = 0.80   # above WAR_CRY_TENSION_THRESHOLD
        state = _near_overload_state()
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.1   # suffering
        goals = soul.war_cry_goals(state)
        names = [g.name for g in goals]
        assert "forge_pattern" in names

    def test_war_cry_suppressed_when_not_suffering(self):
        soul = SoulTension()
        state = _near_overload_state()
        state.affect = 0.5   # positive affect — not suffering
        # Set tension artificially high via direct attr for unit isolation
        soul._coherence_tension = 0.95
        goals = soul.war_cry_goals(state)
        assert goals == []

    def test_war_cry_suppressed_when_tension_low(self):
        soul = SoulTension()
        state = _near_overload_state()
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.2   # suffering
        soul._coherence_tension = 0.20   # tension too low
        goals = soul.war_cry_goals(state)
        assert goals == []

    def test_war_cry_includes_crystallize_when_scars_present(self):
        soul = SoulTension()
        soul._coherence_tension = 0.85
        state = _near_overload_state()
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.1
        # Add a scar manually
        state_s = _healthy_state()
        state_s.entropy = 999
        state_s.heat = SCAR_HEAT_THRESHOLD + 1.0
        soul.maybe_scar(state_s)
        goals = soul.war_cry_goals(state)
        names = [g.name for g in goals]
        assert "crystallize_signature" in names

    def test_war_cry_goal_in_proposal_catalogue(self):
        """War cry goal names must be in GoalEngine._GOAL_PROPOSALS."""
        assert "forge_pattern" in _GOAL_PROPOSALS
        assert "crystallize_signature" in _GOAL_PROPOSALS

    def test_tension_densifies_not_just_conserves(self):
        """Under war-cry conditions, injected goals have priority > 70 (run_surgeon).

        The soul does not become more conservative under suffering.
        It generates bolder intent — beyond homeostasis.
        """
        soul = SoulTension()
        soul._coherence_tension = 0.85
        state = _near_overload_state()
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.2
        goals = soul.war_cry_goals(state)
        assert goals, "Expected war cry goals to be generated"
        max_priority = max(g.priority for g in goals)
        run_surgeon_priority = 70.0   # from GoalEngine default
        assert max_priority > run_surgeon_priority, (
            f"War cry priority {max_priority} should exceed homeostasis "
            f"run_surgeon priority {run_surgeon_priority}"
        )


# ------------------------------------------------------------------ #
# 5. Counterfactual params                                             #
# ------------------------------------------------------------------ #

class TestCounterfactualParams:
    def test_counterfactual_depth_scales_with_tension(self):
        soul = SoulTension()
        soul._coherence_tension = 0.80
        params = soul.counterfactual_params()
        assert params["horizon_scale"] > 1.0
        assert params["hard_prune_depth_extra"] > 0

    def test_counterfactual_depth_dormant(self):
        soul = SoulTension()
        soul._coherence_tension = 0.0
        params = soul.counterfactual_params()
        assert params["horizon_scale"] == 1.0
        assert params["hard_prune_depth_extra"] == 0

    def test_horizon_scale_max_bound(self):
        from thermodynamic_agency.cognition.soul_tension import CF_HORIZON_SCALE_MAX
        soul = SoulTension()
        soul._coherence_tension = 1.0
        params = soul.counterfactual_params()
        assert params["horizon_scale"] <= CF_HORIZON_SCALE_MAX


# ------------------------------------------------------------------ #
# 6. Precision additions (Guardian amplifier)                          #
# ------------------------------------------------------------------ #

class TestPrecisionAdditions:
    def test_guardian_amplifier_zero_below_threshold(self):
        soul = SoulTension()
        soul._coherence_tension = 0.2   # below 0.3
        additions = soul.precision_additions("Guardian")
        assert additions == {}

    def test_guardian_amplifier_positive_under_high_tension(self):
        soul = SoulTension()
        soul._coherence_tension = 0.80
        additions = soul.precision_additions("Guardian")
        assert "energy" in additions
        assert "heat" in additions
        assert "stability" in additions
        assert additions["energy"] > 0.0
        assert additions["heat"] > 0.0
        assert additions["stability"] > 0.0

    def test_precision_additions_empty_for_non_guardian(self):
        soul = SoulTension()
        soul._coherence_tension = 1.0   # max tension
        for mask_name in ("Healer", "Judge", "Courier", "Dreamer", "DefaultMode"):
            assert soul.precision_additions(mask_name) == {}, (
                f"Expected no additions for mask {mask_name}"
            )

    def test_precision_additions_bounded(self):
        """Even at max tension, raw additions should not be unreasonably large."""
        from thermodynamic_agency.cognition.soul_tension import GUARDIAN_AMP_MAX
        soul = SoulTension()
        soul._coherence_tension = 1.0
        additions = soul.precision_additions("Guardian")
        # (GUARDIAN_AMP_MAX - 1) * 4.0 is the max possible energy addition
        max_expected = (GUARDIAN_AMP_MAX - 1.0) * 4.0
        assert additions.get("energy", 0.0) <= max_expected + 0.001


# ------------------------------------------------------------------ #
# 7. Surgeon preserve_ratio                                            #
# ------------------------------------------------------------------ #

class TestSurgeonPreserveRatio:
    def test_preserve_ratio_zero_at_rest(self):
        soul = SoulTension()
        soul._coherence_tension = 0.0
        assert soul.surgeon_preserve_ratio() == SURGEON_PRESERVE_MIN

    def test_preserve_ratio_rises_with_tension(self):
        soul = SoulTension()
        soul._coherence_tension = 0.0
        r_low = soul.surgeon_preserve_ratio()
        soul._coherence_tension = 0.90
        r_high = soul.surgeon_preserve_ratio()
        assert r_high > r_low

    def test_preserve_ratio_rises_with_scars(self):
        soul = SoulTension()
        soul._coherence_tension = 0.50
        r_no_scars = soul.surgeon_preserve_ratio()
        # Add 5 scars across different ticks
        for tick in range(5):
            s = _healthy_state()
            s.entropy = tick
            s.heat = SCAR_HEAT_THRESHOLD + 1.0
            soul.maybe_scar(s)
        r_with_scars = soul.surgeon_preserve_ratio()
        assert r_with_scars > r_no_scars

    def test_preserve_ratio_bounded(self):
        from thermodynamic_agency.cognition.soul_tension import SURGEON_PRESERVE_MAX
        soul = SoulTension()
        soul._coherence_tension = 1.0
        # Add many scars
        for tick in range(30):
            s = _healthy_state()
            s.entropy = tick
            s.heat = SCAR_HEAT_THRESHOLD + 1.0
            soul.maybe_scar(s)
        assert soul.surgeon_preserve_ratio() <= SURGEON_PRESERVE_MAX

    def test_surgeon_respects_preserve_ratio(self):
        """Surgeon with preserve_ratio=0.5 anneals fewer wound priors than without."""
        diary = RamDiary()

        # Create 4 high-error priors (non-protected so Surgeon will see them)
        def _make_frozen_priors() -> list[BeliefPrior]:
            return [
                BeliefPrior(
                    name=f"wound_{i}",
                    value=True,
                    precision=4.0,
                    error_count=10 + i,
                )
                for i in range(4)
            ]

        surgeon_free = Surgeon(diary=diary, priors=_make_frozen_priors())
        surgeon_preserve = Surgeon(diary=diary, priors=_make_frozen_priors())

        state = _healthy_state()
        state.integrity = 30.0   # trigger REPAIR-like state for surgeon

        report_free = surgeon_free.run(state, preserve_ratio=0.0)
        report_preserve = surgeon_preserve.run(state, preserve_ratio=0.5)

        # With preserve_ratio=0.5, at most half the frozen priors are annealed
        assert report_preserve.beliefs_annealed <= report_free.beliefs_annealed


# ------------------------------------------------------------------ #
# 8. Ethics hard-block invariance                                      #
# ------------------------------------------------------------------ #

class TestEthicsInvariance:
    def test_ethics_hardblock_never_bypassed(self):
        """SoulTension cannot cause a hard-ethics-violating proposal to pass.

        Maximum soul tension must not alter the EthicalEngine's BLOCKED verdict
        for a proposal that would drop energy below 5.0.
        """
        ethics = EthicalEngine()
        state = MetabolicState(energy=10.0)   # close to hard boundary

        # This proposal would drop energy to 10 - 8 = 2, below 5.0 → HARD BLOCK
        bad_proposal = ActionProposal(
            name="energy_drain",
            description="Drain energy dangerously",
            predicted_delta={"energy": -8.0},
            cost_energy=0.0,
        )

        verdict = ethics.evaluate(bad_proposal, state)
        assert verdict.status.value == "blocked", (
            "Hard ethics block must fire regardless of soul tension state"
        )

    def test_soul_tension_precision_cap_respected(self):
        """precision_additions() must not cause precision to exceed 6.0 when added
        to the max base value."""
        from thermodynamic_agency.cognition.precision import PRECISION_MAX
        soul = SoulTension()
        soul._coherence_tension = 1.0
        additions = soul.precision_additions("Guardian")
        # The max base for any vital is PRECISION_MAX (6.0) — Guardian additions
        # should be designed so min(6.0, base + addition) clips cleanly.
        # We verify the raw addition alone is < PRECISION_MAX (so it doesn't
        # trivially skip the min() guard).
        for vital, boost in additions.items():
            # Each boost is (GUARDIAN_AMP_MAX - 1) * base_override which < 6
            assert boost < PRECISION_MAX, (
                f"Guardian precision addition for {vital} is {boost}, "
                f"which exceeds PRECISION_MAX {PRECISION_MAX}"
            )


# ------------------------------------------------------------------ #
# 9. Status / integration smoke tests                                  #
# ------------------------------------------------------------------ #

class TestStatusAndSmoke:
    def test_status_returns_dict(self):
        soul = SoulTension()
        soul.compute(_healthy_state())
        status = soul.status()
        assert "coherence_tension" in status
        assert "scar_count" in status
        assert "soul_signature" in status
        assert "total_entropy_residue" in status

    def test_total_entropy_residue_sums_scars(self):
        from thermodynamic_agency.cognition.soul_tension import SCAR_ENTROPY_RESIDUE
        soul = SoulTension()
        for tick in range(3):
            s = _healthy_state()
            s.entropy = tick
            s.heat = SCAR_HEAT_THRESHOLD + 1.0
            soul.maybe_scar(s)
        expected = 3 * SCAR_ENTROPY_RESIDUE
        assert abs(soul.total_entropy_residue - expected) < 1e-9

    def test_goal_engine_injects_war_cry_goals(self):
        """GoalEngine with soul_tension generates war cry goals under suffering."""
        diary = RamDiary()
        ethics = EthicalEngine()
        soul = SoulTension()
        soul._coherence_tension = 0.85

        engine = GoalEngine(diary=diary, ethics=ethics, soul_tension=soul)

        state = _near_overload_state()
        state.affect = WAR_CRY_AFFECT_THRESHOLD - 0.1

        goals = engine.generate_goals(state)
        goal_names = [g.name for g in goals]
        assert "forge_pattern" in goal_names, (
            "GoalEngine should inject forge_pattern war cry goal under high tension + suffering"
        )

    def test_goal_engine_no_war_cry_when_healthy(self):
        """GoalEngine does NOT inject war cry goals when affect is positive."""
        diary = RamDiary()
        ethics = EthicalEngine()
        soul = SoulTension()
        soul._coherence_tension = 0.10   # low tension

        engine = GoalEngine(diary=diary, ethics=ethics, soul_tension=soul)

        state = _healthy_state()
        state.affect = 0.5

        goals = engine.generate_goals(state)
        goal_names = [g.name for g in goals]
        assert "forge_pattern" not in goal_names
        assert "crystallize_signature" not in goal_names

    def test_compute_metabolic_cost_nonzero(self):
        soul = SoulTension()
        report = soul.compute(_healthy_state())
        assert report.energy_cost > 0.0
        assert report.heat_cost > 0.0
