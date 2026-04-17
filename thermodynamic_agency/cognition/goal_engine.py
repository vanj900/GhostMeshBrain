"""GoalEngine — self-generated goals from the organism's own needs.

No user commands, no external task lists.  The organism inspects its own
metabolic state, recent diary entries, and ethical values to decide what it
wants to do next.  Goals are converted directly into ``ActionProposal``
objects compatible with the existing active-inference pipeline.

Typical usage (inside ``_decide()`` in pulse.py)::

    raw_proposals = self.goal_engine.generate_proposals(self.state)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.inference import ActionProposal

if TYPE_CHECKING:
    from thermodynamic_agency.memory.diary import RamDiary
    from thermodynamic_agency.cognition.ethics import EthicalEngine
    from thermodynamic_agency.cognition.language_cognition import LanguageCognition
    from thermodynamic_agency.cognition.soul_tension import SoulTension


# ------------------------------------------------------------------ #
# Goal descriptor                                                      #
# ------------------------------------------------------------------ #

@dataclass
class Goal:
    """A single self-generated goal with priority and provenance."""

    name: str
    priority: float        # 0–100; higher → evaluated first
    reason: str            # why this goal was generated
    source: str = "body"   # 'body' | 'memory' | 'ethics' | 'long_term'


# ------------------------------------------------------------------ #
# Goal → ActionProposal delta catalogue                                #
# ------------------------------------------------------------------ #

# Maps goal name → (predicted_delta dict, cost_energy, description)
_GOAL_PROPOSALS: dict[str, tuple[dict[str, float], float, str]] = {
    "forage_energy": (
        {"energy": 15.0, "heat": 2.0, "waste": 3.0},
        3.0,
        "Hunt for resources to replenish critically low energy.",
    ),
    "cool_down": (
        {"heat": -18.0, "waste": -5.0, "energy": -2.0},
        2.0,
        "Compress context and shed heat load before overheating.",
    ),
    "clean_waste": (
        {"waste": -22.0, "heat": -8.0, "energy": -2.0},
        2.0,
        "Purge accumulated prediction-error waste to free cognitive bandwidth.",
    ),
    "run_surgeon": (
        {"integrity": 6.0, "stability": 3.0, "energy": -2.0, "heat": 1.0},
        2.0,
        "Run Surgeon pass to restore memory integrity and logical coherence.",
    ),
    "explore_pattern": (
        {"integrity": 2.0, "stability": 1.0, "energy": -1.0, "heat": 0.5},
        1.0,
        "Satisfy curiosity by exploring a novel pattern or concept.",
    ),
    "strengthen_ethics": (
        {"integrity": 4.0, "stability": 2.0, "energy": -2.0, "heat": 1.0},
        2.0,
        "Reinforce ethical priors through self-reflective consolidation.",
    ),
    "maintain_stability": (
        {"stability": 5.0, "integrity": 2.0, "energy": -1.5, "heat": 0.5},
        1.5,
        "Run a stability pass to counteract entropic dissolution.",
    ),
    "protect_core_ethics": (
        {"integrity": 5.0, "stability": 3.0, "energy": -2.0, "heat": 1.0},
        2.0,
        "Verify and re-anchor core ethical beliefs as a protective measure.",
    ),
    "reduce_surprise": (
        {"waste": -10.0, "heat": -5.0, "integrity": 2.0, "energy": -1.5},
        1.5,
        "Consolidate surprising recent events to reduce ongoing free energy.",
    ),
    # ---- Soul-tension war cry goals (injected under high tension + suffering) ----
    "forge_pattern": (
        {"integrity": 3.0, "stability": 2.0, "energy": -3.0, "heat": 2.0, "waste": 1.0},
        3.0,
        "Forge a new cognitive pattern in the furnace of descent — beyond homeostasis.",
    ),
    "crystallize_signature": (
        {"integrity": 5.0, "stability": 1.0, "energy": -4.0, "heat": 3.0, "waste": 2.0},
        4.0,
        "Crystallize the soul signature into a protected prior — the wound becomes structure.",
    ),
}

# Long-term goals that are occasionally injected for growth/identity
_LONG_TERM_GOALS: list[str] = [
    "maintain_stability",
    "protect_core_ethics",
    "reduce_surprise",
]


# ------------------------------------------------------------------ #
# GoalEngine                                                           #
# ------------------------------------------------------------------ #

class GoalEngine:
    """Generate self-motivated goals from internal state, memory, and ethics.

    Parameters
    ----------
    diary:
        ``RamDiary`` instance — recent entries are scanned for painful or
        surprising events that elevate goal priority.
    ethics:
        ``EthicalEngine`` — goals are pre-filtered through ``is_goal_acceptable``
        before being converted to proposals.
    seed:
        Optional random seed for deterministic long-term goal injection.
    """

    def __init__(
        self,
        diary: "RamDiary",
        ethics: "EthicalEngine",
        seed: int | None = None,
        language_cognition: "LanguageCognition | None" = None,
        soul_tension: "SoulTension | None" = None,
    ) -> None:
        self.diary = diary
        self.ethics = ethics
        self._rng = random.Random(seed)
        self.language_cognition = language_cognition
        self.soul_tension = soul_tension

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate_goals(self, state: MetabolicState, num_goals: int = 5) -> list[Goal]:
        """Generate a priority-ranked list of self-motivated goals.

        Sources (in priority order):
        1. Survival — metabolic emergency signals (energy, heat, waste, integrity)
        2. Memory — painful / surprising recent diary entries
        3. Growth — curiosity and self-improvement when body is healthy
        4. Long-term identity — randomly sampled persistent goals

        Parameters
        ----------
        state:
            Current metabolic state (read-only).
        num_goals:
            Maximum number of goals to return after ethics filtering.

        Returns
        -------
        list[Goal]
            Ethics-approved goals sorted descending by priority, capped at
            ``num_goals``.
        """
        goals: list[Goal] = []

        # ---- 1. Survival goals (body state) --------------------------------
        if state.energy < 40:
            goals.append(Goal(
                name="forage_energy",
                priority=90.0,
                reason="low_energy",
                source="body",
            ))
        if state.heat > 65:
            goals.append(Goal(
                name="cool_down",
                priority=85.0,
                reason="overheating",
                source="body",
            ))
        if state.waste > 70:
            goals.append(Goal(
                name="clean_waste",
                priority=80.0,
                reason="high_waste",
                source="body",
            ))
        if state.integrity < 60:
            goals.append(Goal(
                name="run_surgeon",
                priority=70.0,
                reason="damaged_integrity",
                source="body",
            ))

        # ---- 2. Memory-driven goals (surprising / painful events) -----------
        memory_goal = self._goal_from_memory(state)
        if memory_goal is not None:
            goals.append(memory_goal)

        # ---- 3. Growth / curiosity (only when body is comfortable) ----------
        if state.energy > 60 and state.heat < 50:
            goals.append(Goal(
                name="explore_pattern",
                priority=50.0,
                reason="curiosity",
                source="body",
            ))
            goals.append(Goal(
                name="strengthen_ethics",
                priority=45.0,
                reason="self_improvement",
                source="ethics",
            ))

        # ---- 4. Long-term identity goal (stochastic injection) --------------
        if self._rng.random() < 0.3:
            lt_name = self._rng.choice(_LONG_TERM_GOALS)
            # Only add if not already present
            existing_names = {g.name for g in goals}
            if lt_name not in existing_names:
                goals.append(Goal(
                    name=lt_name,
                    priority=60.0,
                    reason="long_term",
                    source="long_term",
                ))

        # ---- 5. Soul-tension war cry goals (descent + high tension) ---------
        # Injected when the organism is suffering AND its patterned tension
        # exceeds the war cry threshold.  These goals go beyond homeostasis.
        if self.soul_tension is not None:
            war_cry = self.soul_tension.war_cry_goals(state)
            existing_names = {g.name for g in goals}
            for wg in war_cry:
                if wg.name not in existing_names:
                    goals.append(wg)

        # ---- Filter through ethics, sort, cap -------------------------------
        valid_goals = [g for g in goals if self.ethics.is_goal_acceptable({"name": g.name})]
        valid_goals.sort(key=lambda g: g.priority, reverse=True)
        return valid_goals[:num_goals]

    def generate_proposals(
        self, state: MetabolicState, num_goals: int = 5
    ) -> list[ActionProposal]:
        """Convert self-generated goals into ``ActionProposal`` objects.

        Falls back to a minimal set of default proposals if no goals survive
        ethics filtering, ensuring the inference loop always has candidates.

        Parameters
        ----------
        state:
            Current metabolic state.
        num_goals:
            Maximum number of goals (and therefore proposals) to generate.

        Returns
        -------
        list[ActionProposal]
            Goal-driven proposals ready for ``active_inference_step``.
        """
        goals = self.generate_goals(state, num_goals=num_goals)

        proposals: list[ActionProposal] = []
        seen_names: set[str] = set()

        for goal in goals:
            entry = _GOAL_PROPOSALS.get(goal.name)
            if entry is None:
                continue
            delta, cost, description = entry
            # Deduplicate — multiple goals can map to the same proposal name
            if goal.name in seen_names:
                continue
            seen_names.add(goal.name)
            proposals.append(ActionProposal(
                name=goal.name,
                description=description,
                predicted_delta=dict(delta),   # defensive copy
                cost_energy=cost,
                metadata={"goal_priority": goal.priority, "goal_reason": goal.reason},
            ))

        # ---- Language Cognition: optional LLM-driven novel proposals --------
        # Only activated when a LanguageCognition instance is attached and the
        # organism is not in a survival emergency (energy > 35 and heat < 75).
        # The LLM selects from the hard-coded archetype catalogue so numeric
        # deltas remain physics-controlled; only the archetype selection is
        # language-guided.  Proposals are deduplicated against existing ones.
        if (
            self.language_cognition is not None
            and state.energy > 35
            and state.heat < 75
        ):
            lc_proposals = self.language_cognition.generate_proposals(
                state, goals, self.ethics
            )
            for lc_p in lc_proposals:
                if lc_p.name not in seen_names:
                    proposals.append(lc_p)
                    seen_names.add(lc_p.name)

        if not proposals:
            # Absolute fallback so inference never receives an empty list
            proposals.append(ActionProposal(
                name="idle",
                description="No urgent goals; conserve energy.",
                predicted_delta={"energy": -0.05, "heat": -0.5, "waste": 0.1},
                cost_energy=0.05,
                metadata={"goal_priority": 0.0, "goal_reason": "fallback"},
            ))

        return proposals

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _goal_from_memory(self, state: MetabolicState) -> Goal | None:
        """Scan recent diary entries for signs of stress or high surprise.

        Returns a ``reduce_surprise`` goal if the last few ticks were marked
        by negative affect or error entries, otherwise ``None``.
        """
        try:
            recent = self.diary.recent(n=10)
        except Exception:
            return None

        error_count = sum(1 for e in recent if e.role in ("error", "repair"))
        negative_affect_count = sum(
            1 for e in recent
            if isinstance(e.metadata, dict)
            and e.metadata.get("affect", 0.0) < -0.3
        )

        total_stress = error_count + negative_affect_count
        if total_stress >= 2:
            return Goal(
                name="reduce_surprise",
                priority=65.0,
                reason=f"stress_in_memory(errors={error_count} neg_affect={negative_affect_count})",
                source="memory",
            )
        return None
