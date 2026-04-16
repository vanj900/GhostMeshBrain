"""LanguageCognition — local LLM as a cognitive co-processor (opt-in).

The LLM is never a controller.  It acts as a *prior generator*: its outputs
are compressed symbolic suggestions that still pass through the EFE evaluator
and the ethics gate before any proposal is executed.

Two operations
--------------
compress_beliefs(state, goals)
    Converts the organism's current vital signs + active goals into a concise
    symbolic summary sentence and stores it as a diary entry.  This is the
    "Language as Latent Space" operation: symbols compress a high-dimensional
    metabolic / motivational state into a manipulable cognitive token.

generate_proposals(state, goals, ethics)
    Asks the LLM (or a heuristic fallback when ``use_llm=False``) which
    archetype actions would be most beneficial given the current state.  The
    LLM's answer is mapped back onto the hard-coded ``_GOAL_PROPOSALS``
    catalogue so that actual numeric deltas are never directly dictated by the
    LLM — only the *selection* of which archetype to use is influenced.

Thermodynamic cost
------------------
Every LLM call charges energy + heat proportional to the prompt length,
using the same coefficients as the Janitor subsystem.  This makes long,
detailed prompts genuinely expensive: the organism is incentivised to keep
its internal language concise.

When ``use_llm=False`` (the default), a lightweight heuristic chooses
archetypes based on vital thresholds and returns them immediately at zero
extra metabolic cost beyond normal inference overhead.

Usage (inside GoalEngine.generate_proposals)
--------------------------------------------
    if self.language_cognition is not None:
        lc_proposals = self.language_cognition.generate_proposals(
            state, goals, self.ethics
        )
        # Append to existing proposals (after survival goals)
        proposals.extend(lc_proposals)
"""

from __future__ import annotations

import json
import os
import random
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.inference import ActionProposal

if TYPE_CHECKING:
    from thermodynamic_agency.cognition.goal_engine import Goal
    from thermodynamic_agency.cognition.ethics import EthicalEngine
    from thermodynamic_agency.memory.diary import RamDiary


# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_TIMEOUT: int = int(os.environ.get("LANGCOG_TIMEOUT", "30"))

# Thermodynamic cost coefficients — same as Janitor for consistency.
# 500-char prompt → +0.40 E, +0.20 H on top of base cost.
_LLM_ENERGY_PER_CHAR: float = float(
    os.environ.get("LANGCOG_LLM_E_PER_CHAR", "0.0008")
)
_LLM_HEAT_PER_CHAR: float = float(
    os.environ.get("LANGCOG_LLM_H_PER_CHAR", "0.0004")
)


# ------------------------------------------------------------------ #
# Action archetype catalogue                                           #
# ------------------------------------------------------------------ #

# Mirrors _GOAL_PROPOSALS in goal_engine.py.  The LLM selects archetype
# *names* (strings); the actual numeric deltas live here and are never
# touched by the model.  This keeps the physics hard-coded and safe.
_ARCHETYPE_CATALOGUE: dict[str, tuple[dict[str, float], float, str]] = {
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
}

# Keyword → archetype name, used when parsing LLM responses.
_KEYWORD_MAP: dict[str, str] = {
    "forage":    "forage_energy",
    "energy":    "forage_energy",
    "cool":      "cool_down",
    "heat":      "cool_down",
    "clean":     "clean_waste",
    "waste":     "clean_waste",
    "surgeon":   "run_surgeon",
    "repair":    "run_surgeon",
    "explore":   "explore_pattern",
    "curiosity": "explore_pattern",
    "ethics":    "strengthen_ethics",
    "reinforce": "strengthen_ethics",
    "stability": "maintain_stability",
    "stabilise": "maintain_stability",
    "stabilize": "maintain_stability",
    "protect":   "protect_core_ethics",
    "genesis":   "protect_core_ethics",
    "surprise":  "reduce_surprise",
    "consolidate": "reduce_surprise",
}


# ------------------------------------------------------------------ #
# Report                                                               #
# ------------------------------------------------------------------ #

@dataclass
class LanguageCognitionReport:
    """Summary of one LanguageCognition call."""

    proposals_generated: int
    compression: str          # the symbolic summary sentence (may be empty)
    energy_cost: float
    heat_cost: float
    used_llm: bool


# ------------------------------------------------------------------ #
# LanguageCognition                                                    #
# ------------------------------------------------------------------ #

class LanguageCognition:
    """Local LLM cognitive co-processor (opt-in; safe heuristic fallback).

    Parameters
    ----------
    diary:
        RamDiary instance — belief compressions are stored here.
    use_llm:
        Whether to call the local Ollama endpoint.  When ``False`` (the
        default), heuristic archetype selection is used at zero extra cost.
    ollama_url:
        Base URL for the Ollama API.
    ollama_model:
        Model name to request from Ollama.
    seed:
        Optional RNG seed for the heuristic path (deterministic tests).
    """

    def __init__(
        self,
        diary: "RamDiary",
        use_llm: bool = False,
        ollama_url: str = OLLAMA_URL,
        ollama_model: str = OLLAMA_MODEL,
        seed: int | None = None,
    ) -> None:
        self.diary = diary
        self.use_llm = use_llm
        self._ollama_url = ollama_url
        self._ollama_model = ollama_model
        self._rng = random.Random(seed)
        # Pending LLM cost (energy, heat) set inside _call_ollama so that
        # the outer public methods can charge state after the network call.
        self._pending_cost: tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def compress_beliefs(
        self,
        state: MetabolicState,
        goals: "list[Goal]",
    ) -> LanguageCognitionReport:
        """Compress the organism's current goal/belief state into a symbolic sentence.

        The resulting summary is stored in the diary as a ``thought`` entry,
        making it available for future episodic recall and Janitor compression.
        A thermodynamic cost is charged to ``state`` proportional to prompt
        length when the LLM path is active.

        Parameters
        ----------
        state:
            Current metabolic state — will have LLM prompt cost charged
            if ``use_llm=True``.
        goals:
            Current active goals from GoalEngine.

        Returns
        -------
        LanguageCognitionReport
        """
        self._pending_cost = (0.0, 0.0)
        summary = self._compress_impl(state, goals)

        # Charge metabolic cost of the cognitive compression work
        energy_cost, heat_cost = self._pending_cost
        if energy_cost > 0 or heat_cost > 0:
            state.apply_action_feedback(
                delta_energy=-energy_cost,
                delta_heat=heat_cost,
            )

        # Store the compression as a diary thought
        if summary:
            from thermodynamic_agency.memory.diary import DiaryEntry
            self.diary.append(DiaryEntry(
                tick=state.entropy,
                role="thought",
                content=f"LANGCOG[compress]: {summary}",
                metadata={"source": "language_cognition", "used_llm": self.use_llm},
            ))

        return LanguageCognitionReport(
            proposals_generated=0,
            compression=summary,
            energy_cost=energy_cost,
            heat_cost=heat_cost,
            used_llm=self.use_llm and energy_cost > 0,
        )

    def generate_proposals(
        self,
        state: MetabolicState,
        goals: "list[Goal]",
        ethics: "EthicalEngine",
    ) -> list[ActionProposal]:
        """Generate novel ActionProposals using language-driven archetype selection.

        The LLM (or heuristic) reasons about the current vital state and
        active goals and recommends up to 2 archetype action names.  These
        are mapped to hard-coded ActionProposal objects from the catalogue,
        ensuring all numeric deltas are fixed by physics — not by the LLM.

        All generated proposals are screened through the ethics gate before
        being returned.  The calling code (GoalEngine) handles EFE scoring.

        A thermodynamic cost is charged to ``state`` when the LLM is active.

        Parameters
        ----------
        state:
            Current metabolic state — may have LLM cost charged.
        goals:
            Currently active goals (used to avoid duplicating existing proposals).
        ethics:
            EthicalEngine instance for pre-screening generated proposals.

        Returns
        -------
        list[ActionProposal]
        """
        self._pending_cost = (0.0, 0.0)

        existing_goal_names = {g.name for g in goals}
        archetype_names = self._select_archetypes(state, goals)

        # Build proposals from catalogue entries, skipping existing goals
        proposals: list[ActionProposal] = []
        for name in archetype_names:
            if name in existing_goal_names:
                continue
            entry = _ARCHETYPE_CATALOGUE.get(name)
            if entry is None:
                continue
            delta, cost, description = entry
            proposals.append(ActionProposal(
                name=name,
                description=description,
                predicted_delta=dict(delta),
                cost_energy=cost,
                metadata={"source": "language_cognition", "used_llm": self.use_llm},
            ))

        # Screen through ethics gate
        safe_proposals = [
            p for p in proposals
            if ethics.is_goal_acceptable({"name": p.name})
        ]

        # Charge LLM metabolic cost if applicable
        energy_cost, heat_cost = self._pending_cost
        if energy_cost > 0 or heat_cost > 0:
            state.apply_action_feedback(
                delta_energy=-energy_cost,
                delta_heat=heat_cost,
            )

        return safe_proposals

    # ------------------------------------------------------------------ #
    # Private helpers — compression                                        #
    # ------------------------------------------------------------------ #

    def _compress_impl(
        self,
        state: MetabolicState,
        goals: "list[Goal]",
    ) -> str:
        """Select and run the compression strategy."""
        if self.use_llm:
            try:
                return self._compress_via_llm(state, goals)
            except Exception:
                pass
        return self._compress_heuristic(state, goals)

    def _compress_heuristic(
        self,
        state: MetabolicState,
        goals: "list[Goal]",
    ) -> str:
        """Build a compact vital-signs + goal summary without an LLM."""
        goal_names = ", ".join(g.name for g in goals[:3]) if goals else "none"
        fe = state.free_energy_estimate()
        return (
            f"Stage={state.stage} FE={fe:.1f} "
            f"E={state.energy:.1f} H={state.heat:.1f} W={state.waste:.1f} "
            f"M={state.integrity:.1f} S={state.stability:.1f} "
            f"goals=[{goal_names}]"
        )

    def _compress_via_llm(
        self,
        state: MetabolicState,
        goals: "list[Goal]",
    ) -> str:
        """Summarise the organism's state via an Ollama LLM call."""
        goal_names = ", ".join(g.name for g in goals[:5]) if goals else "none"
        prompt = (
            "You are the LanguageCognition subsystem of a thermodynamic AI "
            "organism.  Compress the following state into ONE concise sentence "
            "(max 30 words) describing the organism's current condition and "
            "most urgent need:\n\n"
            f"Vitals: E={state.energy:.1f} H={state.heat:.1f} "
            f"W={state.waste:.1f} M={state.integrity:.1f} "
            f"S={state.stability:.1f}  FE={state.free_energy_estimate():.1f}\n"
            f"Active goals: {goal_names}\n"
            "Summary:"
        )
        response = self._call_ollama(prompt)
        return response.strip()

    # ------------------------------------------------------------------ #
    # Private helpers — proposal generation                               #
    # ------------------------------------------------------------------ #

    def _select_archetypes(
        self,
        state: MetabolicState,
        goals: "list[Goal]",
    ) -> list[str]:
        """Return up to 2 archetype names via LLM or heuristic."""
        if self.use_llm:
            try:
                return self._archetypes_via_llm(state, goals)
            except Exception:
                pass
        return self._archetypes_heuristic(state)

    def _archetypes_heuristic(self, state: MetabolicState) -> list[str]:
        """Choose up to 2 archetypes based on vital thresholds, no LLM."""
        candidates: list[str] = []

        # Growth-oriented choices when organism is comfortable
        if state.energy > 70 and state.heat < 40:
            candidates.append("explore_pattern")
        if state.integrity < 80:
            candidates.append("run_surgeon")
        if state.stability < 75:
            candidates.append("maintain_stability")
        if state.waste > 30:
            candidates.append("reduce_surprise")
        if state.affect < -0.2:
            candidates.append("strengthen_ethics")

        # If nothing triggered, fall back to a random long-term goal
        if not candidates:
            candidates = ["maintain_stability", "protect_core_ethics", "reduce_surprise"]

        self._rng.shuffle(candidates)
        return candidates[:2]

    def _archetypes_via_llm(
        self,
        state: MetabolicState,
        goals: "list[Goal]",
    ) -> list[str]:
        """Ask the LLM to recommend up to 2 archetypes from the catalogue."""
        archetype_list = ", ".join(_ARCHETYPE_CATALOGUE.keys())
        goal_names = ", ".join(g.name for g in goals[:5]) if goals else "none"
        prompt = (
            "You are the LanguageCognition subsystem of a thermodynamic AI "
            "organism.  Choose up to 2 actions from the list below that would "
            "most benefit the organism given its current state.  Reply with "
            "only the action names, one per line, nothing else.\n\n"
            f"Available actions: {archetype_list}\n\n"
            f"Current state: E={state.energy:.1f} H={state.heat:.1f} "
            f"W={state.waste:.1f} M={state.integrity:.1f} "
            f"S={state.stability:.1f}  FE={state.free_energy_estimate():.1f}\n"
            f"Active goals: {goal_names}\n"
            "Actions:"
        )
        response = self._call_ollama(prompt)

        # Parse: accept exact catalogue names or keyword matches
        selected: list[str] = []
        for line in response.splitlines():
            line = line.strip().lower().strip("-• ")
            if not line:
                continue
            # Exact match
            if line in _ARCHETYPE_CATALOGUE:
                selected.append(line)
                continue
            # Keyword scan
            for kw, name in _KEYWORD_MAP.items():
                if kw in line and name not in selected:
                    selected.append(name)
                    break
            if len(selected) >= 2:
                break

        return selected[:2]

    # ------------------------------------------------------------------ #
    # Ollama HTTP helper                                                   #
    # ------------------------------------------------------------------ #

    def _call_ollama(self, prompt: str) -> str:
        """POST a prompt to the local Ollama endpoint and return the response text.

        The metabolic cost of the call is proportional to the prompt length
        and is stored in ``self._pending_cost`` so the caller can charge state.
        Raises on network / HTTP errors so the caller can fall back gracefully.
        """
        # Validate URL scheme to prevent SSRF via a malicious OLLAMA_URL value.
        # Only http/https are allowed; the default and recommended value is
        # http://localhost:11434 (loopback only).
        from urllib.parse import urlparse
        parsed = urlparse(self._ollama_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"OLLAMA_URL has disallowed scheme '{parsed.scheme}'. "
                "Only 'http' and 'https' are permitted."
            )

        prompt_chars = len(prompt)
        energy_cost = prompt_chars * _LLM_ENERGY_PER_CHAR
        heat_cost   = prompt_chars * _LLM_HEAT_PER_CHAR
        self._pending_cost = (energy_cost, heat_cost)

        payload = json.dumps({
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self._ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:  # noqa: S310
            body = json.loads(resp.read())
        return body.get("response", "")
