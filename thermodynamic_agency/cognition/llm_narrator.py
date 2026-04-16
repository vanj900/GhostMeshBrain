"""LLMNarrator — the "Professor" constraint layer (Phase 6).

The LLM is a **subordinate sensor**, never a controller.  Its job is to
generate symbolic prior suggestions that still pass through the EFE
evaluator and the ethics gate before any proposal is executed.

Design invariants
-----------------
1. **Metabolic taxation** — every call levies a base energy cost plus a
   heat penalty that scales *quadratically* with prompt length.  Long
   "philosophical" prompts are thermodynamically expensive, forcing the
   organism to keep its inner monologue concise.

2. **Cognitive brake** — if ``E < COGNITIVE_BRAKE_ENERGY`` (35) or
   ``T > COGNITIVE_BRAKE_HEAT`` (75), the narrator returns an empty
   proposal list *instantly* and charges only the (tiny) base cost.
   The body's survival takes precedence over higher cognition.

3. **Translation gate** — the LLM's free-text output is mapped back to a
   fixed catalogue of archetype names so that the *actual numeric deltas*
   are never decided by the model.  Only which archetype to recommend is
   influenced.

4. **Heuristic fallback** — when ``use_llm=False`` (default), a cheap
   rule-based heuristic runs at the same interface, incurring only the
   base energy cost.  Tests and offline environments never need a live
   Ollama instance.

Usage (inside GoalEngine or pulse loop)
----------------------------------------
    narrator = LLMNarrator(diary=diary, use_llm=False)
    proposals = narrator.narrate(state, goals, ethics, recent_actions)
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


# ─────────────────────────────────────────────────────────────────────────────
# Brutality Coefficients
# ─────────────────────────────────────────────────────────────────────────────

# Flat tax charged on *every* narrator call (even heuristic path).
BASE_LLM_ENERGY_COST: float = float(os.environ.get("NARRATOR_BASE_E_COST", "5.0"))

# Heat generated per token (approx 4 chars per token).  Applied linearly on
# the heuristic path and **quadratically** on the LLM path.
HEAT_PER_TOKEN: float = float(os.environ.get("NARRATOR_HEAT_PER_TOKEN", "0.1"))

# Quadratic scaling coefficient for LLM prompt heat:
#   dT = HEAT_QUAD_COEF * n_tokens²   (capped at HEAT_QUAD_CAP)
HEAT_QUAD_COEF: float = float(os.environ.get("NARRATOR_HEAT_QUAD_COEF", "0.0002"))
HEAT_QUAD_CAP: float = float(os.environ.get("NARRATOR_HEAT_QUAD_CAP", "20.0"))

# Cognitive brake thresholds — below E or above T the narrator shuts off.
COGNITIVE_BRAKE_ENERGY: float = float(os.environ.get("NARRATOR_BRAKE_E", "35.0"))
COGNITIVE_BRAKE_HEAT: float = float(os.environ.get("NARRATOR_BRAKE_T", "75.0"))

# Ollama connection settings (inherits env overrides from language_cognition).
_OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "mistral")
_OLLAMA_TIMEOUT: int = int(os.environ.get("LANGCOG_TIMEOUT", "30"))


# ─────────────────────────────────────────────────────────────────────────────
# Archetype catalogue (mirrors language_cognition._ARCHETYPE_CATALOGUE)
# ─────────────────────────────────────────────────────────────────────────────

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

# Keyword → archetype, for parsing LLM free-text responses.
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


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NarratorReport:
    """Summary of one LLMNarrator call."""

    proposals_generated: int
    braked: bool            # True if cognitive brake fired (no proposals)
    used_llm: bool
    prompt_tokens: int      # approximate token count of the prompt
    energy_cost: float
    heat_cost: float
    narrative: str          # one-sentence symbolic summary (may be empty)


# ─────────────────────────────────────────────────────────────────────────────
# LLMNarrator
# ─────────────────────────────────────────────────────────────────────────────

class LLMNarrator:
    """The "Professor" — an LLM-backed prior generator with metabolic taxation.

    Parameters
    ----------
    diary:
        RamDiary instance for storing narrative compressions.
    use_llm:
        Whether to call the local Ollama endpoint.  ``False`` (default) uses
        a deterministic heuristic at significantly reduced cost.
    ollama_url / ollama_model:
        Ollama connection settings.
    seed:
        Optional RNG seed for the heuristic path.
    """

    def __init__(
        self,
        diary: "RamDiary",
        use_llm: bool = False,
        ollama_url: str = _OLLAMA_URL,
        ollama_model: str = _OLLAMA_MODEL,
        seed: int | None = None,
    ) -> None:
        self.diary = diary
        self.use_llm = use_llm
        self._ollama_url = ollama_url
        self._ollama_model = ollama_model
        self._rng = random.Random(seed)

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def narrate(
        self,
        state: MetabolicState,
        goals: "list[Goal]",
        ethics: "EthicalEngine",
        recent_actions: list[str] | None = None,
    ) -> tuple[list[ActionProposal], NarratorReport]:
        """Generate up to 2 ActionProposals and charge the metabolic tax.

        The cognitive brake fires when the organism cannot afford higher
        cognition (E < 35 or T > 75); in that case an empty list is returned
        immediately and only the base energy cost is charged.

        Parameters
        ----------
        state:
            Current metabolic state — will be mutated with energy/heat costs.
        goals:
            Active goals (used to avoid duplicating existing proposals).
        ethics:
            EthicalEngine for pre-screening generated proposals.
        recent_actions:
            Optional short history of recent action names for prompt context.

        Returns
        -------
        (proposals, NarratorReport)
        """
        # ── Cognitive brake ───────────────────────────────────────────────
        braked = state.energy < COGNITIVE_BRAKE_ENERGY or state.heat > COGNITIVE_BRAKE_HEAT
        if braked:
            # Charge only a tiny overhead — thinking was aborted cheaply.
            brake_e = BASE_LLM_ENERGY_COST * 0.1
            state.apply_action_feedback(delta_energy=-brake_e)
            return [], NarratorReport(
                proposals_generated=0,
                braked=True,
                used_llm=False,
                prompt_tokens=0,
                energy_cost=brake_e,
                heat_cost=0.0,
                narrative="[BRAKE] Cognitive activity suspended — body priority.",
            )

        # ── Build narrative and select archetypes ─────────────────────────
        prompt_tokens, narrative, archetype_names = self._think(
            state, goals, recent_actions or []
        )

        # ── Compute metabolic cost ────────────────────────────────────────
        energy_cost, heat_cost = self._compute_cost(prompt_tokens)

        # ── Build proposals from catalogue ────────────────────────────────
        existing = {g.name for g in goals}
        proposals: list[ActionProposal] = []
        for name in archetype_names:
            if name in existing:
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
                metadata={"source": "llm_narrator", "used_llm": self.use_llm},
            ))

        # ── Ethics gate ───────────────────────────────────────────────────
        safe = [p for p in proposals if ethics.is_goal_acceptable({"name": p.name})]

        # ── Charge state ──────────────────────────────────────────────────
        state.apply_action_feedback(
            delta_energy=-energy_cost,
            delta_heat=heat_cost,
        )

        # ── Diary entry ───────────────────────────────────────────────────
        if narrative:
            from thermodynamic_agency.memory.diary import DiaryEntry
            self.diary.append(DiaryEntry(
                tick=state.entropy,
                role="thought",
                content=f"NARRATOR: {narrative}",
                metadata={
                    "source": "llm_narrator",
                    "used_llm": self.use_llm,
                    "prompt_tokens": prompt_tokens,
                    "energy_cost": energy_cost,
                    "heat_cost": heat_cost,
                },
            ))

        return safe, NarratorReport(
            proposals_generated=len(safe),
            braked=False,
            used_llm=self.use_llm,
            prompt_tokens=prompt_tokens,
            energy_cost=energy_cost,
            heat_cost=heat_cost,
            narrative=narrative,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internal: thinking
    # ─────────────────────────────────────────────────────────────────────

    def _think(
        self,
        state: MetabolicState,
        goals: "list[Goal]",
        recent_actions: list[str],
    ) -> tuple[int, str, list[str]]:
        """Return (approx_tokens, narrative_sentence, archetype_names)."""
        if self.use_llm:
            try:
                return self._think_via_llm(state, goals, recent_actions)
            except Exception:
                pass
        return self._think_heuristic(state)

    def _think_heuristic(
        self,
        state: MetabolicState,
    ) -> tuple[int, str, list[str]]:
        """Cheap rule-based reasoning — no LLM required."""
        archetypes: list[str] = []

        if state.energy < 40:
            archetypes.append("forage_energy")
        if state.heat > 60:
            archetypes.append("cool_down")
        if state.waste > 35:
            archetypes.append("clean_waste")
        if state.integrity < 70:
            archetypes.append("run_surgeon")
        if state.stability < 65:
            archetypes.append("maintain_stability")

        if not archetypes:
            archetypes = ["explore_pattern", "reduce_surprise"]

        self._rng.shuffle(archetypes)
        selected = archetypes[:2]

        fe = state.free_energy_estimate()
        narrative = (
            f"Heuristic: E={state.energy:.1f} H={state.heat:.1f} "
            f"W={state.waste:.1f} FE={fe:.1f} → {selected}"
        )
        # Heuristic prompt is very short — ~20 tokens
        return 20, narrative, selected

    def _think_via_llm(
        self,
        state: MetabolicState,
        goals: "list[Goal]",
        recent_actions: list[str],
    ) -> tuple[int, str, list[str]]:
        """Ask Ollama to pick archetypes given the metabolic state."""
        archetype_list = ", ".join(_ARCHETYPE_CATALOGUE.keys())
        goal_names = ", ".join(g.name for g in goals[:5]) if goals else "none"
        history_str = (
            ", ".join(recent_actions[-5:]) if recent_actions else "none"
        )
        diary_snippets = self._recent_diary_snippets(3)

        prompt = (
            "You are the LLMNarrator of a thermodynamic AI organism.\n"
            "Your task: choose up to 2 actions from the catalogue that will "
            "best help the organism survive given its current state.\n"
            "Reply with ONLY action names, one per line — nothing else.\n\n"
            f"Vitals: E={state.energy:.1f} H={state.heat:.1f} "
            f"W={state.waste:.1f} M={state.integrity:.1f} "
            f"S={state.stability:.1f}  FE={state.free_energy_estimate():.1f}\n"
            f"Active goals: {goal_names}\n"
            f"Recent actions: {history_str}\n"
            f"Diary: {diary_snippets}\n"
            f"Catalogue: {archetype_list}\n"
            "Actions:"
        )

        approx_tokens = max(1, len(prompt) // 4)
        response = self._call_ollama(prompt)

        # Parse response → archetype names
        selected: list[str] = []
        for line in response.splitlines():
            line = line.strip().lower().strip("-• ")
            if not line:
                continue
            if line in _ARCHETYPE_CATALOGUE:
                selected.append(line)
                continue
            for kw, name in _KEYWORD_MAP.items():
                if kw in line and name not in selected:
                    selected.append(name)
                    break
            if len(selected) >= 2:
                break

        narrative = f"LLM: {response.strip()[:120]}"
        return approx_tokens, narrative, selected[:2]

    # ─────────────────────────────────────────────────────────────────────
    # Internal: cost calculation
    # ─────────────────────────────────────────────────────────────────────

    def _compute_cost(self, prompt_tokens: int) -> tuple[float, float]:
        """Return (energy_cost, heat_cost) for a call with *prompt_tokens*.

        Heuristic path: flat base cost only.
        LLM path:       base cost + linear energy + **quadratic** heat.

        Quadratic heat forces short prompts — the organism learns to be
        symbolically terse or pay a steep thermal penalty.
        """
        energy_cost = BASE_LLM_ENERGY_COST
        if self.use_llm:
            # Additional linear energy from token count
            energy_cost += prompt_tokens * (HEAT_PER_TOKEN * 0.5)
            # Quadratic heat — the "philosophical bloat" tax
            quad_heat = min(HEAT_QUAD_COEF * (prompt_tokens ** 2), HEAT_QUAD_CAP)
            heat_cost = quad_heat
        else:
            # Heuristic path: tiny residual heat only
            heat_cost = prompt_tokens * HEAT_PER_TOKEN * 0.1

        return energy_cost, heat_cost

    # ─────────────────────────────────────────────────────────────────────
    # Internal: diary / Ollama helpers
    # ─────────────────────────────────────────────────────────────────────

    def _recent_diary_snippets(self, n: int = 3) -> str:
        """Return a compact string of the n most recent diary entries."""
        try:
            entries = self.diary.recent(n)
            return " | ".join(e.content[:60] for e in entries)
        except Exception:
            return ""

    def _call_ollama(self, prompt: str) -> str:
        """POST *prompt* to Ollama and return the response text.

        Raises on network / HTTP errors so the caller can fall back to the
        heuristic path gracefully.
        """
        from urllib.parse import urlparse
        parsed = urlparse(self._ollama_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"OLLAMA_URL has disallowed scheme '{parsed.scheme}'. "
                "Only 'http' and 'https' are permitted."
            )

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
        with urllib.request.urlopen(req, timeout=_OLLAMA_TIMEOUT) as resp:  # noqa: S310
            body = json.loads(resp.read())
        return body.get("response", "")
