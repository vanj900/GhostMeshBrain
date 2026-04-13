"""Janitor — waste management and context summarisation.

Triggered when MetabolicState.tick() returns "REST" (high waste or heat).

Responsibilities
----------------
1. Compress the N most-recent diary entries into 5-10 insight bullets.
2. Persist compressed insights to the diary's insight table.
3. Wipe raw entries that have been compressed.
4. Apply hard metabolic cooling: -35 waste, -28 heat (configurable).
5. Optionally call an LLM (via Ollama HTTP) for higher-quality summaries.
"""

from __future__ import annotations

import json
import os
import textwrap
import urllib.request
from dataclasses import dataclass

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.memory.diary import RamDiary, DiaryEntry

# Metabolic rewards for a Janitor pass
JANITOR_WASTE_REDUCTION: float = float(os.environ.get("JANITOR_WASTE_DELTA", "-35"))
JANITOR_HEAT_REDUCTION: float = float(os.environ.get("JANITOR_HEAT_DELTA", "-28"))
JANITOR_ENERGY_COST: float = float(os.environ.get("JANITOR_ENERGY_COST", "4"))

OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "mistral")

# LLM prompt complexity cost coefficients.
# Longer / more detailed prompts burn more metabolic resources, forcing the
# system to prefer efficient summarisation over exhaustive context dumping.
# Calibrated: 500-char prompt → +0.40 energy, +0.20 heat on top of base cost;
#             2000-char prompt → +1.60 energy, +0.80 heat.
LLM_ENERGY_COST_PER_CHAR: float = float(os.environ.get("JANITOR_LLM_E_PER_CHAR", "0.0008"))
LLM_HEAT_COST_PER_CHAR: float = float(os.environ.get("JANITOR_LLM_H_PER_CHAR", "0.0004"))


@dataclass
class JanitorReport:
    entries_compressed: int
    insights_generated: int
    summary: str
    delta_waste: float
    delta_heat: float
    delta_energy: float


class Janitor:
    """Waste-management subsystem — compresses diary, cools metabolic state."""

    def __init__(self, diary: RamDiary, use_llm: bool = False) -> None:
        self.diary = diary
        self.use_llm = use_llm
        # Populated by _compress_via_llm so run() can apply the LLM cost
        self._pending_llm_cost: tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------ #
    # Main entry-point                                                     #
    # ------------------------------------------------------------------ #

    def run(self, state: MetabolicState, compress_last_n: int = 50) -> JanitorReport:
        """Run a full Janitor pass.

        Parameters
        ----------
        state:
            MetabolicState — will be mutated via apply_action_feedback().
        compress_last_n:
            How many diary entries to collapse into insights.

        Returns
        -------
        JanitorReport
        """
        # Reset pending LLM cost each run
        self._pending_llm_cost = (0.0, 0.0)
        entries = self.diary.recent(compress_last_n)
        if not entries:
            return JanitorReport(
                entries_compressed=0,
                insights_generated=0,
                summary="Diary empty — nothing to compress.",
                delta_waste=0.0,
                delta_heat=0.0,
                delta_energy=0.0,
            )

        insights = self._compress(entries)

        # Persist insights and wipe raw entries
        if entries:
            first_id = min(e.id for e in entries if e.id is not None)
            last_id = max(e.id for e in entries if e.id is not None)
        else:
            first_id = last_id = 0

        for insight_text in insights:
            self.diary.add_insight(insight_text, first_id, last_id)

        # Wipe compressed entries (keep any after the last compressed id)
        self._wipe_range(first_id, last_id)

        # Apply metabolic feedback — base cost plus any LLM prompt cost
        llm_energy_cost, llm_heat_cost = self._pending_llm_cost
        delta_energy = -JANITOR_ENERGY_COST - llm_energy_cost
        state.apply_action_feedback(
            delta_energy=delta_energy,
            delta_heat=JANITOR_HEAT_REDUCTION + llm_heat_cost,
            delta_waste=JANITOR_WASTE_REDUCTION,
            delta_integrity=1.5,
        )

        return JanitorReport(
            entries_compressed=len(entries),
            insights_generated=len(insights),
            summary="\n".join(f"• {i}" for i in insights),
            delta_waste=JANITOR_WASTE_REDUCTION,
            delta_heat=JANITOR_HEAT_REDUCTION,
            delta_energy=delta_energy,
        )

    # ------------------------------------------------------------------ #
    # Compression strategies                                               #
    # ------------------------------------------------------------------ #

    def _compress(self, entries: list[DiaryEntry]) -> list[str]:
        if self.use_llm:
            try:
                return self._compress_via_llm(entries)
            except Exception:
                pass
        return self._compress_heuristic(entries)

    def _compress_heuristic(self, entries: list[DiaryEntry]) -> list[str]:
        """Simple rule-based compression — bucket by role, extract key lines."""
        buckets: dict[str, list[str]] = {}
        for e in entries:
            buckets.setdefault(e.role, []).append(e.content)

        insights: list[str] = []
        for role, contents in buckets.items():
            # Take the last (most recent) 2 of each role
            sample = contents[-2:]
            for c in sample:
                short = textwrap.shorten(c, width=120, placeholder="…")
                insights.append(f"[{role}] {short}")

        return insights[:10]  # cap at 10 insights

    def _compress_via_llm(self, entries: list[DiaryEntry]) -> list[str]:
        """Summarise via Ollama local LLM.

        The prompt length is a proxy for computational complexity — longer
        prompts burn more energy and generate more heat.  This cost is charged
        to the metabolic state *before* the result is applied so that the
        organism truly pays for expensive summarisation.
        """
        text = "\n".join(
            f"[{e.role}] {e.content}" for e in entries
        )
        prompt = (
            "You are the Janitor subsystem of a thermodynamic AI organism. "
            "Summarise the following diary entries into 5-10 concise insight bullets "
            "(each on its own line, starting with a dash):\n\n" + text
        )

        # Charge metabolic cost proportional to prompt complexity.
        # Calibrated: 500 chars ≈ +0.40 energy, +0.20 heat (on top of base cost).
        # 2000 chars ≈ +1.60 energy, +0.80 heat — genuinely expensive.
        prompt_chars = len(prompt)
        llm_energy_cost = prompt_chars * LLM_ENERGY_COST_PER_CHAR
        llm_heat_cost = prompt_chars * LLM_HEAT_COST_PER_CHAR
        # This is stored so _compress can pass it back via run(); charge inline
        # via a dedicated callback stored on the instance during this call.
        self._pending_llm_cost = (llm_energy_cost, llm_heat_cost)

        payload = json.dumps(
            {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            body = json.loads(resp.read())
        raw = body.get("response", "")
        lines = [l.lstrip("- ").strip() for l in raw.splitlines() if l.strip().startswith("-")]
        return lines[:10] if lines else self._compress_heuristic(entries)

    # ------------------------------------------------------------------ #
    # Diary surgery                                                        #
    # ------------------------------------------------------------------ #

    def _wipe_range(self, from_id: int, to_id: int) -> None:
        assert self.diary._conn is not None
        self.diary._conn.execute(
            "DELETE FROM entries WHERE id >= ? AND id <= ?", (from_id, to_id)
        )
        self.diary._conn.commit()
