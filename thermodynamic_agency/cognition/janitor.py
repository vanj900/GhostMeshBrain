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

        # Apply metabolic feedback
        delta_energy = -JANITOR_ENERGY_COST
        state.apply_action_feedback(
            delta_energy=delta_energy,
            delta_heat=JANITOR_HEAT_REDUCTION,
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
        """Summarise via Ollama local LLM."""
        text = "\n".join(
            f"[{e.role}] {e.content}" for e in entries
        )
        prompt = (
            "You are the Janitor subsystem of a thermodynamic AI organism. "
            "Summarise the following diary entries into 5-10 concise insight bullets "
            "(each on its own line, starting with a dash):\n\n" + text
        )
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
