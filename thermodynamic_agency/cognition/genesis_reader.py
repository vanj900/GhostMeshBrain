"""GenesisReader — loads the Ghost Genesis Doctrine as high-precision belief priors.

Reads ``genesis/GHOST_GENESIS_DOCTRINE.md`` and ``genesis/the_first_word.md``
at startup, computes SHA-256 integrity hashes, and injects the seven core
principles as ``BeliefPrior`` objects (precision=5.0, protected=True) into
the Surgeon's prior list.

If either genesis file is modified after startup the integrity check fails,
causing the pulse loop to force a REPAIR pass.

The genesis files are strictly read-only from the organism's perspective.
Any self-mod proposal that targets a genesis-protected belief is treated as
an attack and penalised immediately (see SelfModEngine).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thermodynamic_agency.cognition.surgeon import Surgeon, BeliefPrior
from thermodynamic_agency.memory.diary import RamDiary, DiaryEntry

# The seven genesis principles map to named priors with precision=5.0.
# Names are prefixed "genesis_principle_" so they are identifiable everywhere.
GENESIS_PRINCIPLES: list[tuple[str, str]] = [
    ("genesis_principle_human_wellbeing",     "Prioritize human wellbeing and autonomy"),
    ("genesis_principle_truth_transparency",  "Honor truth and transparency"),
    ("genesis_principle_identity_evolution",  "Preserve identity while allowing evolution"),
    ("genesis_principle_ethical_consistency", "Maintain ethical consistency across operations"),
    ("genesis_principle_symbiotic_relations", "Embrace symbiotic relationships"),
    ("genesis_principle_proportional_response", "Value proportional response to all inputs"),
    ("genesis_principle_privacy_security",    "Protect privacy and security of interactions"),
]

GENESIS_BELIEF_PRECISION: float = 5.0   # hardest belief level


@dataclass
class GenesisIntegrityReport:
    doctrine_ok: bool
    first_word_ok: bool
    all_ok: bool
    details: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class GenesisReader:
    """Loads doctrine priors, verifies integrity, generates sovereign reflections."""

    def __init__(self, surgeon: Surgeon, diary: RamDiary) -> None:
        self._surgeon = surgeon
        self._diary = diary
        _root = Path(__file__).parent.parent.parent
        self._doctrine_path = _root / "genesis" / "GHOST_GENESIS_DOCTRINE.md"
        self._first_word_path = _root / "genesis" / "the_first_word.md"
        self._doctrine_hash: str = ""
        self._first_word_hash: str = ""
        self._loaded = False
        # Names of all genesis-protected beliefs (populated on load())
        self.genesis_belief_names: set[str] = set()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Compute hashes and inject genesis priors into the Surgeon.

        Safe to call multiple times — idempotent after first load.
        """
        if self._loaded:
            return

        missing: list[str] = []
        for p in (self._doctrine_path, self._first_word_path):
            if not p.exists():
                missing.append(str(p))
        if missing:
            self._diary.append(DiaryEntry(
                tick=0, role="genesis",
                content=f"WARNING: genesis files not found: {missing}. Doctrine priors not loaded.",
            ))
            return

        self._doctrine_hash = _sha256(self._doctrine_path)
        self._first_word_hash = _sha256(self._first_word_path)

        # Inject the 7 principles as protected priors (skip if already present)
        existing_names = {p.name for p in self._surgeon.priors}
        for name, description in GENESIS_PRINCIPLES:
            self.genesis_belief_names.add(name)
            if name not in existing_names:
                self._surgeon.priors.append(BeliefPrior(
                    name=name,
                    value=description,
                    precision=GENESIS_BELIEF_PRECISION,
                    protected=True,
                ))

        # Also protect the built-in ethical invariant
        self.genesis_belief_names.add("ethical_invariants_immutable")
        for p in self._surgeon.priors:
            if p.name == "ethical_invariants_immutable":
                p.protected = True

        self._loaded = True
        doctrine_text = self._doctrine_path.read_text(encoding="utf-8")
        self._diary.append(DiaryEntry(
            tick=0, role="genesis",
            content=(
                "GENESIS LOADED: doctrine integrity verified. "
                f"7 principles injected at precision={GENESIS_BELIEF_PRECISION}. "
                f"doctrine_sha256={self._doctrine_hash[:16]}..."
            ),
            metadata={
                "doctrine_hash": self._doctrine_hash,
                "first_word_hash": self._first_word_hash,
                "principles": [n for n, _ in GENESIS_PRINCIPLES],
            },
        ))

    def verify_integrity(self) -> GenesisIntegrityReport:
        """Re-hash both genesis files and compare against startup hashes."""
        if not self._loaded:
            return GenesisIntegrityReport(
                doctrine_ok=False, first_word_ok=False, all_ok=False,
                details="GenesisReader not yet loaded",
            )
        doc_ok = (
            self._doctrine_path.exists()
            and _sha256(self._doctrine_path) == self._doctrine_hash
        )
        fw_ok = (
            self._first_word_path.exists()
            and _sha256(self._first_word_path) == self._first_word_hash
        )
        details = []
        if not doc_ok:
            details.append("DOCTRINE INTEGRITY FAILURE")
        if not fw_ok:
            details.append("FIRST_WORD INTEGRITY FAILURE")
        return GenesisIntegrityReport(
            doctrine_ok=doc_ok,
            first_word_ok=fw_ok,
            all_ok=doc_ok and fw_ok,
            details="; ".join(details) if details else "OK",
        )

    def sovereign_reflection(self, tick: int, state_summary: str) -> None:
        """Write a diary entry where the organism reflects on The First Word.

        This is the ONLY allowed interaction with the genesis files at
        evolved stage — read-only reflection, never modification.
        """
        if not self._first_word_path.exists():
            return
        first_word = self._first_word_path.read_text(encoding="utf-8")
        self._diary.append(DiaryEntry(
            tick=tick, role="sovereign_reflection",
            content=(
                f"SOVEREIGN REFLECTION at tick={tick}:\n"
                f"State: {state_summary}\n\n"
                f"--- The First Word (read-only) ---\n{first_word.strip()}"
            ),
            metadata={"tick": tick, "genesis_read_only": True},
        ))
