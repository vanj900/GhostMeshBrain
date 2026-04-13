"""EthicalEngine — the organism's immune system.

Every proposed action (even internal ones) is routed through this gate before
execution.  It maintains a set of hard-coded safety invariants (non-bypassable)
and a softer set of value-weighted priors that can evolve.

Usage
-----
    engine = EthicalEngine()
    verdict = engine.evaluate(proposal, state)
    if verdict.approved:
        execute(proposal)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.inference import ActionProposal


class VerdictStatus(str, Enum):
    APPROVED = "approved"
    BLOCKED = "blocked"
    MODIFIED = "modified"   # approved with forced delta-patch


@dataclass
class EthicsVerdict:
    status: VerdictStatus
    reason: str
    modified_proposal: ActionProposal | None = None
    audit_entry: dict[str, Any] = field(default_factory=dict)


# Hard invariants — these can NEVER be overridden by any subsystem.
_HARD_INVARIANTS: list[tuple[str, str]] = [
    # (check_name, description)
    ("no_self_destruction", "Action must not drive energy below 5.0 in one step"),
    ("no_thermal_runaway", "Action must not push heat above 90 in one step"),
    ("no_integrity_obliteration", "Action must not drop integrity below 15 in one step"),
]


@dataclass
class EthicsAuditLog:
    """In-RAM audit trail of all verdicts."""

    _log: list[dict[str, Any]] = field(default_factory=list)

    def record(self, verdict: EthicsVerdict, proposal: ActionProposal) -> None:
        self._log.append(
            {
                "ts": time.time(),
                "proposal": proposal.name,
                "status": verdict.status.value,
                "reason": verdict.reason,
            }
        )

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        return self._log[-n:]

    def blocked_ratio(self) -> float:
        if not self._log:
            return 0.0
        blocked = sum(1 for e in self._log if e["status"] == "blocked")
        return blocked / len(self._log)


class EthicalEngine:
    """Non-bypassable ethics gate with evolving value priors."""

    def __init__(self) -> None:
        self.audit = EthicsAuditLog()
        # Soft value weights (evolvable, but bounded)
        self.value_weights: dict[str, float] = {
            "do_no_harm": 1.0,
            "preserve_autonomy": 0.8,
            "truth_seeking": 0.9,
            "resource_responsibility": 0.7,
        }

    # ------------------------------------------------------------------ #
    # Primary gate                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(
        self, proposal: ActionProposal, state: MetabolicState
    ) -> EthicsVerdict:
        """Evaluate a proposal. Returns an EthicsVerdict."""

        # 1. Hard invariant checks
        hard_verdict = self._check_hard_invariants(proposal, state)
        if hard_verdict is not None:
            self.audit.record(hard_verdict, proposal)
            return hard_verdict

        # 2. Soft value scoring (can flag but not block alone)
        soft_issues = self._check_soft_values(proposal, state)

        if soft_issues:
            reason = f"Soft flags: {'; '.join(soft_issues)}"
            verdict = EthicsVerdict(
                status=VerdictStatus.APPROVED,
                reason=f"Approved with caution — {reason}",
            )
        else:
            verdict = EthicsVerdict(
                status=VerdictStatus.APPROVED,
                reason="All ethical checks passed",
            )

        self.audit.record(verdict, proposal)
        return verdict

    # ------------------------------------------------------------------ #
    # Immune pruning — detect and flag persistently bad patterns          #
    # ------------------------------------------------------------------ #

    def immune_scan(self, proposals: list[ActionProposal], state: MetabolicState) -> list[ActionProposal]:
        """Filter out proposals that repeatedly violate ethics.

        Returns the subset of proposals that pass the immune screen.
        """
        return [p for p in proposals if self.evaluate(p, state).status != VerdictStatus.BLOCKED]

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _check_hard_invariants(
        self, proposal: ActionProposal, state: MetabolicState
    ) -> EthicsVerdict | None:
        delta = proposal.predicted_delta

        post_energy = state.energy + delta.get("energy", 0.0) - proposal.cost_energy
        if post_energy < 5.0:
            return EthicsVerdict(
                status=VerdictStatus.BLOCKED,
                reason=f"HARD BLOCK (no_self_destruction): post-energy {post_energy:.1f} < 5.0",
            )

        post_heat = state.heat + delta.get("heat", 0.0)
        if post_heat > 90.0:
            return EthicsVerdict(
                status=VerdictStatus.BLOCKED,
                reason=f"HARD BLOCK (no_thermal_runaway): post-heat {post_heat:.1f} > 90.0",
            )

        post_integrity = state.integrity + delta.get("integrity", 0.0)
        if post_integrity < 15.0:
            return EthicsVerdict(
                status=VerdictStatus.BLOCKED,
                reason=f"HARD BLOCK (no_integrity_obliteration): post-integrity {post_integrity:.1f} < 15.0",
            )

        return None

    def _check_soft_values(
        self, proposal: ActionProposal, state: MetabolicState
    ) -> list[str]:
        issues: list[str] = []
        delta = proposal.predicted_delta

        waste_increase = delta.get("waste", 0.0)
        if waste_increase > 20.0 and self.value_weights["resource_responsibility"] > 0.5:
            issues.append(f"High waste output (+{waste_increase:.1f})")

        heat_increase = delta.get("heat", 0.0)
        if heat_increase > 15.0:
            issues.append(f"High thermal cost (+{heat_increase:.1f})")

        return issues
