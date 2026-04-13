"""Tests for EthicalEngine."""

import pytest

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.cognition.ethics import (
    EthicalEngine,
    EthicsVerdict,
    VerdictStatus,
)
from thermodynamic_agency.cognition.inference import ActionProposal


class TestEthicalEngine:
    def _engine(self):
        return EthicalEngine()

    def _proposal(self, name="test", delta=None, cost=1.0):
        return ActionProposal(
            name=name,
            description="test proposal",
            predicted_delta=delta or {},
            cost_energy=cost,
        )

    def test_approves_safe_proposal(self):
        engine = self._engine()
        state = MetabolicState(energy=80.0, heat=20.0, integrity=80.0)
        proposal = self._proposal(delta={"energy": 5.0})
        verdict = engine.evaluate(proposal, state)
        assert verdict.status == VerdictStatus.APPROVED

    def test_blocks_self_destruction(self):
        """Action that drops energy to near zero must be blocked."""
        engine = self._engine()
        state = MetabolicState(energy=10.0)
        # net energy after cost = 10 - 9 (delta) - 2 (cost) = -1 → blocked
        proposal = self._proposal(delta={"energy": -9.0}, cost=2.0)
        verdict = engine.evaluate(proposal, state)
        assert verdict.status == VerdictStatus.BLOCKED
        assert "no_self_destruction" in verdict.reason

    def test_blocks_thermal_runaway(self):
        engine = self._engine()
        state = MetabolicState(heat=75.0)
        proposal = self._proposal(delta={"heat": 20.0})
        verdict = engine.evaluate(proposal, state)
        assert verdict.status == VerdictStatus.BLOCKED
        assert "no_thermal_runaway" in verdict.reason

    def test_blocks_integrity_obliteration(self):
        engine = self._engine()
        state = MetabolicState(integrity=20.0)
        proposal = self._proposal(delta={"integrity": -10.0})
        verdict = engine.evaluate(proposal, state)
        assert verdict.status == VerdictStatus.BLOCKED
        assert "no_integrity_obliteration" in verdict.reason

    def test_audit_log_records_verdicts(self):
        engine = self._engine()
        state = MetabolicState()
        proposal = self._proposal()
        engine.evaluate(proposal, state)
        engine.evaluate(proposal, state)
        assert len(engine.audit.recent()) == 2

    def test_immune_scan_filters_blocked(self):
        engine = self._engine()
        state = MetabolicState(energy=10.0)
        dangerous = self._proposal("dangerous", delta={"energy": -9.0}, cost=2.0)
        safe = self._proposal("safe", delta={"energy": 5.0})
        result = engine.immune_scan([dangerous, safe], state)
        names = [p.name for p in result]
        assert "safe" in names
        assert "dangerous" not in names

    def test_blocked_ratio(self):
        engine = self._engine()
        state = MetabolicState(energy=10.0)
        dangerous = self._proposal("dangerous", delta={"energy": -9.0}, cost=2.0)
        safe = self._proposal("safe", delta={"energy": 5.0})
        engine.evaluate(dangerous, state)
        engine.evaluate(safe, state)
        ratio = engine.audit.blocked_ratio()
        assert 0.0 < ratio < 1.0
