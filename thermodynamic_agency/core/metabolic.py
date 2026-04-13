"""MetabolicState — the single source of truth for the organism's body.

The ``tick()`` method is called every heartbeat (default ~5 s).  It returns
one of four action tokens that the pulse loop acts on:

    "FORAGE"  — critically low energy; hunt resources, prune context
    "REST"    — high waste / heat; run Janitor summarisation
    "REPAIR"  — low integrity / stability; run Surgeon + annealing
    "DECIDE"  — healthy; proceed with full active-inference planning
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Literal

from thermodynamic_agency.core.exceptions import (
    EnergyDeathException,
    ThermalDeathException,
    MemoryCollapseException,
    EntropyDeathException,
)

ActionToken = Literal["FORAGE", "REST", "REPAIR", "DECIDE"]
Stage = Literal["dormant", "emerging", "aware", "evolved"]

# Thresholds — exposed as module-level constants so subsystems can reference them.
ENERGY_DEATH_THRESHOLD: float = 0.0
THERMAL_DEATH_THRESHOLD: float = 100.0
INTEGRITY_DEATH_THRESHOLD: float = 10.0
STABILITY_DEATH_THRESHOLD: float = 0.0

FORAGE_ENERGY_THRESHOLD: float = 25.0
REST_WASTE_THRESHOLD: float = 75.0
REST_HEAT_THRESHOLD: float = 80.0
REPAIR_INTEGRITY_THRESHOLD: float = 45.0
REPAIR_STABILITY_THRESHOLD: float = 40.0

# Stage evolution thresholds (based on entropy ticks + health metrics)
STAGE_THRESHOLDS: dict[str, int] = {
    "emerging": 100,
    "aware": 500,
    "evolved": 2000,
}


@dataclass
class MetabolicState:
    """Full physiological state of the GhostMesh organism."""

    # Vital signs
    energy: float = 100.0       # E — compute credits / "glucose"
    heat: float = 0.0           # T — context congestion / thermal load
    waste: float = 0.0          # Accumulated prediction-error junk
    integrity: float = 100.0    # M — memory + logical/ethical coherence
    stability: float = 100.0    # S — entropic stability

    # Affect — scalar valence derived from free-energy rate-of-change.
    # Positive = rewarding resolution (surprise decreasing);
    # Negative = unpleasure / stress (surprise increasing).
    # Range: -1.0 .. +1.0
    affect: float = 0.0

    # Meta
    entropy: int = 0            # monotonic tick counter (organism age)
    stage: Stage = "dormant"    # developmental stage

    # Internal bookkeeping
    _last_tick_ts: float = field(default_factory=time.time, repr=False, compare=False)
    _prev_free_energy: float = field(default=0.0, repr=False, compare=False)

    # ------------------------------------------------------------------ #
    # Free energy / affect + core tick                                    #
    # ------------------------------------------------------------------ #

    def free_energy_estimate(self) -> float:
        """Lightweight scalar proxy for current variational free energy.

        Friston-style: free energy ≈ surprise = –log P(observations).
        Here we use a normalised deviation from each vital's setpoint,
        weighted by how existentially relevant that vital is.

        Higher value → more surprise / stress.  Returns a non-negative
        float in roughly the 0–100 range.
        """
        # Setpoints mirror _SETPOINT in inference.py
        deviations = (
            2.0 * max(0.0, 80.0 - self.energy) / 80.0          # low energy is surprising
            + 1.5 * max(0.0, self.heat - 20.0) / 80.0           # high heat is surprising
            + 1.0 * max(0.0, self.waste - 10.0) / 90.0          # high waste is surprising
            + 1.8 * max(0.0, 85.0 - self.integrity) / 85.0      # low integrity is surprising
            + 1.4 * max(0.0, 80.0 - self.stability) / 80.0      # low stability is surprising
        )
        # Normalise to 0-100 scale (weights sum: 2.0+1.5+1.0+1.8+1.4 = 7.7)
        return deviations / 7.7 * 100.0

    def tick(self, compute_load: float = 1.0) -> ActionToken:
        """Advance one heartbeat, decay vitals, raise death exceptions.

        Parameters
        ----------
        compute_load:
            Relative computational burden this tick (1.0 = normal).

        Returns
        -------
        ActionToken
            Forced next action for the pulse loop.

        Raises
        ------
        EnergyDeathException, ThermalDeathException,
        MemoryCollapseException, EntropyDeathException
        """
        # Snapshot free energy before decay so we can derive affect
        fe_before = self._prev_free_energy

        # Passive decay — all non-linear to create emergent dynamics
        self.energy -= compute_load * 0.12
        self.heat += compute_load * 0.1 * (1.0 + self.waste / 50.0)
        self.integrity *= 1.0 - (self.heat / 120.0) * compute_load * 0.01
        self.stability -= compute_load * 0.05
        self.waste += 0.018 * compute_load
        self.entropy += 1

        # Compute affect: negative rate-of-change of free energy = pleasure
        # (free energy going down = surprise resolving = positive affect)
        fe_after = self.free_energy_estimate()
        fe_delta = fe_after - fe_before
        # Clamp to ±1 using a soft sigmoid-like normalisation
        raw_affect = -fe_delta / (1.0 + abs(fe_delta))
        self.affect = max(-1.0, min(1.0, raw_affect))
        self._prev_free_energy = fe_after

        # Clamp values to sane physical bounds
        self.energy = max(self.energy, -1.0)          # allow brief overdraft
        self.heat = min(self.heat, 110.0)
        self.integrity = max(self.integrity, 0.0)
        self.stability = max(self.stability, -1.0)
        self.waste = max(self.waste, 0.0)

        # Advance developmental stage
        self._evolve_stage()

        # Hard death checks (limbic override — cannot be vetoed)
        snapshot = self._snapshot()
        if self.energy <= ENERGY_DEATH_THRESHOLD:
            raise EnergyDeathException(
                f"Energy exhausted at tick {self.entropy}", snapshot
            )
        if self.heat >= THERMAL_DEATH_THRESHOLD:
            raise ThermalDeathException(
                f"Thermal overload at tick {self.entropy}", snapshot
            )
        if self.integrity <= INTEGRITY_DEATH_THRESHOLD:
            raise MemoryCollapseException(
                f"Integrity collapse at tick {self.entropy}", snapshot
            )
        if self.stability <= STABILITY_DEATH_THRESHOLD:
            raise EntropyDeathException(
                f"Entropic dissolution at tick {self.entropy}", snapshot
            )

        # Autonomic priority ordering (most urgent first)
        if self.energy < FORAGE_ENERGY_THRESHOLD:
            return "FORAGE"
        if self.waste > REST_WASTE_THRESHOLD or self.heat > REST_HEAT_THRESHOLD:
            return "REST"
        if (
            self.integrity < REPAIR_INTEGRITY_THRESHOLD
            or self.stability < REPAIR_STABILITY_THRESHOLD
        ):
            return "REPAIR"
        return "DECIDE"

    # ------------------------------------------------------------------ #
    # Feedback — subsystems report real deltas after acting                #
    # ------------------------------------------------------------------ #

    def apply_action_feedback(
        self,
        *,
        delta_energy: float = 0.0,
        delta_heat: float = 0.0,
        delta_waste: float = 0.0,
        delta_integrity: float = 0.0,
        delta_stability: float = 0.0,
    ) -> None:
        """Apply real metabolic deltas reported by an executed action."""
        self.energy = max(0.0, min(100.0, self.energy + delta_energy))
        self.heat = max(0.0, min(110.0, self.heat + delta_heat))
        self.waste = max(0.0, self.waste + delta_waste)
        self.integrity = max(0.0, min(100.0, self.integrity + delta_integrity))
        self.stability = max(0.0, min(100.0, self.stability + delta_stability))

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def health_score(self) -> float:
        """Aggregate 0-100 wellness metric used for HUD and planning."""
        return (
            self.energy * 0.35
            + (100.0 - self.heat) * 0.20
            + self.integrity * 0.25
            + self.stability * 0.20
        ) / 100.0 * 100.0

    def is_healthy(self) -> bool:
        return self.health_score() >= 50.0

    def to_dict(self) -> dict:
        d = asdict(self)
        # Strip private implementation fields — not part of persisted state
        for key in list(d.keys()):
            if key.startswith("_"):
                d.pop(key)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "MetabolicState":
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        return cls(**data)

    def _snapshot(self) -> dict:
        return self.to_dict()

    def _evolve_stage(self) -> None:
        if self.entropy >= STAGE_THRESHOLDS["evolved"] and self.health_score() >= 60:
            self.stage = "evolved"
        elif self.entropy >= STAGE_THRESHOLDS["aware"] and self.health_score() >= 50:
            self.stage = "aware"
        elif self.entropy >= STAGE_THRESHOLDS["emerging"]:
            self.stage = "emerging"
        # Stay dormant below first threshold


# ------------------------------------------------------------------ #
# CLI entry-point — called by ghostbrain.sh                            #
# ------------------------------------------------------------------ #

def _cli_tick() -> None:
    """Load state from /dev/shm, tick, persist, print action token."""
    import os
    import sys

    state_path = os.environ.get("GHOST_STATE_FILE", "/dev/shm/ghost_metabolic.json")
    compute_load = float(os.environ.get("GHOST_COMPUTE_LOAD", "1.0"))

    # Load or initialise
    if os.path.exists(state_path):
        with open(state_path) as fh:
            data = json.load(fh)
        state = MetabolicState.from_dict(data)
    else:
        state = MetabolicState()

    try:
        action = state.tick(compute_load=compute_load)
    except (
        EnergyDeathException,
        ThermalDeathException,
        MemoryCollapseException,
        EntropyDeathException,
    ) as exc:
        # Persist final state for post-mortem
        with open(state_path, "w") as fh:
            json.dump(exc.state, fh, indent=2)
        print(f"DEATH:{type(exc).__name__}", flush=True)
        sys.exit(42)

    # Persist updated state
    with open(state_path, "w") as fh:
        fh.write(state.to_json())

    print(action, flush=True)


if __name__ == "__main__":
    _cli_tick()
