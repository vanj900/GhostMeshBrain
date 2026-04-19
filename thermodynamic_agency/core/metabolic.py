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

# Anticipatory REPAIR thresholds (Phase 3)
REPAIR_ALLOSTATIC_THRESHOLD: float = 70.0   # trigger repair when load is high (raised: avoid repair-zombie at moderate AL)
REPAIR_PROJECTED_INTEGRITY: float = 50.0    # trigger if projected integrity crosses this
REPAIR_PROJECTED_STABILITY: float = 45.0    # trigger if projected stability crosses this

# Anticipatory REST threshold — REST reduces heat/waste; REPAIR adds heat/waste.
# Proactively resting at moderate AL prevents the AL↔heat feedback loop from
# escalating to crisis levels that only REPAIR (which adds heat) can nominally address.
REST_ALLOSTATIC_THRESHOLD: float = 65.0

# Stage evolution thresholds (based on entropy ticks + health metrics)
STAGE_THRESHOLDS: dict[str, int] = {
    "emerging": 100,
    "aware": 500,
    "evolved": 10000,
}

# Allostatic load constants (Phase 2)
_AL_T_SAFE: float = 35.0       # safe heat ceiling
_AL_W_SAFE: float = 25.0       # safe waste ceiling
_AL_ALPHA_T: float = 0.015     # heat contribution rate
_AL_ALPHA_W: float = 0.010     # waste contribution rate
_AL_ALPHA_FE: float = 0.003    # free-energy contribution rate
_AL_BETA_REPAIR: float = 0.22  # repair reduces allostatic load (raised: repair must break the AL↔heat loop)
_AL_BETA_REST: float = 0.14    # rest reduces allostatic load (raised: rest must be a real pressure valve)
# Evolved-stage passive allostatic recovery: when vitals are all within safe bounds
# the organism slowly self-regulates back toward baseline without needing explicit
# REST/REPAIR. This rate is calibrated so that at minimum FE (~15) and stable
# vitals (heat<40, waste<25) there is a net drain of ~0.035/tick. The condition
# gate (energy>60, heat<40, waste<25) ensures the drain only fires during genuine
# stability — if the organism is actually stressed, REST/REPAIR are still required.
_AL_PASSIVE_DRAIN_EVOLVED: float = 0.08

# Passive cooling — baseline heat dissipation every tick (applied after death checks).
# Acts as thermal radiation; prevents slow heat runaway at normal waste/load without
# rescuing terminal spikes (waste>60 + AL>80 will still overwhelm cooling and kill).
# At load=1.0 with waste≈11 and AL≈30 this keeps net heat near zero; higher waste or AL
# still accumulate heat, so REST/REPAIR remain necessary and death remains real.
_PASSIVE_COOL_RATE: float = 0.20

# Adaptive thermal throttle: when heat exceeds this level the tick() method
# applies a mild compute-load reduction (THERMAL_THROTTLE_FACTOR) that forces
# self-regulation rather than blind grinding.  Death remains possible if the
# organism ignores heat signals long enough for waste→heat cascade to take over.
_THERMAL_THROTTLE_HEAT: float = 65.0   # heat level that starts throttling
_THERMAL_THROTTLE_FACTOR: float = 0.80 # effective_load = load * factor above threshold

# DECIDE streak fatigue constants (Phase 4)
_DECIDE_STREAK_FREE: int = 10   # streak ticks before fatigue kicks in
_DECIDE_ETA_E: float = 0.01     # energy cost per streak tick beyond free threshold
_DECIDE_ETA_T: float = 0.02     # heat cost per streak tick beyond free threshold

# Affect → cortisol accumulation rate (per tick of negative affect).
# Exposed as a module constant so the ablation runner can vary affect
# sensitivity without monkey-patching the tick() method body.
_CORTISOL_RATE: float = 0.02


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

    # Hormone proxies — allostatic modulators driven by sustained affect.
    # cortisol_proxy: rises with chronic negative affect; amplifies decay rates
    #   (allostatic load); range 0-1.
    # dopamine_proxy: spikes on positive affect bursts; grants a brief DECIDE
    #   efficiency bonus; decays rapidly; range 0-1.
    cortisol_proxy: float = 0.0
    dopamine_proxy: float = 0.0

    # Allostatic load (Phase 2) — accumulated wear from repeated near-crises.
    # Range: [0, 100].  High values worsen metabolism and trigger anticipatory REPAIR.
    allostatic_load: float = 0.0

    # DECIDE streak tracking (Phase 4) — consecutive DECIDE ticks.
    # Prolonged thinking without acting becomes metabolically taxing.
    decide_streak: int = 0

    # Last executed action — used to credit repair/rest allostatic reduction
    # in the following tick.
    last_action: str = ""

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

        # Adaptive thermal throttle: reduce effective load when heat is elevated,
        # mimicking thermal-throttling in physical hardware.  Forces self-regulation
        # (REST/REPAIR) rather than blind grinding through rising heat.  Death is
        # still possible — the throttle only reduces the rate; it cannot stop a
        # waste→heat cascade that the organism is neglecting.
        if self.heat > _THERMAL_THROTTLE_HEAT:
            compute_load = compute_load * _THERMAL_THROTTLE_FACTOR

        # Passive decay — all non-linear to create emergent dynamics
        self.energy -= compute_load * 0.12
        # Non-linear waste → heat cascade: high waste amplifies heat exponentially
        # (biological analogy: inflammation runaway above critical toxin threshold).
        # Factor uses /60 (was /50) for a milder baseline waste-to-heat conversion.
        _waste_heat_factor = 1.0 + self.waste / 60.0
        if self.waste > 60.0:
            _waste_heat_factor += ((self.waste - 60.0) / 40.0) ** 2
        self.heat += compute_load * 0.1 * _waste_heat_factor
        self.integrity *= 1.0 - (self.heat / 120.0) * compute_load * 0.01
        self.stability -= compute_load * 0.05
        self.waste += 0.018 * compute_load
        self.entropy += 1

        # DECIDE streak fatigue tax (Phase 4) — prolonged cognition without
        # acting becomes metabolically expensive, nudging the organism toward
        # REPAIR / REST / FORAGE after long unbroken DECIDE runs.
        if self.last_action == "DECIDE" and self.decide_streak > _DECIDE_STREAK_FREE:
            _streak_excess = self.decide_streak - _DECIDE_STREAK_FREE
            self.energy -= _DECIDE_ETA_E * _streak_excess * compute_load
            self.heat += _DECIDE_ETA_T * _streak_excess * compute_load

        # Compute affect: negative rate-of-change of free energy = pleasure
        # (free energy going down = surprise resolving = positive affect)
        fe_after = self.free_energy_estimate()
        fe_delta = fe_after - fe_before
        # Clamp to ±1 using a soft sigmoid-like normalisation
        raw_affect = -fe_delta / (1.0 + abs(fe_delta))
        self.affect = max(-1.0, min(1.0, raw_affect))
        self._prev_free_energy = fe_after

        # ---- Hormone proxies --------------------------------------------- #
        # Cortisol: accumulates under sustained negative affect (chronic stress).
        # At high levels it amplifies both waste accumulation and heat rise —
        # the organism literally runs hotter and dirtier under allostatic load.
        if self.affect < -0.3:
            self.cortisol_proxy = min(1.0, self.cortisol_proxy + _CORTISOL_RATE * compute_load)
        else:
            self.cortisol_proxy = max(0.0, self.cortisol_proxy - 0.01)

        # Dopamine: spikes on positive affect bursts (surprise resolving fast).
        # Grants a transient efficiency bonus: less waste per tick.
        if self.affect > 0.4:
            self.dopamine_proxy = min(1.0, self.dopamine_proxy + 0.05)
        else:
            self.dopamine_proxy = max(0.0, self.dopamine_proxy - 0.03)

        # Apply allostatic load from cortisol
        if self.cortisol_proxy > 0.3:
            allostatic = (self.cortisol_proxy - 0.3) * compute_load
            self.waste += 0.015 * allostatic
            self.heat += 0.08 * allostatic

        # Apply dopamine efficiency bonus: reduce waste increment slightly
        if self.dopamine_proxy > 0.3:
            dopamine_efficiency = (self.dopamine_proxy - 0.3) * 0.5
            self.waste = max(0.0, self.waste - 0.005 * dopamine_efficiency)

        # ---- Allostatic load update (Phase 2) ----------------------------- #
        # Accumulates from sustained heat, waste, and free energy.  Repair and
        # rest reduce it.  High load feeds back into thermal and waste dynamics,
        # so stress leaves a residue even after acute crisis passes.
        al_gain = (
            _AL_ALPHA_T * max(0.0, self.heat - _AL_T_SAFE)
            + _AL_ALPHA_W * max(0.0, self.waste - _AL_W_SAFE)
            + _AL_ALPHA_FE * fe_after
        )
        al_loss = 0.0
        if self.last_action == "REPAIR":
            al_loss += _AL_BETA_REPAIR
        elif self.last_action == "REST":
            al_loss += _AL_BETA_REST
        # Passive recovery at evolved stage: when all vitals are within safe bounds
        # the organism can slowly self-regulate allostatic load without explicit
        # REST/REPAIR cycles.  This prevents the "no drain path" saturation that
        # pins AL at 65 post-evolution and locks the organism into Guardian attractor.
        if (
            self.stage == "evolved"
            and self.energy > 60.0
            and self.heat < 40.0
            and self.waste < 25.0
        ):
            al_loss += _AL_PASSIVE_DRAIN_EVOLVED
        self.allostatic_load = max(0.0, min(100.0, self.allostatic_load + al_gain - al_loss))

        # Allostatic load feeds back into metabolism — stress leaves a residue.
        # Coefficient reduced 0.08→0.05 to soften the AL↔heat positive-feedback
        # loop; the organism can still spiral to thermal death if it ignores
        # sustained high waste/AL, but a competent manager can break the cycle.
        _al_ratio = self.allostatic_load / 100.0
        self.heat += 0.05 * _al_ratio * compute_load
        self.waste += 0.02 * _al_ratio * compute_load

        # ---- Reticular arousal gate -------------------------------------- #
        # High surprise (FE > 60) triggers a global arousal spike that boosts
        # attention precision (handled by PrecisionEngine) but also generates
        # extra heat — sustained arousal risks ThermalDeath.
        if fe_after > 60.0:
            arousal_overshoot = min(1.0, (fe_after - 60.0) / 40.0)
            self.heat += arousal_overshoot * 0.4 * compute_load
        # ------------------------------------------------------------------ #

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

        # Mild passive cooling — baseline thermal radiation applied *after* death
        # checks so it cannot rescue a terminal heat spike but does prevent slow
        # runaway accumulation across long DECIDE streaks.
        self.heat = max(0.0, self.heat - _PASSIVE_COOL_RATE * compute_load)

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

        # Anticipatory REST (Phase 3b) — proactively rest when allostatic load is
        # building but hasn't hit crisis level yet.  REST dumps heat AND waste via
        # the Janitor, directly breaking the AL↔heat feedback loop.  This fires
        # BEFORE the REPAIR trigger so the organism cools down rather than doing
        # repair (which adds heat/waste) when the real problem is thermal buildup.
        if self.allostatic_load > REST_ALLOSTATIC_THRESHOLD:
            return "REST"

        # Anticipatory REPAIR (Phase 3) — act early based on projected trajectory
        # or accumulated allostatic load, not just current crisis.
        if self.allostatic_load > REPAIR_ALLOSTATIC_THRESHOLD:
            return "REPAIR"
        proj = self._project_vitals(horizon=5)
        if (
            proj["integrity"] < REPAIR_PROJECTED_INTEGRITY
            or proj["stability"] < REPAIR_PROJECTED_STABILITY
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

    def _project_vitals(self, horizon: int = 5) -> dict[str, float]:
        """Fast linear projection of vital signs over ``horizon`` ticks.

        Applies the same passive decay rates used in ``tick()`` iteratively,
        incorporating the current allostatic load feedback.  Used by tick()
        for anticipatory REPAIR triggering (Phase 3).

        Returns a dict of the *minimum* (worst-case) projected value for each
        vital over the horizon, which is the relevant measure for safety.
        """
        e = self.energy
        h = self.heat
        w = self.waste
        m = self.integrity
        s = self.stability
        al_ratio = self.allostatic_load / 100.0

        min_integrity = m
        min_stability = s

        for _ in range(horizon):
            w = w + 0.018 + 0.02 * al_ratio
            _wf = 1.0 + w / 60.0
            if w > 60.0:
                _wf += ((w - 60.0) / 40.0) ** 2
            h = h + 0.1 * _wf + 0.08 * al_ratio - _PASSIVE_COOL_RATE
            h = max(0.0, h)
            m = m * (1.0 - (h / 120.0) * 0.01)
            s = s - 0.05
            e = e - 0.12
            if m < min_integrity:
                min_integrity = m
            if s < min_stability:
                min_stability = s

        return {
            "energy": e,
            "heat": h,
            "waste": w,
            "integrity": min_integrity,
            "stability": min_stability,
        }

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
        # Strip private fields (not persisted) and any unknown keys from old
        # state files that are no longer valid fields, while tolerating missing
        # new fields by falling through to dataclass defaults.
        known = set(cls.__dataclass_fields__.keys())
        filtered = {
            k: v for k, v in data.items()
            if not k.startswith("_") and k in known
        }
        return cls(**filtered)

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
