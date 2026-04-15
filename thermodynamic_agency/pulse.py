"""GhostMesh main pulse — the organism's central heartbeat.

This module ties together the metabolic spine, cognition subsystems, and
memory diary into one runnable loop.  Invoked directly::

    python -m thermodynamic_agency.pulse

Or imported for programmatic control::

    from thermodynamic_agency.pulse import GhostMesh
    mesh = GhostMesh()
    mesh.run()
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.core.exceptions import GhostDeathException
from thermodynamic_agency.core.environment import sample_event
from thermodynamic_agency.cognition.inference import (
    active_inference_step,
    generate_default_proposals,
    ForwardModel,
)
from thermodynamic_agency.cognition.ethics import EthicalEngine
from thermodynamic_agency.cognition.janitor import Janitor
from thermodynamic_agency.cognition.surgeon import Surgeon
from thermodynamic_agency.cognition.personality import MaskRotator
from thermodynamic_agency.cognition.precision import PrecisionEngine
from thermodynamic_agency.cognition.environment import EnvironmentStressor
from thermodynamic_agency.cognition.limbic import LimbicLayer
from thermodynamic_agency.cognition.predictive_hierarchy import PredictiveHierarchy
from thermodynamic_agency.cognition.thalamus import ThalamusGate
from thermodynamic_agency.cognition.basal_ganglia import BasalGanglia
from thermodynamic_agency.memory.diary import RamDiary, DiaryEntry
from thermodynamic_agency.interface.hud import print_hud
from thermodynamic_agency.run_logger import RunLogger, TickRecord

STATE_FILE = os.environ.get("GHOST_STATE_FILE", "/dev/shm/ghost_metabolic.json")
DIARY_PATH = os.environ.get("GHOST_DIARY_PATH", "/dev/shm/ghost_diary.db")
PULSE_SECONDS = float(os.environ.get("GHOST_PULSE", "5"))
COMPUTE_LOAD = float(os.environ.get("GHOST_COMPUTE_LOAD", "1.0"))
SHOW_HUD = os.environ.get("GHOST_HUD", "1") == "1"
# Optional path for per-tick vitals JSONL log (empty string = disabled)
VITALS_LOG = os.environ.get("GHOST_VITALS_LOG", "")
# Whether to inject stochastic environmental events (default: enabled)
ENABLE_ENV_EVENTS = os.environ.get("GHOST_ENV_EVENTS", "1") == "1"


class GhostMesh:
    """The full thermodynamic organism."""

    def __init__(self, seed: int | None = None) -> None:
        # Re-read env vars at construction time so tests/callers can override them.
        self._state_file = os.environ.get("GHOST_STATE_FILE", STATE_FILE)
        self._diary_path = os.environ.get("GHOST_DIARY_PATH", DIARY_PATH)
        self._pulse_seconds = float(os.environ.get("GHOST_PULSE", str(PULSE_SECONDS)))
        self._compute_load = float(os.environ.get("GHOST_COMPUTE_LOAD", str(COMPUTE_LOAD)))
        self._show_hud = os.environ.get("GHOST_HUD", "1") == "1"
        self._vitals_log = os.environ.get("GHOST_VITALS_LOG", VITALS_LOG)
        self._env_events = os.environ.get("GHOST_ENV_EVENTS", "1") == "1"
        self._rng = random.Random(seed)  # seeded for reproducibility
        # Load or initialise metabolic state
        self.state = self._load_state()
        self.diary = RamDiary(path=self._diary_path)
        self.ethics = EthicalEngine()
        self.janitor = Janitor(diary=self.diary)
        self.surgeon = Surgeon(diary=self.diary)
        self.rotator = MaskRotator(initial_mask="Guardian")
        self.precision_engine = PrecisionEngine()
        self.limbic = LimbicLayer()
        self.forward_model = ForwardModel()
        # Phase 3: hierarchical predictive coding + thalamic routing + habit loops
        self.hierarchy = PredictiveHierarchy()
        self.thalamus = ThalamusGate()
        self.basal_ganglia = BasalGanglia()
        self._running = False
        # Vitals log file handle (opened lazily on first write)
        self._vitals_fh = None

        # Optional stochastic environment stressor
        _stressor_prob = float(os.environ.get("GHOST_STRESSOR_PROB", "0.0"))
        _stressor_intensity = float(os.environ.get("GHOST_STRESSOR_INTENSITY", "1.0"))
        _stressor_seed_str = os.environ.get("GHOST_STRESSOR_SEED", "")
        _stressor_seed = int(_stressor_seed_str) if _stressor_seed_str else None
        _stressor_mode = os.environ.get("GHOST_STRESSOR_MODE", "flat")
        self.stressor: EnvironmentStressor | None = (
            EnvironmentStressor(
                prob=_stressor_prob,
                intensity=_stressor_intensity,
                seed=_stressor_seed,
                mode=_stressor_mode,  # type: ignore[arg-type]
            )
            if _stressor_prob > 0.0
            else None
        )

        # Optional per-tick run logger (writes JSONL to GHOST_LOG_FILE if set)
        _log_file = os.environ.get("GHOST_LOG_FILE", "")
        self.run_logger = RunLogger(path=_log_file if _log_file else None)

        # Tracks precision regime set during the last DECIDE step for logging
        self._last_precision_regime: str = "dormant"

        # Register graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, max_ticks: int | None = None) -> None:
        """Start the main pulse loop.

        Parameters
        ----------
        max_ticks:
            If set, stop after this many ticks (useful for tests/demos).
        """
        self._running = True
        ticks = 0

        self.diary.append(
            DiaryEntry(
                tick=self.state.entropy,
                role="thought",
                content="GhostMesh awakening — pulse loop starting.",
            )
        )

        try:
            while self._running:
                if max_ticks is not None and ticks >= max_ticks:
                    break
                try:
                    self._pulse()
                except GhostDeathException as exc:
                    self._handle_death(exc)
                    break
                ticks += 1
                if self._running:
                    time.sleep(self._pulse_seconds)
        finally:
            self._close_vitals_log()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------ #
    # Single pulse                                                         #
    # ------------------------------------------------------------------ #

    def _pulse(self) -> None:
        """Execute one heartbeat."""
        # 1. Inject stochastic environmental event BEFORE metabolic tick
        #    so the organism must react to a world that surprises it.
        env_event = None
        if self._env_events:
            env_event = sample_event(rng=self._rng)
            if not env_event.is_null():
                self.state.apply_action_feedback(
                    delta_energy=env_event.delta_energy,
                    delta_heat=env_event.delta_heat,
                    delta_waste=env_event.delta_waste,
                    delta_integrity=env_event.delta_integrity,
                    delta_stability=env_event.delta_stability,
                )

        mask = self.rotator.active
        action = self.state.tick(compute_load=self._compute_load)
        self.rotator.tick(self.state.entropy)

        # 2. Limbic processing — amygdala threat detection + accumbens reward.
        #    This runs after tick() so it can read the just-updated affect signal.
        limbic_signal = self.limbic.process(self.state)
        if limbic_signal.heat_cost > 0 or limbic_signal.integrity_cost > 0:
            self.state.apply_action_feedback(
                delta_heat=-limbic_signal.heat_cost,
                delta_integrity=-limbic_signal.integrity_cost,
            )

        # 2a. Phase 3 — Thalamic gating: compute per-layer routing coefficients.
        gate_report = self.thalamus.route(
            self.state,
            limbic_signal=limbic_signal,
            precision_regime=self._last_precision_regime,
        )

        # 2b. Phase 3 — Hierarchical predictive coding update.
        #     Errors flow upward from brainstem→limbic→prefrontal.
        #     Predictions flow downward prefrontal→limbic.
        hierarchy_signal = self.hierarchy.update(
            self.state,
            limbic_signal=limbic_signal,
            l1_precision=gate_report.l1_precision,
            l2_precision=gate_report.l2_precision,
        )
        # Apply hierarchy heat cost (neural activation for error propagation)
        if hierarchy_signal.heat_cost > 0:
            self.state.apply_action_feedback(delta_heat=hierarchy_signal.heat_cost)

        # 2c. Phase 3 — Basal ganglia tick decay (habit erosion under stress)
        self.basal_ganglia.tick_decay(self.state)

        # 3. Record the current vitals as an episodic memory slot.
        self.limbic.push_episode(
            tick=self.state.entropy,
            content=(
                f"tick={self.state.entropy} action={action} "
                f"E={self.state.energy:.1f} T={self.state.heat:.1f} "
                f"affect={self.state.affect:.2f}"
            ),
            metadata={"action": action, "affect": self.state.affect},
        )

        # Apply stochastic environmental disturbance (if stressor is active)
        stressor_event = ""
        if self.stressor is not None:
            stressor_event = self.stressor.maybe_disturb(self.state)
            if stressor_event:
                self.diary.append(
                    DiaryEntry(
                        tick=self.state.entropy,
                        role="environment",
                        content=f"STRESSOR: {stressor_event}",
                    )
                )

        if self._show_hud:
            print_hud(self.state.to_dict(), self.rotator.status())

        # 4. Log per-tick vitals (if enabled)
        self._write_vitals_log(action, env_event)

        # Rotate mask based on action, affect, and limbic threat.
        # High amygdala threat → SalienceNet override.
        # Negative affect + stress → Guardian/Judge dominance.
        # Positive affect → DefaultMode / Dreamer exploration.
        affect = self.state.affect
        threat = limbic_signal.threat_level
        if affect < -0.4:
            self.rotator.maybe_rotate(
                self.state.entropy,
                metabolic_hint="REPAIR",
                affect=affect,
                threat_level=threat,
            )
        elif affect > 0.4:
            self.rotator.maybe_rotate(
                self.state.entropy,
                metabolic_hint="REST",
                affect=affect,
                threat_level=threat,
            )
        else:
            self.rotator.maybe_rotate(
                self.state.entropy,
                metabolic_hint=action,
                affect=affect,
                threat_level=threat,
            )

        # Dispatch action; _decide() returns ethics-blocks count
        ethics_blocks = 0
        # Snapshot vitals before action for habit outcome recording
        _vitals_before = {
            "energy": self.state.energy,
            "heat": self.state.heat,
            "waste": self.state.waste,
            "integrity": self.state.integrity,
            "stability": self.state.stability,
        }
        if action == "FORAGE":
            self._forage()
        elif action == "REST":
            self._rest()
        elif action == "REPAIR":
            self._repair()
        else:  # "DECIDE"
            ethics_blocks = self._decide(
                limbic_signal=limbic_signal,
                hierarchy_signal=hierarchy_signal,
                gate_report=gate_report,
            )

        # Phase 3 — Record habit outcome for the executed action token.
        # This lets the basal ganglia learn which metabolic actions reliably
        # reduce free energy so they can be cheaply re-executed as habits.
        _vitals_after = {
            "energy": self.state.energy,
            "heat": self.state.heat,
            "waste": self.state.waste,
            "integrity": self.state.integrity,
            "stability": self.state.stability,
        }
        self.basal_ganglia.record_outcome(action, _vitals_before, _vitals_after)

        # Update decide streak and last_action in state (Phase 4 + Phase 2)
        if action == "DECIDE":
            self.state.decide_streak += 1
        else:
            self.state.decide_streak = 0
        self.state.last_action = action

        # Update forward model with actual post-action vitals
        self.forward_model.update(self.state)

        # Log this tick
        self.run_logger.record(
            TickRecord(
                tick=self.state.entropy,
                action=action,
                mask=self.rotator.active.name,
                energy=self.state.energy,
                heat=self.state.heat,
                waste=self.state.waste,
                integrity=self.state.integrity,
                stability=self.state.stability,
                affect=self.state.affect,
                free_energy=self.state.free_energy_estimate(),
                precision_regime=self._last_precision_regime,
                health_score=self.state.health_score(),
                stage=self.state.stage,
                ethics_blocks=ethics_blocks,
                stressor_event=stressor_event,
                allostatic_load=self.state.allostatic_load,
                decide_streak=self.state.decide_streak,
            )
        )

        # Persist state
        self._save_state()

    # ------------------------------------------------------------------ #
    # Action dispatchers                                                   #
    # ------------------------------------------------------------------ #

    def _forage(self) -> None:
        """Hunt for resources — lightweight refill."""
        self.diary.append(
            DiaryEntry(
                tick=self.state.entropy,
                role="action",
                content="FORAGE: replenishing energy reserves.",
            )
        )
        # Simple resource injection (real deployment: call external tools)
        self.state.apply_action_feedback(
            delta_energy=20.0,
            delta_heat=2.0,
            delta_waste=1.0,
        )

    def _rest(self) -> None:
        """Run Janitor to compress context and cool heat/waste."""
        # Consolidate episodic memory during REST (reduces integrity overflow cost)
        flushed = self.limbic.consolidate(n=10)
        if flushed:
            self.diary.append(
                DiaryEntry(
                    tick=self.state.entropy,
                    role="thought",
                    content=f"LIMBIC: consolidated {len(flushed)} episodic slots.",
                )
            )
        report = self.janitor.run(self.state)
        self.diary.append(
            DiaryEntry(
                tick=self.state.entropy,
                role="action",
                content=(
                    f"REST/Janitor: compressed {report.entries_compressed} entries "
                    f"into {report.insights_generated} insights. "
                    f"ΔW={report.delta_waste:.1f} ΔT={report.delta_heat:.1f}"
                ),
            )
        )

    def _repair(self) -> None:
        """Run Surgeon to restore integrity and stability."""
        report = self.surgeon.run(self.state)
        self.diary.append(
            DiaryEntry(
                tick=self.state.entropy,
                role="repair",
                content=(
                    f"REPAIR/Surgeon: {report.diagnosis}. "
                    f"ΔM=+{report.integrity_gain:.1f} ΔS=+{report.stability_gain:.1f} "
                    f"allostatic_cost={report.allostatic_cost:.2f}"
                ),
            )
        )

    def _decide(self, limbic_signal=None, hierarchy_signal=None, gate_report=None) -> int:
        """Full active-inference planning cycle with thermodynamic precision cost.

        Parameters
        ----------
        limbic_signal:
            Optional ``LimbicSignal`` from the current tick's limbic processing.
            Used to fold amygdala precision overrides and accumbens reward
            discount into the inference step.
        hierarchy_signal:
            Optional ``HierarchySignal`` from Phase 3 predictive hierarchy.
            Contributes top-down precision modulation and a hierarchical EFE
            penalty reflecting how badly the model mis-predicted this tick.
        gate_report:
            Optional ``GateReport`` from the thalamic precision router.
            Channel weights are merged into the final precision_weights dict.

        Returns
        -------
        int
            Number of proposals blocked by the ethics gate this cycle.
        """
        # Tune precision weights based on current metabolic state.
        # The PrecisionEngine applies its own metabolic cost for sharpening attention.
        precision_report = self.precision_engine.tune(
            self.state, compute_load=self._compute_load
        )
        # Store regime for the run logger
        self._last_precision_regime = precision_report.regime
        # Apply the precision-sharpening metabolic cost
        if precision_report.energy_cost > 0 or precision_report.heat_cost > 0:
            self.state.apply_action_feedback(
                delta_energy=-precision_report.energy_cost,
                delta_heat=precision_report.heat_cost,
            )

        # Merge amygdala precision overrides into base weights
        precision_weights = dict(precision_report.weights)
        if limbic_signal and limbic_signal.precision_overrides:
            for vital, boost in limbic_signal.precision_overrides.items():
                if vital in precision_weights:
                    precision_weights[vital] = min(6.0, precision_weights[vital] + boost)

        # Phase 3 — Merge top-down precision from PredictiveHierarchy (L2→L1)
        # and thalamic channel weights into the precision dict.
        hier_efe_penalty = 0.0
        if hierarchy_signal is not None:
            for vital, td_prec in hierarchy_signal.top_down_precision.items():
                if vital in precision_weights:
                    # Blend: thalamic channel weight gates how much top-down signal flows
                    channel_w = gate_report.channel_weights.get(vital, 1.0) if gate_report else 1.0
                    precision_weights[vital] = min(
                        6.0,
                        precision_weights[vital] * (1.0 + 0.2 * (td_prec - 1.0) * channel_w),
                    )
            hier_efe_penalty = hierarchy_signal.hierarchical_error

        # Retrieve forward-model prediction error penalty (cerebellum layer)
        fm_penalty = self.forward_model.prediction_error_term()

        raw_proposals = generate_default_proposals(self.state)

        # Phase 3 — Basal ganglia habit check on highest-priority proposal.
        # If the first proposal (idle / most conservative) is a known habit and
        # stress is low, we can skip the full EFE computation for it and return
        # a cheap cached estimate.  We still run ethics screening.
        habit_note = ""
        first_proposal = raw_proposals[0] if raw_proposals else None
        first_proposal_name = first_proposal.name if first_proposal else ""
        bg_signal = self.basal_ganglia.consult(first_proposal_name, self.state)
        use_habit = bg_signal.is_habit

        # Ethics immune screening — count how many proposals are blocked
        safe_proposals = self.ethics.immune_scan(raw_proposals, self.state)
        ethics_blocks = len(raw_proposals) - len(safe_proposals)
        if not safe_proposals:
            # All proposals were blocked — fall back to the full set rather than
            # halting entirely.  This prevents the organism from deadlocking when
            # extreme metabolic state causes every proposal to trip a hard
            # invariant.  The tick() death-threshold checks still apply, so
            # truly lethal situations still raise a GhostDeathException.
            safe_proposals = raw_proposals
            use_habit = False  # habit bypass disabled when all proposals were blocked

        # EFE discount from nucleus accumbens (positive affect reward signal)
        reward_discount = limbic_signal.efe_discount if limbic_signal else 0.0

        # If ethics blocked the habit-candidate proposal, fall back to full planning.
        if use_habit and (first_proposal is None or first_proposal not in safe_proposals):
            use_habit = False

        if use_habit:
            # ---- Cheap habit execution path (basal ganglia bypass) --------
            # Apply the cheap habit metabolic cost instead of full inference cost
            self.state.apply_action_feedback(
                delta_energy=-bg_signal.energy_cost,
                delta_heat=bg_signal.heat_cost,
            )
            selected = first_proposal  # type: ignore[assignment]  # guarded by use_habit check above
            efe_scores = {p.name: bg_signal.estimated_efe for p in safe_proposals}
            efe_scores[selected.name] = bg_signal.estimated_efe
            habit_note = f" [habit: {bg_signal.reason}]"
            cost_str = (
                f"habit_cost=[E={bg_signal.energy_cost:.3f} "
                f"H={bg_signal.heat_cost:.3f} strength={bg_signal.strength:.2f}]"
            )
            fm_str = f" fm_penalty={fm_penalty:.3f}" if fm_penalty > 0 else ""
            hier_str = (
                f" hier_efe_penalty={hier_efe_penalty:.3f}"
                if hier_efe_penalty > 0 else ""
            )
            self.diary.append(
                DiaryEntry(
                    tick=self.state.entropy,
                    role="thought",
                    content=(
                        f"DECIDE[habit]: selected '{selected.name}' "
                        f"(EFE≈{bg_signal.estimated_efe:.2f}). "
                        f"{cost_str}{fm_str}{hier_str}"
                    ),
                    metadata={
                        "mask": self.rotator.active.name,
                        "precision_regime": precision_report.regime,
                        "affect": self.state.affect,
                        "free_energy": precision_report.free_energy,
                        "ethics_blocks": ethics_blocks,
                        "threat_level": limbic_signal.threat_level if limbic_signal else 0.0,
                        "reward_discount": reward_discount,
                        "fm_penalty": fm_penalty,
                        "habit": True,
                        "hier_efe_penalty": hier_efe_penalty,
                    },
                )
            )
        else:
            # ---- Full active-inference planning path ----------------------
            # active_inference_step charges the cognitive cost of evaluating proposals
            result = active_inference_step(
                self.state,
                safe_proposals,
                precision_weights=precision_weights,
                compute_load=self._compute_load,
                reward_discount=reward_discount,
            )
            selected = result.selected

            # Add hierarchical EFE penalty to each proposal's score
            if hier_efe_penalty > 0:
                for k in result.efe_scores:
                    result.efe_scores[k] += hier_efe_penalty
            efe_scores = result.efe_scores

            cost = result.inference_cost
            cost_str = (
                f"inference_cost=[E={cost.energy_cost:.3f} H={cost.heat_cost:.3f} "
                f"KL={cost.kl_complexity:.1f} prec={cost.precision_used:.2f}]"
                if cost else ""
            )
            fm_str = f" fm_penalty={fm_penalty:.3f}" if fm_penalty > 0 else ""
            hier_str = (
                f" hier_efe_penalty={hier_efe_penalty:.3f}"
                if hier_efe_penalty > 0 else ""
            )

            self.diary.append(
                DiaryEntry(
                    tick=self.state.entropy,
                    role="thought",
                    content=(
                        f"DECIDE: selected '{selected.name}' "
                        f"(EFE={efe_scores[selected.name]:.2f}). "
                        f"{result.reasoning} {cost_str}{fm_str}{hier_str}"
                    ),
                    metadata={
                        "mask": self.rotator.active.name,
                        "precision_regime": precision_report.regime,
                        "affect": self.state.affect,
                        "free_energy": precision_report.free_energy,
                        "ethics_blocks": ethics_blocks,
                        "threat_level": limbic_signal.threat_level if limbic_signal else 0.0,
                        "reward_discount": reward_discount,
                        "fm_penalty": fm_penalty,
                        "habit": False,
                        "hier_efe_penalty": hier_efe_penalty,
                    },
                )
            )

        # Apply feedback from the selected action
        delta = selected.predicted_delta
        self.state.apply_action_feedback(
            delta_energy=delta.get("energy", 0.0),
            delta_heat=delta.get("heat", 0.0),
            delta_waste=delta.get("waste", 0.0),
            delta_integrity=delta.get("integrity", 0.0),
            delta_stability=delta.get("stability", 0.0),
        )

        return ethics_blocks

    # ------------------------------------------------------------------ #
    # Death handler                                                        #
    # ------------------------------------------------------------------ #

    def _handle_death(self, exc: GhostDeathException) -> None:
        cause = type(exc).__name__
        self.diary.append(
            DiaryEntry(
                tick=self.state.entropy,
                role="error",
                content=f"DEATH — {cause}: {exc}",
            )
        )
        self._save_state()
        self.run_logger.close()
        print(f"\n[GhostMesh] *** {cause} *** — organism terminated.", file=sys.stderr)
        print(f"  Last state: {json.dumps(exc.state, indent=2)}", file=sys.stderr)

    # ------------------------------------------------------------------ #
    # State persistence                                                    #
    # ------------------------------------------------------------------ #

    def _save_state(self) -> None:
        with open(self._state_file, "w") as fh:
            fh.write(self.state.to_json())

    def _load_state(self) -> MetabolicState:
        state_file = os.environ.get("GHOST_STATE_FILE", STATE_FILE)
        if os.path.exists(state_file):
            with open(state_file) as fh:
                data = json.load(fh)
            return MetabolicState.from_dict(data)
        return MetabolicState()

    # ------------------------------------------------------------------ #
    # Vitals logging                                                        #
    # ------------------------------------------------------------------ #

    def _write_vitals_log(self, action: str, env_event) -> None:
        """Append one JSON-lines record to the vitals log (if enabled)."""
        if not self._vitals_log:
            return
        if self._vitals_fh is None:
            self._vitals_fh = open(self._vitals_log, "a", buffering=1)  # line-buffered
        s = self.state
        # Use 'none' for null events so every record has a meaningful name
        event_name = (
            env_event.name
            if env_event is not None and not env_event.is_null()
            else "none"
        )
        record = {
            "tick": s.entropy,
            "energy": round(s.energy, 3),
            "heat": round(s.heat, 3),
            "waste": round(s.waste, 3),
            "integrity": round(s.integrity, 3),
            "stability": round(s.stability, 3),
            "affect": round(s.affect, 4),
            "free_energy": round(s.free_energy_estimate(), 3),
            "health": round(s.health_score(), 3),
            "stage": s.stage,
            "action": action,
            "mask": self.rotator.active.name,
            "env_event": event_name,
            "compute_load": self._compute_load,
            "allostatic_load": round(s.allostatic_load, 3),
            "decide_streak": s.decide_streak,
        }
        self._vitals_fh.write(json.dumps(record) + "\n")

    def _close_vitals_log(self) -> None:
        if self._vitals_fh is not None:
            self._vitals_fh.close()
            self._vitals_fh = None

    # ------------------------------------------------------------------ #
    # Signal handler                                                       #
    # ------------------------------------------------------------------ #

    def _handle_signal(self, signum: int, frame: object) -> None:
        print("\n[GhostMesh] Signal received — shutting down gracefully.", file=sys.stderr)
        self._running = False
        self.run_logger.close()


# ------------------------------------------------------------------ #
# Module entry-point                                                   #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    mesh = GhostMesh()
    mesh.run()
