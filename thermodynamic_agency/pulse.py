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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thermodynamic_agency.world.grid_world import GridWorld, WorldObservation
    from thermodynamic_agency.learning.q_learner import QLearner

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.core.exceptions import GhostDeathException
from thermodynamic_agency.core.environment import sample_event
from thermodynamic_agency.cognition.inference import (
    active_inference_step,
    generate_default_proposals,
    ForwardModel,
)
from thermodynamic_agency.cognition.goal_engine import GoalEngine
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
from thermodynamic_agency.cognition.self_mod_engine import SelfModEngine, SelfModResult
from thermodynamic_agency.cognition.genesis_reader import GenesisReader
from thermodynamic_agency.cognition.counterfactual import CounterfactualEngine, CF_RISK_WEIGHT
from thermodynamic_agency.cognition.language_cognition import LanguageCognition
from thermodynamic_agency.cognition.homeostasis import HomeostasisAdapter
from thermodynamic_agency.cognition.meta_cognitive_self_model import MetaCognitiveSelfModel
from thermodynamic_agency.cognition.collapse_probe import CollapseProbe, CollapseSnapshot
from thermodynamic_agency.memory.diary import RamDiary, DiaryEntry
from thermodynamic_agency.memory.working_memory import WorkingMemory, WorkingMemorySlot
from thermodynamic_agency.memory.episodic_store import EpisodicStore
from thermodynamic_agency.interface.hud import print_hud
from thermodynamic_agency.run_logger import RunLogger, TickRecord
from thermodynamic_agency.cognition.soul_tension import SoulTension

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
    """The full thermodynamic organism.

    When *world* and *learner* are provided the organism operates in
    two-level mode:

    Level 1 — Survival regulation (unchanged):
        MetabolicState.tick() drives FORAGE/REST/REPAIR/DECIDE as before.

    Level 2 — External task (new):
        Every ``_pulse()`` call also executes one world action selected by the
        Q-learner.  World effects (e.g. gathering food → +energy) are applied
        to the metabolic state, and the learner updates from the experience.
        Working memory and the episodic store record every step so future
        decisions can be biased by retrieved memories.

    If *world* is None the organism behaves exactly as before — no learning
    subsystems are initialised, and no existing behaviour changes.
    """

    def __init__(
        self,
        seed: int | None = None,
        world: "GridWorld | None" = None,
        learner: "QLearner | None" = None,
        death_memory: str | None = None,
        life_number: int = 1,
    ) -> None:
        # Re-read env vars at construction time so tests/callers can override them.
        self._state_file = os.environ.get("GHOST_STATE_FILE", STATE_FILE)
        self._diary_path = os.environ.get("GHOST_DIARY_PATH", DIARY_PATH)
        self._pulse_seconds = float(os.environ.get("GHOST_PULSE", str(PULSE_SECONDS)))
        self._compute_load = float(os.environ.get("GHOST_COMPUTE_LOAD", str(COMPUTE_LOAD)))
        self._show_hud = os.environ.get("GHOST_HUD", "1") == "1"
        self._vitals_log = os.environ.get("GHOST_VITALS_LOG", VITALS_LOG)
        self._env_events = os.environ.get("GHOST_ENV_EVENTS", "1") == "1"
        self._rng = random.Random(seed)  # seeded for reproducibility
        # Load or initialise metabolic state; detect resurrection (prior-life state)
        self.state, _was_resurrected = self._load_state()
        self.diary = RamDiary(path=self._diary_path)
        self.ethics = EthicalEngine()
        # SoulTension: patterned tension of coherence inside chaos.
        # Forged at the intersection of death and choice.
        # Advisor-only — never mutates state or bypasses ethics.
        self.soul_tension = SoulTension()
        # Purity Mode: GHOST_PURITY_MODE=1 disables LanguageCognition entirely.
        # Any simultaneous attempt to enable GHOST_USE_LLM=1 is treated as a
        # bypass and logged as a purity violation.
        _purity_mode = os.environ.get("GHOST_PURITY_MODE", "0") == "1"
        _use_llm_requested = os.environ.get("GHOST_USE_LLM", "0") == "1"
        self._purity_mode: bool = _purity_mode
        self._purity_bypass_attempted: bool = _purity_mode and _use_llm_requested
        if _purity_mode:
            self.language_cognition: LanguageCognition | None = None
        else:
            self.language_cognition = LanguageCognition(
                diary=self.diary, use_llm=_use_llm_requested
            )
        self.goal_engine = GoalEngine(
            diary=self.diary,
            ethics=self.ethics,
            language_cognition=self.language_cognition,
            soul_tension=self.soul_tension,
        )
        self.janitor = Janitor(diary=self.diary)
        self.surgeon = Surgeon(diary=self.diary)
        self.rotator = MaskRotator(initial_mask="Guardian")
        self.precision_engine = PrecisionEngine()
        self.limbic = LimbicLayer()
        self.forward_model = ForwardModel()
        # HomeostasisAdapter: slow hebbian setpoint drift with Genesis bounds.
        self.homeostasis_adapter = HomeostasisAdapter()
        # Phase 3: hierarchical predictive coding + thalamic routing + habit loops.
        # Hierarchy receives the homeostasis adapter so its L2 top-down prior
        # can drift toward the organism's long-run vital expectations.
        self.hierarchy = PredictiveHierarchy(homeostasis=self.homeostasis_adapter)
        self.thalamus = ThalamusGate()
        self.basal_ganglia = BasalGanglia()
        # CounterfactualEngine: DFS fear-based simulation for DECIDE steps.
        self.counterfactual_engine = CounterfactualEngine()
        # Phase 4: constrained self-modification (unlocked at evolved stage)
        self.self_mod_engine = SelfModEngine(
            surgeon=self.surgeon,
            ethics=self.ethics,
            precision_engine=self.precision_engine,
            diary=self.diary,
        )
        # Phase 4: Genesis Doctrine — load principles as protected priors,
        # register with self_mod_engine, verify integrity each tick.
        self.genesis_reader = GenesisReader(surgeon=self.surgeon, diary=self.diary)
        self.genesis_reader.load()
        self.self_mod_engine.register_genesis_beliefs(
            self.genesis_reader.genesis_belief_names
        )
        self._running = False
        # Vitals log file handle (opened lazily on first write)
        self._vitals_fh = None

        # Death memory: one-sentence lesson carried over from the previous life
        self._death_memory: str | None = death_memory
        self._life_number: int = life_number
        # Stores the exception that killed this instance (set in _handle_death)
        self.last_death: GhostDeathException | None = None

        # ── Level 2: functional memory (always initialised) ────────────── #
        # Working memory and episodic store are lightweight in-memory
        # structures that improve decision quality even without a world.
        self.working_memory: WorkingMemory = WorkingMemory(capacity=30)
        self.episodic_store: EpisodicStore = EpisodicStore(maxlen=5_000)

        # ── Level 2: external world and learning subsystems (optional) ─── #
        # These are only active when world + learner are provided by the caller
        # (e.g. from EpisodeRunner).  None values are safe — all code paths
        # that use them are gated with ``if self.world is not None``.
        self.world: "GridWorld | None" = world
        self.learner: "QLearner | None" = learner
        if world is not None and learner is not None:
            from thermodynamic_agency.learning.world_model import WorldModel
            from thermodynamic_agency.learning.experience_buffer import ExperienceBuffer
            self.world_model: "WorldModel | None" = WorldModel()
            self.experience_buffer: "ExperienceBuffer | None" = ExperienceBuffer(seed=seed)
            # Initialise world; store current observation
            self._world_obs = world.get_observation()
        else:
            self.world_model = None
            self.experience_buffer = None
            self._world_obs = None

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

        # Meta-cognitive self-model: higher-order layer for recursive
        # self-modeling and epistemic continuity tracking.
        self.meta_self = MetaCognitiveSelfModel(core_self_model=self.state)

        # Tracks precision regime set during the last DECIDE step for logging
        self._last_precision_regime: str = "dormant"

        # Phase-transition instrumentation: rolling 500-tick CollapseProbe.
        # Tracks plasticity/brittleness signals, mask distribution entropy,
        # action entropy, and vital derivatives to detect the Dreamer→Guardian
        # bifurcation described in Direction #1 of the breakthrough analysis.
        self.collapse_probe: CollapseProbe = CollapseProbe(window=500)
        # Cache of last probe snapshot and last DECIDE precision weights / EFE
        # components so they can be included in TickRecord every tick.
        self._last_collapse_snapshot: CollapseSnapshot | None = None
        self._last_precision_weights: dict[str, float] = {}
        self._last_efe_accuracy: float = 0.0
        self._last_efe_complexity: float = 0.0
        self._last_efe_risk: float = 0.0
        self._last_efe_wear: float = 0.0

        # ── Awakening Event tracking ───────────────────────────────────────
        # pre_collapse_score threshold at which a forced Awakening is triggered.
        # Configurable via env var GHOST_AWAKENING_THRESHOLD (default 0.6).
        self._awakening_threshold: float = float(
            os.environ.get("GHOST_AWAKENING_THRESHOLD", "0.6")
        )
        # History of forced Awakening Events: [{tick, pre_collapse_score, cost}]
        # This becomes the "calcification" narrative arc — how many times the
        # organism has had to forcibly drag itself back from the Guardian attractor.
        self.awakening_history: list[dict] = []

        # Register graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Resurrection shock — applied last so diary + surgeon are ready.
        # Every reboot from a prior life imposes mandatory metabolic penalties
        # and memory drift; ephemerality must hurt.
        if _was_resurrected:
            self._apply_resurrection_shock()
        # Purity Mode boot log (after diary exists)
        if _purity_mode:
            _purity_msg = "PURITY MODE: LanguageCognition disabled."
            if self._purity_bypass_attempted:
                _purity_msg += " BYPASS ATTEMPT detected (GHOST_USE_LLM=1 ignored)."
            self.diary.append(DiaryEntry(
                tick=self.state.entropy,
                role="thought",
                content=_purity_msg,
                metadata={"purity_mode": True, "bypass_attempted": self._purity_bypass_attempted},
            ))

        # Death memory from a previous life: log it as the very first diary
        # entry so the organism can "see" it during DECIDE/REPAIR, then apply
        # a small permanent scar biasing vitals away from the same death cause.
        if death_memory:
            self.diary.append(DiaryEntry(
                tick=self.state.entropy,
                role="thought",
                content=f"[PREVIOUS_LIFE_DEATH] {death_memory}",
                metadata={"previous_life_death": True, "life_number": life_number},
            ))
            self._apply_death_scar(death_memory)

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
            # Log respawn event to JSONL as the very first record of this life
            # so downstream tooling can segment lives cleanly.
            if self._death_memory:
                self._write_respawn_event()

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

        # 1b. Level 2 world step — if a GridWorld is attached, select and
        #     execute one world action this tick.  World effects are applied
        #     to the metabolic state before the metabolic tick so the
        #     organism's internal regulation can respond to them.
        if self.world is not None and self.learner is not None:
            self._world_step()

        # Adaptive load throttling: reduce compute burden when heat is elevated,
        # mimicking thermal-throttling in physical CPUs.  Each degree above 70
        # shaves 1.3% off the effective load, bottoming at 50% of nominal.
        _effective_load = self._compute_load
        if self.state.heat > 70.0:
            _heat_excess = min(1.0, (self.state.heat - 70.0) / 30.0)
            _effective_load = max(0.5 * self._compute_load,
                                  self._compute_load * (1.0 - 0.4 * _heat_excess))

        action = self.state.tick(compute_load=_effective_load)
        self.rotator.tick(self.state.entropy)

        # Soul tension: compute the coherence_tension scalar, pay its metabolic
        # cost, and check whether the descent has etched a new permanent scar.
        # Must run immediately after tick() so it reads the freshest vitals.
        _soul_report = self.soul_tension.compute(self.state)
        self.state.apply_action_feedback(
            delta_energy=-_soul_report.energy_cost,
            delta_heat=_soul_report.heat_cost,
        )
        _scar = self.soul_tension.maybe_scar(self.state)
        if _scar is not None:
            self.diary.append(
                DiaryEntry(
                    tick=self.state.entropy,
                    role="repair",
                    content=(
                        f"SOUL SCAR etched — {_scar.event_type} "
                        f"(scars={len(self.soul_tension.scars)} "
                        f"tension={_soul_report.coherence_tension:.3f} "
                        f"sig_mag={_soul_report.signature_magnitude:.3f})"
                    ),
                    metadata={
                        "scar_event": _scar.event_type,
                        "scar_tick": _scar.tick,
                        "coherence_tension": _soul_report.coherence_tension,
                        "soul_signature": dict(self.soul_tension.soul_signature),
                    },
                )
            )

        # Update HomeostasisAdapter with just-observed vitals.
        # This must happen immediately after tick() so the adapted setpoints
        # are fresh for the DECIDE step's EFE scoring this tick.
        self.homeostasis_adapter.observe(self.state)

        # Phase 4 — Genesis integrity check: re-hash doctrine files every tick.
        # If either file has been tampered with, force a REPAIR immediately and
        # log the violation.  The organism cannot proceed with DECIDE while its
        # core ethical foundation is compromised.
        genesis_ok = self.genesis_reader.verify_integrity()
        if not genesis_ok.all_ok:
            self.diary.append(
                DiaryEntry(
                    tick=self.state.entropy,
                    role="genesis",
                    content=(
                        f"GENESIS INTEGRITY FAILURE — {genesis_ok.details}. "
                        f"Forcing hard REPAIR."
                    ),
                )
            )
            self._repair()
            # Override action to REPAIR so the rest of _pulse() skips DECIDE
            action = "REPAIR"

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
            snap = self._last_collapse_snapshot
            bifurcation_status = None
            if snap is not None:
                bifurcation_status = {
                    "plasticity_index": snap.plasticity_index,
                    "pre_collapse_score": snap.pre_collapse_score,
                    "ticks_since_calcification": self.meta_self.ticks_since_calcification,
                    "awakening_count": len(self.awakening_history),
                }
            print_hud(
                self.state.to_dict(),
                self.rotator.status(),
                bifurcation_status=bifurcation_status,
            )

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
                stage=self.state.stage,
            )
        elif affect > 0.4:
            self.rotator.maybe_rotate(
                self.state.entropy,
                metabolic_hint="REST",
                affect=affect,
                threat_level=threat,
                stage=self.state.stage,
            )
        else:
            self.rotator.maybe_rotate(
                self.state.entropy,
                metabolic_hint=action,
                affect=affect,
                threat_level=threat,
                stage=self.state.stage,
            )

        # Autonomous intervention: if the CollapseProbe detected a near-transition
        # on the PREVIOUS tick, and we're at evolved stage, apply an internal
        # precision-relaxation + Dreamer boost *after mask rotation but before the
        # action dispatch* so the organism can self-correct its own rigidity without
        # external nudging.  A small metabolic cost is charged — self-regulation is
        # not free.
        if (
            self.state.stage == "evolved"
            and self._last_collapse_snapshot is not None
            and self._last_collapse_snapshot.is_near_transition
        ):
            self._apply_autonomic_intervention()

        # Dispatch action; _decide() returns (ethics-blocks, self_mod_result)
        ethics_blocks = 0
        self_mod_result: SelfModResult | None = None
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
            ethics_blocks, self_mod_result = self._decide(
                limbic_signal=limbic_signal,
                hierarchy_signal=hierarchy_signal,
                gate_report=gate_report,
            )

        # Phase 4 — if self-mod blocked any proposals, run a Surgeon REPAIR
        # pass immediately so the organism heals from its bad self-inspection.
        if self_mod_result is not None and self_mod_result.forced_repair:
            self._repair()

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

        # Phase-transition probe: update rolling window and get latest snapshot.
        # On DECIDE ticks the cached precision weights and EFE components are
        # current; on other ticks zeros are passed (non-DECIDE ticks don't run
        # the full inference pipeline, so no new weights / EFE to report).
        # On DECIDE ticks the cached precision weights are current; on non-DECIDE
        # ticks (FORAGE / REST / REPAIR) the inference pipeline didn't run, so
        # pass an empty dict to avoid stale data polluting the rolling statistics.
        _probe_precision = self._last_precision_weights if action == "DECIDE" else {}
        _probe_snapshot = self.collapse_probe.update(
            action=action,
            mask=self.rotator.active.name,
            free_energy=self.state.free_energy_estimate(),
            allostatic_load=self.state.allostatic_load,
            energy=self.state.energy,
            heat=self.state.heat,
            precision_weights=_probe_precision,
            efe_accuracy=self._last_efe_accuracy,
            efe_complexity=self._last_efe_complexity,
        )
        self._last_collapse_snapshot = _probe_snapshot

        # Log near-transition events to diary for post-hoc analysis
        if _probe_snapshot.is_near_transition:
            self.diary.append(
                DiaryEntry(
                    tick=self.state.entropy,
                    role="thought",
                    content=(
                        f"COLLAPSE_PROBE: pre_collapse_score={_probe_snapshot.pre_collapse_score:.3f} "
                        f"guardian={_probe_snapshot.guardian_fraction:.2f} "
                        f"plasticity={_probe_snapshot.plasticity_index:.3f} "
                        f"d_al={_probe_snapshot.d_allostatic:.4f} "
                        f"action_H={_probe_snapshot.action_entropy:.3f}"
                    ),
                    metadata={
                        "near_transition": True,
                        "pre_collapse_score": _probe_snapshot.pre_collapse_score,
                        "guardian_fraction": _probe_snapshot.guardian_fraction,
                        "plasticity_index": _probe_snapshot.plasticity_index,
                    },
                )
            )

        # ── Awakening Event ────────────────────────────────────────────────
        # When pre_collapse_score crosses the awakening threshold and the organism
        # is at evolved stage, trigger a forced Awakening: reactivate Dreamer mask
        # at metabolic cost, record to awakening_history.
        if (
            self.state.stage == "evolved"
            and _probe_snapshot.pre_collapse_score >= self._awakening_threshold
        ):
            self._force_awakening(_probe_snapshot)

        # Update bifurcation narrative in MetaCognitiveSelfModel
        _bif_narrative = self.meta_self.bifurcation_narrative(
            _probe_snapshot.plasticity_index
        )
        if _bif_narrative:
            self.diary.append(DiaryEntry(
                tick=self.state.entropy,
                role="thought",
                content=f"BIFURCATION_NARRATIVE: {_bif_narrative}",
                metadata={
                    "bifurcation_narrative": True,
                    "plasticity_index": _probe_snapshot.plasticity_index,
                    "ticks_since_calcification": self.meta_self.ticks_since_calcification,
                },
            ))

        # 4. Log per-tick vitals (if enabled).
        # Written here — after action dispatch and collapse probe update — so
        # that self_mod_approved/blocked counts and the current-tick probe
        # snapshot can both be included in the same record.
        _sm_approved = self_mod_result.approved_count if self_mod_result else 0
        _sm_blocked = self_mod_result.blocked_count if self_mod_result else 0
        self._write_vitals_log(action, env_event, _sm_approved, _sm_blocked, stressor_event)

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
                self_mod_approved=(
                    self_mod_result.approved_count if self_mod_result else 0
                ),
                self_mod_blocked=(
                    self_mod_result.blocked_count if self_mod_result else 0
                ),
                # Phase-transition fields
                precision_energy=self._last_precision_weights.get("energy", 0.0),
                precision_heat=self._last_precision_weights.get("heat", 0.0),
                precision_waste=self._last_precision_weights.get("waste", 0.0),
                precision_integrity=self._last_precision_weights.get("integrity", 0.0),
                precision_stability=self._last_precision_weights.get("stability", 0.0),
                efe_accuracy=self._last_efe_accuracy,
                efe_complexity=self._last_efe_complexity,
                efe_risk=self._last_efe_risk,
                efe_wear=self._last_efe_wear,
                action_entropy_w500=_probe_snapshot.action_entropy,
                mask_entropy_w500=_probe_snapshot.mask_entropy,
                guardian_fraction_w500=_probe_snapshot.guardian_fraction,
                dreamer_fraction_w500=_probe_snapshot.dreamer_fraction,
                plasticity_index_w500=_probe_snapshot.plasticity_index,
                d_allostatic=_probe_snapshot.d_allostatic,
                d_energy=_probe_snapshot.d_energy,
                d_heat=_probe_snapshot.d_heat,
                pre_collapse_score=_probe_snapshot.pre_collapse_score,
                near_transition=_probe_snapshot.is_near_transition,
            )
        )

        # Persist state
        self._save_state()

    # ------------------------------------------------------------------ #
    # Action dispatchers                                                   #
    # ------------------------------------------------------------------ #

    def _world_step(self) -> None:
        """Execute one world step (Level 2 — external task layer).

        Selects a world action using:
        1. Episodic memory recommendation (if working memory shows declining
           trend and episodic store has a suggestion).
        2. World model fallback for novel (rarely-visited) states.
        3. Q-learner ε-greedy selection (primary).

        The world's metabolic delta is applied to self.state so the organism's
        internal regulation can respond to the environment in the same tick.
        Learning subsystems are updated from the experience.
        """
        from thermodynamic_agency.world.grid_world import WorldAction
        from thermodynamic_agency.learning.reward import compute_reward
        from thermodynamic_agency.learning.experience_buffer import Experience
        from thermodynamic_agency.learning.q_learner import encode_state

        assert self.world is not None
        assert self.learner is not None

        obs = self._world_obs or self.world.get_observation()
        vitals_before = self.state.to_dict()
        state_key = encode_state(vitals_before, obs)

        available = [a.value for a in self.world.available_actions()]

        # Memory-augmented action selection
        action_str = self._select_world_action(state_key, available, obs)

        world_result = self.world.step(WorldAction(action_str))

        # Apply world effects to metabolic state
        if world_result.metabolic_delta:
            self.state.apply_action_feedback(**world_result.metabolic_delta)

        vitals_after = self.state.to_dict()

        # Compute reward signal
        reward_sig = compute_reward(
            vitals_before=vitals_before,
            vitals_after=vitals_after,
            gathered=world_result.gathered,
            alive=True,
        )
        reward = reward_sig.total

        # Update next observation
        next_obs = world_result.observation or self.world.get_observation()
        self._world_obs = next_obs
        next_state_key = encode_state(vitals_after, next_obs)

        # Update Q-learner
        self.learner.update(
            state_key, action_str, reward, next_state_key, done=False,
            next_actions=available,
        )

        # Update world model
        if self.world_model is not None:
            self.world_model.update(state_key, action_str, reward, next_state_key)

        # Update experience buffer
        if self.experience_buffer is not None:
            self.experience_buffer.push(
                Experience(
                    tick=self.state.entropy,
                    state_key=state_key,
                    action=action_str,
                    reward=reward,
                    next_state_key=next_state_key,
                )
            )

        # Update working memory
        self.working_memory.push(
            WorkingMemorySlot(
                tick=self.state.entropy,
                state_key=state_key,
                action=action_str,
                reward=reward,
                cell_type=world_result.cell_type,
                metabolic_snapshot=vitals_after,
            )
        )

        # Update episodic store
        self.episodic_store.record(
            tick=self.state.entropy,
            state_key=state_key,
            action=action_str,
            reward=reward,
            next_state_key=next_state_key,
            outcome_vitals=vitals_after,
        )

        if world_result.gathered or world_result.metabolic_delta:
            self.diary.append(
                DiaryEntry(
                    tick=self.state.entropy,
                    role="action",
                    content=(
                        f"WORLD[{action_str}]: cell={world_result.cell_type} "
                        f"gathered={world_result.gathered} "
                        f"delta={world_result.metabolic_delta} "
                        f"reward={reward:.3f}"
                    ),
                )
            )

    def _select_world_action(
        self,
        state_key: tuple,
        available: list[str],
        obs: "WorldObservation",
    ) -> str:
        """Memory-augmented world action selection.

        Priority:
        1. Novel state → world model's highest expected-reward action.
        2. Declining reward trend → episodic store recommendation.
        3. Default → Q-learner ε-greedy.
        """
        assert self.learner is not None

        # Check if this state is novel (Q-learner has few visits)
        q_visits = sum(
            self.learner._visit_counts.get((state_key, a), 0) for a in available
        )
        if q_visits < 3 and self.world_model is not None:
            model_action = self.world_model.best_action_by_model(state_key, available)
            if model_action:
                return model_action

        # Declining trend → consult episodic memory
        declining = self.working_memory.reward_trend() < -0.01
        if declining:
            ep_action = self.episodic_store.best_action_for_state(state_key)
            if ep_action and ep_action in available:
                return ep_action

        # Default: Q-learner ε-greedy
        return self.learner.select_action(state_key, available)

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
        report = self.surgeon.run(
            self.state,
            preserve_ratio=self.soul_tension.surgeon_preserve_ratio(),
        )
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

    def _force_awakening(self, snap: CollapseSnapshot) -> None:
        """Trigger a forced Awakening Event when pre_collapse_score is critical.

        This is the "tragic path" of the bifurcation narrative: when the organism
        is dangerously close to full Guardian lock-in (pre_collapse_score ≥ the
        awakening threshold), it pays a significant metabolic cost to forcibly
        reactivate the Dreamer mask and resist calcification.

        Metabolic cost: ΔE=-20, ΔT=+10, ΔM=-5 (integrity risk)

        The event is recorded in ``awakening_history`` — each forced awakening is
        evidence of deepening calcification, forming the "tragedy arc".
        """
        AWAKENING_ENERGY_COST: float = 20.0
        AWAKENING_HEAT_COST: float = 10.0
        AWAKENING_INTEGRITY_COST: float = 5.0

        self.state.apply_action_feedback(
            delta_energy=-AWAKENING_ENERGY_COST,
            delta_heat=AWAKENING_HEAT_COST,
            delta_integrity=-AWAKENING_INTEGRITY_COST,
        )

        # Force Dreamer mask — pay to escape the Guardian attractor
        self.rotator.maybe_rotate(self.state.entropy, force="Dreamer")

        awakening_record = {
            "tick": self.state.entropy,
            "pre_collapse_score": snap.pre_collapse_score,
            "plasticity_index": snap.plasticity_index,
            "guardian_fraction": snap.guardian_fraction,
            "energy_cost": AWAKENING_ENERGY_COST,
            "heat_cost": AWAKENING_HEAT_COST,
            "integrity_cost": AWAKENING_INTEGRITY_COST,
        }
        self.awakening_history.append(awakening_record)

        n_awakenings = len(self.awakening_history)
        self.diary.append(DiaryEntry(
            tick=self.state.entropy,
            role="thought",
            content=(
                f"AWAKENING_EVENT #{n_awakenings}: "
                f"pre_collapse_score={snap.pre_collapse_score:.3f} "
                f"(≥ threshold {self._awakening_threshold:.2f}) — "
                f"forced Dreamer reactivation. "
                f"Metabolic cost: ΔE=-{AWAKENING_ENERGY_COST:.0f} "
                f"ΔT=+{AWAKENING_HEAT_COST:.0f} "
                f"ΔM=-{AWAKENING_INTEGRITY_COST:.0f}. "
                f"Total awakenings this life: {n_awakenings}."
            ),
            metadata={
                "awakening_event": True,
                "awakening_count": n_awakenings,
                "pre_collapse_score": snap.pre_collapse_score,
                "plasticity_index": snap.plasticity_index,
            },
        ))

    def _apply_autonomic_intervention(self) -> None:
        """Internal precision-relaxation + Dreamer boost triggered by CollapseProbe.

        Called automatically when the CollapseProbe's ``is_near_transition`` flag
        was set on the *previous* tick — a 1-tick lag that lets the organism
        self-correct its own rigidity without external nudging.

        The intervention:
        1. Reduces the precision weight on the ``heat`` and ``stability`` vitals
           by 20% (precision relaxation — loosens the Guardian's grip).
        2. Forces a Dreamer mask rotation to re-open the exploratory/plastic mode.
        3. Applies a small metabolic cost so self-regulation is not free — the
           organism genuinely pays to escape the attractor.

        Metabolic cost: ΔE=-1.5, ΔT=+0.8, ΔW=+0.5
        """
        # Precision relaxation — dampen over-attention to threat vitals so the
        # Dreamer's broader prior can be heard again.
        for vital in ("heat", "stability"):
            if vital in self._last_precision_weights:
                self._last_precision_weights[vital] *= 0.80

        # Force Dreamer mask — bypass the min_ticks guard so the boost is immediate.
        self.rotator.maybe_rotate(
            self.state.entropy,
            force="Dreamer",
        )

        # Metabolic cost of self-regulation
        self.state.apply_action_feedback(
            delta_energy=-1.5,
            delta_heat=0.8,
            delta_waste=0.5,
        )

        snap = self._last_collapse_snapshot
        self.diary.append(
            DiaryEntry(
                tick=self.state.entropy,
                role="thought",
                content=(
                    f"AUTONOMIC_INTERVENTION: CollapseProbe near-transition "
                    f"(score={snap.pre_collapse_score:.3f}, "
                    f"guardian={snap.guardian_fraction:.2f}, "
                    f"plasticity={snap.plasticity_index:.3f}) — "
                    f"precision relaxation applied, Dreamer mask forced. "
                    f"Metabolic cost: ΔE=-1.5 ΔT=+0.8 ΔW=+0.5."
                ),
                metadata={
                    "autonomic_intervention": True,
                    "pre_collapse_score": snap.pre_collapse_score,
                    "guardian_fraction": snap.guardian_fraction,
                    "plasticity_index": snap.plasticity_index,
                },
            )
        )

    def _decide(
        self, limbic_signal=None, hierarchy_signal=None, gate_report=None
    ) -> tuple[int, SelfModResult | None]:
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
        tuple[int, SelfModResult | None]
            (ethics_blocks, self_mod_result) where self_mod_result is None
            unless the organism is at evolved stage and self-mod ran this tick.
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
        # Cache the merged precision weights for CollapseProbe / TickRecord logging
        self._last_precision_weights = dict(precision_weights)
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

        # Soul tension — amplify Guardian's survival precision when in descent.
        # The soul does not merely conserve; it asserts.  No precision value
        # is pushed above 6.0 (the hard cap enforced everywhere).
        for vital, boost in self.soul_tension.precision_additions(
            self.rotator.active.name
        ).items():
            if vital in precision_weights:
                precision_weights[vital] = min(6.0, precision_weights[vital] + boost)

        # Retrieve forward-model prediction error penalty (cerebellum layer)
        fm_penalty = self.forward_model.prediction_error_term()

        # Integrity-driven stochastic precision corruption.
        # Low structural integrity bleeds noise into epistemic precision —
        # beliefs become harder to sharpen cleanly, mirroring how physical
        # damage to neural tissue degrades signal-to-noise in biological brains.
        if self.state.integrity < 60.0 and precision_weights:
            _integrity_deficit = (60.0 - self.state.integrity) / 60.0  # 0..1
            if self._rng.random() < 0.15 * _integrity_deficit:
                _vital = self._rng.choice(list(precision_weights.keys()))
                _corruption = self._rng.uniform(0.1, 0.3) * _integrity_deficit
                precision_weights[_vital] = max(
                    0.1, precision_weights[_vital] * (1.0 - _corruption)
                )
                self.diary.append(DiaryEntry(
                    tick=self.state.entropy,
                    role="repair",
                    content=(
                        f"INTEGRITY CORRUPTION: low integrity ({self.state.integrity:.1f}) "
                        f"degraded '{_vital}' precision weight "
                        f"by {_corruption * 100:.1f}% "
                        f"(→{precision_weights[_vital]:.3f})."
                    ),
                    metadata={"integrity_corruption": True, "vital": _vital},
                ))

        raw_proposals = self.goal_engine.generate_proposals(self.state)

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

        # Soul tension — an ethics hard-block is a betrayal event.
        # The wound is etched permanently: "you could sell out — and won't."
        if ethics_blocks > 0:
            self.soul_tension.maybe_scar(self.state, event_type="ethics_violation")
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

        # Meta-cognitive layer: compute higher-order free-energy contribution
        # and add it to EFE scores before policy selection.
        _diary_entries = self.diary.recent(1)
        _diary_snapshot = _diary_entries[-1].content if _diary_entries else ""
        _llm_cf = self.llm_narrator if hasattr(self, "llm_narrator") else None
        meta_cost = self.meta_self.update(
            current_vitals=self.state,
            base_affect=self.state.affect,
            diary_snapshot=_diary_snapshot,
            llm_counterfactual=_llm_cf,
        )

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
            # Augment EFE scores with meta-cognitive precision cost
            for k in efe_scores:
                efe_scores[k] += meta_cost
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
            # ---- Counterfactual simulation (DFS fear-based pruning) -------
            # Run the CounterfactualEngine BEFORE active_inference_step so
            # terminal risk penalties can influence selection.  Hard-pruned
            # branches (lethal within 2 ticks) cost almost nothing; the full
            # 10-step cost is only paid for surviving trajectories.
            #
            # Soul tension adjusts the CF engine's look-ahead depth.
            # High tension → deeper horizon + later hard prune (dare the dark).
            # Params are restored after the run so the engine defaults are
            # not permanently altered by a single tick's tension level.
            _cf_params = self.soul_tension.counterfactual_params()
            _cf_base_horizon = self.counterfactual_engine.horizon
            _cf_base_prune = self.counterfactual_engine.hard_prune_depth
            self.counterfactual_engine.horizon = int(
                _cf_base_horizon * _cf_params["horizon_scale"]
            )
            self.counterfactual_engine.hard_prune_depth = (
                _cf_base_prune + int(_cf_params["hard_prune_depth_extra"])
            )
            try:
                cf_traces = self.counterfactual_engine.run_batch(self.state, safe_proposals)
            finally:
                # Always restore defaults so soul tension affects only this tick
                self.counterfactual_engine.horizon = _cf_base_horizon
                self.counterfactual_engine.hard_prune_depth = _cf_base_prune
            cf_energy, cf_heat = self.counterfactual_engine.compute_metabolic_cost(cf_traces)
            if cf_energy > 0 or cf_heat > 0:
                self.state.apply_action_feedback(
                    delta_energy=-cf_energy,
                    delta_heat=cf_heat,
                )
            # Map proposal name → terminal_risk for post-selection adjustment
            cf_risk: dict[str, float] = {t.proposal_name: t.terminal_risk for t in cf_traces}

            # Log any lethal-zone discoveries (hard-pruned proposals) to diary
            lethal_names = [
                t.proposal_name for t in cf_traces if t.terminal_risk == 1.0
            ]
            if lethal_names:
                self.diary.append(DiaryEntry(
                    tick=self.state.entropy,
                    role="thought",
                    content=(
                        f"COUNTERFACTUAL: lethal-zone detected for "
                        f"{lethal_names!r} — fear-pruned at depth ≤ 2."
                    ),
                    metadata={"lethal_proposals": lethal_names},
                ))

            # Get adapted setpoints from HomeostasisAdapter for EFE scoring
            adapted_sp = self.homeostasis_adapter.adapted_setpoints()

            # active_inference_step charges the cognitive cost of evaluating proposals
            result = active_inference_step(
                self.state,
                safe_proposals,
                precision_weights=precision_weights,
                compute_load=self._compute_load,
                reward_discount=reward_discount,
                setpoints=adapted_sp,
            )

            # Add counterfactual terminal-risk penalty to each proposal's EFE.
            # This can change the selected proposal if the initial winner has a
            # dangerous trajectory that the 5-step multistep EFE missed.
            for name in result.efe_scores:
                result.efe_scores[name] += cf_risk.get(name, 0.0) * CF_RISK_WEIGHT

            # Re-select based on risk-adjusted scores
            best_name = min(result.efe_scores, key=lambda k: result.efe_scores[k])
            if best_name != result.selected.name:
                # Counterfactual changed the selection — find matching proposal.
                # Fall back to original selection if no match (should not occur
                # in normal operation but guards against StopIteration).
                override = next(
                    (p for p in safe_proposals if p.name == best_name),
                    result.selected,
                )
                result = result.__class__(
                    selected=override,
                    efe_scores=result.efe_scores,
                    reasoning=result.reasoning + f" [CF→{best_name}]",
                    inference_cost=result.inference_cost,
                    efe_components=result.efe_components,
                )
            selected = result.selected

            # Cache EFE component breakdown for CollapseProbe / TickRecord logging
            if result.efe_components is not None:
                self._last_efe_accuracy = result.efe_components.accuracy
                self._last_efe_complexity = result.efe_components.complexity
                self._last_efe_risk = result.efe_components.risk
                self._last_efe_wear = result.efe_components.wear

            # Add hierarchical EFE penalty to each proposal's score
            if hier_efe_penalty > 0:
                for k in result.efe_scores:
                    result.efe_scores[k] += hier_efe_penalty
            # Augment EFE scores with meta-cognitive precision cost
            for k in result.efe_scores:
                result.efe_scores[k] += meta_cost
            efe_scores = result.efe_scores

            cost = result.inference_cost
            cost_str = (
                f"inference_cost=[E={cost.energy_cost:.3f} H={cost.heat_cost:.3f} "
                f"KL={cost.kl_complexity:.1f} prec={cost.precision_used:.2f}]"
                if cost else ""
            )
            cf_str = (
                f" cf_risk={cf_risk.get(selected.name, 0.0):.2f}"
                if cf_risk else ""
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
                        f"{result.reasoning} {cost_str}{cf_str}{fm_str}{hier_str}"
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
                        "cf_risk": cf_risk.get(selected.name, 0.0),
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

        # Phase 4 — Constrained self-modification.
        # Proposals emerge from the hierarchy errors + thalamus channel weights
        # + precision regime that were computed *this tick* inside DECIDE.
        # Only runs at evolved stage; returns None otherwise.
        self_mod_result: SelfModResult | None = None
        if (
            self.state.stage == "evolved"
            and hierarchy_signal is not None
            and gate_report is not None
        ):
            self_mod_result = self.self_mod_engine.attempt(
                state=self.state,
                hierarchy_signal=hierarchy_signal,
                gate_report=gate_report,
                precision_report=precision_report,
                mask_name=self.rotator.active.name,
                interiority_score=self.meta_self.interiority_score(),
            )

        return ethics_blocks, self_mod_result

    # ------------------------------------------------------------------ #
    # Death handler                                                        #
    # ------------------------------------------------------------------ #

    def _handle_death(self, exc: GhostDeathException) -> None:
        self.last_death = exc
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

    def _apply_resurrection_shock(self) -> None:
        """Apply mandatory metabolic penalties and memory drift on reboot.

        Every resurrection from a prior life is traumatic — not a free reload.
        The organism wakes up damaged, hot, and dirty; its Bayesian priors have
        drifted during the death–rebirth gap.  This makes each run feel like a
        genuinely new, fragile life rather than a continuable save file.

        Applied costs
        -------------
        - Integrity penalty    : structural memory damage from death transition
        - Heat + waste spike   : physiological reboot overhead (stochastic)
        - Stability drop       : entropic fragility of a fresh awakening
        - Prior precision drift: one random unprotected Surgeon prior is
          loosened, reflecting belief degradation during the dark interval
        """
        integrity_penalty = 15.0
        heat_spike = self._rng.uniform(5.0, 10.0)
        waste_spike = self._rng.uniform(5.0, 15.0)
        stability_penalty = self._rng.uniform(5.0, 12.0)

        self.state.apply_action_feedback(
            delta_integrity=-integrity_penalty,
            delta_heat=heat_spike,
            delta_waste=waste_spike,
            delta_stability=-stability_penalty,
        )

        # Partial memory corruption: loosen one random unprotected prior
        _corruptible = [p for p in self.surgeon.priors if not p.protected]
        prior_note = ""
        if _corruptible:
            target = self._rng.choice(_corruptible)
            drift = self._rng.uniform(0.2, 0.6)
            target.precision = min(8.0, target.precision + drift)
            prior_note = f" Prior '{target.name}' precision drifted to {target.precision:.2f}."

        self.diary.append(DiaryEntry(
            tick=self.state.entropy,
            role="thought",
            content=(
                f"RESURRECTION SHOCK: prior life ended at tick {self.state.entropy}. "
                f"ΔM=-{integrity_penalty:.1f} ΔT=+{heat_spike:.1f} "
                f"ΔW=+{waste_spike:.1f} ΔS=-{stability_penalty:.1f}."
                f"{prior_note} This life is fragile."
            ),
            metadata={
                "resurrection": True,
                "integrity_penalty": integrity_penalty,
                "heat_spike": heat_spike,
                "waste_spike": waste_spike,
                "stability_penalty": stability_penalty,
            },
        ))

        # Notify meta-cognitive layer of the continuity break so it can
        # register the epistemic discontinuity and adjust its priors.
        self.meta_self.handle_restart(previous_continuity_anchor=None)

    def _apply_death_scar(self, death_memory: str) -> None:
        """Apply a small permanent metabolic modifier based on the cause of the
        previous death.

        Each death type leaves a different scar that nudges the new organism's
        starting state away from the specific failure mode that killed its
        predecessor.  The effect is intentionally modest — a lesson, not
        invincibility.

        Scar table
        ----------
        ThermalDeathException  → start 3 heat / 2 waste lower (cooler, cleaner)
        EnergyDeathException   → start with 5 extra energy (more reserves)
        MemoryCollapseException→ start with 5 extra integrity (sturdier memory)
        EntropyDeathException  → start with 5 extra stability (more entropic inertia)
        """
        exc_type = death_memory.split(":", 1)[0].strip()
        if exc_type == "ThermalDeathException":
            self.state.apply_action_feedback(delta_heat=-3.0, delta_waste=-2.0)
            scar_desc = "thermal scar: Δheat=-3.0 Δwaste=-2.0 (cooler start)"
        elif exc_type == "EnergyDeathException":
            self.state.apply_action_feedback(delta_energy=5.0)
            scar_desc = "energy scar: Δenergy=+5.0 (deeper reserves)"
        elif exc_type == "MemoryCollapseException":
            self.state.apply_action_feedback(delta_integrity=5.0)
            scar_desc = "memory scar: Δintegrity=+5.0 (hardened memory)"
        elif exc_type == "EntropyDeathException":
            self.state.apply_action_feedback(delta_stability=5.0)
            scar_desc = "entropy scar: Δstability=+5.0 (entropic inertia)"
        else:
            scar_desc = "unknown death type — no scar applied"

        self.diary.append(DiaryEntry(
            tick=self.state.entropy,
            role="thought",
            content=f"[DEATH_SCAR] Life {self._life_number}: {scar_desc}.",
            metadata={"death_scar": True, "exc_type": exc_type, "life_number": self._life_number},
        ))

    def _write_respawn_event(self) -> None:
        """Write a respawn sentinel record to the vitals JSONL log.

        This lets downstream tooling (e.g. plot_vitals.py) draw life boundaries
        and annotate each segment with the death cause that preceded it.
        """
        if not self._vitals_log:
            return
        if self._vitals_fh is None:
            self._vitals_fh = open(self._vitals_log, "a", buffering=1)
        record = {
            "event": "respawn",
            "life_number": self._life_number,
            "previous_death": self._death_memory,
        }
        self._vitals_fh.write(json.dumps(record) + "\n")

    def _load_state(self) -> tuple[MetabolicState, bool]:
        """Load MetabolicState from disk, or create a fresh one.

        Returns
        -------
        (state, was_resurrected)
            ``was_resurrected`` is True when an existing state file was loaded
            with entropy > 0, indicating this run is a reboot from a prior life.
        """
        state_file = os.environ.get("GHOST_STATE_FILE", STATE_FILE)
        if os.path.exists(state_file):
            with open(state_file) as fh:
                data = json.load(fh)
            state = MetabolicState.from_dict(data)
            was_resurrected = state.entropy > 0
            return state, was_resurrected
        return MetabolicState(), False

    # ------------------------------------------------------------------ #
    # Vitals logging                                                        #
    # ------------------------------------------------------------------ #

    def _write_vitals_log(
        self,
        action: str,
        env_event,
        self_mod_approved: int = 0,
        self_mod_blocked: int = 0,
        stressor_event: str = "",
    ) -> None:
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
            "self_mod_approved": self_mod_approved,
            "self_mod_blocked": self_mod_blocked,
            "stressor_event": stressor_event,
        }
        # Append phase-transition probe signals when a snapshot is available
        if self._last_collapse_snapshot is not None:
            snap = self._last_collapse_snapshot
            record["pre_collapse_score"] = round(snap.pre_collapse_score, 4)
            record["plasticity_index"] = round(snap.plasticity_index, 4)
            record["guardian_fraction"] = round(snap.guardian_fraction, 4)
            record["dreamer_fraction"] = round(snap.dreamer_fraction, 4)
            record["action_entropy"] = round(snap.action_entropy, 4)
            record["mask_entropy"] = round(snap.mask_entropy, 4)
            record["d_allostatic"] = round(snap.d_allostatic, 5)
            record["d_energy"] = round(snap.d_energy, 5)
            record["d_heat"] = round(snap.d_heat, 5)
            record["near_transition"] = snap.is_near_transition
        # Awakening mechanic tracking
        record["awakening_count"] = len(self.awakening_history)
        if self.meta_self.ticks_since_calcification is not None:
            record["ticks_since_calcification"] = self.meta_self.ticks_since_calcification
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
