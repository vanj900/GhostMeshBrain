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
import signal
import sys
import time

from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.core.exceptions import GhostDeathException
from thermodynamic_agency.cognition.inference import (
    active_inference_step,
    generate_default_proposals,
)
from thermodynamic_agency.cognition.ethics import EthicalEngine
from thermodynamic_agency.cognition.janitor import Janitor
from thermodynamic_agency.cognition.surgeon import Surgeon
from thermodynamic_agency.cognition.personality import MaskRotator
from thermodynamic_agency.cognition.precision import PrecisionEngine
from thermodynamic_agency.memory.diary import RamDiary, DiaryEntry
from thermodynamic_agency.interface.hud import print_hud

STATE_FILE = os.environ.get("GHOST_STATE_FILE", "/dev/shm/ghost_metabolic.json")
DIARY_PATH = os.environ.get("GHOST_DIARY_PATH", "/dev/shm/ghost_diary.db")
PULSE_SECONDS = float(os.environ.get("GHOST_PULSE", "5"))
COMPUTE_LOAD = float(os.environ.get("GHOST_COMPUTE_LOAD", "1.0"))
SHOW_HUD = os.environ.get("GHOST_HUD", "1") == "1"


class GhostMesh:
    """The full thermodynamic organism."""

    def __init__(self) -> None:
        # Re-read env vars at construction time so tests/callers can override them.
        self._state_file = os.environ.get("GHOST_STATE_FILE", STATE_FILE)
        self._diary_path = os.environ.get("GHOST_DIARY_PATH", DIARY_PATH)
        self._pulse_seconds = float(os.environ.get("GHOST_PULSE", str(PULSE_SECONDS)))
        self._compute_load = float(os.environ.get("GHOST_COMPUTE_LOAD", str(COMPUTE_LOAD)))
        self._show_hud = os.environ.get("GHOST_HUD", "1") == "1"
        # Load or initialise metabolic state
        self.state = self._load_state()
        self.diary = RamDiary(path=self._diary_path)
        self.ethics = EthicalEngine()
        self.janitor = Janitor(diary=self.diary)
        self.surgeon = Surgeon(diary=self.diary)
        self.rotator = MaskRotator(initial_mask="Guardian")
        self.precision_engine = PrecisionEngine()
        self._running = False

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

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------ #
    # Single pulse                                                         #
    # ------------------------------------------------------------------ #

    def _pulse(self) -> None:
        """Execute one heartbeat."""
        mask = self.rotator.active
        action = self.state.tick(compute_load=self._compute_load)
        self.rotator.tick(self.state.entropy)

        if self._show_hud:
            print_hud(self.state.to_dict(), self.rotator.status())

        # Rotate mask based on action and affect signal.
        # High negative affect (rising surprise) biases toward Judge/Guardian.
        # High positive affect (resolving surprise) biases toward Dreamer/Courier.
        affect = self.state.affect
        if affect < -0.4:
            self.rotator.maybe_rotate(self.state.entropy, metabolic_hint="REPAIR")
        elif affect > 0.4:
            self.rotator.maybe_rotate(self.state.entropy, metabolic_hint="REST")
        else:
            self.rotator.maybe_rotate(self.state.entropy, metabolic_hint=action)

        # Dispatch action
        if action == "FORAGE":
            self._forage()
        elif action == "REST":
            self._rest()
        elif action == "REPAIR":
            self._repair()
        else:  # "DECIDE"
            self._decide()

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

    def _decide(self) -> None:
        """Full active-inference planning cycle with thermodynamic precision cost."""
        # Tune precision weights based on current metabolic state.
        # The PrecisionEngine applies its own metabolic cost for sharpening attention.
        precision_report = self.precision_engine.tune(
            self.state, compute_load=self._compute_load
        )
        # Apply the precision-sharpening metabolic cost
        if precision_report.energy_cost > 0 or precision_report.heat_cost > 0:
            self.state.apply_action_feedback(
                delta_energy=-precision_report.energy_cost,
                delta_heat=precision_report.heat_cost,
            )

        proposals = generate_default_proposals(self.state)

        # Ethics immune screening
        safe_proposals = self.ethics.immune_scan(proposals, self.state)
        if not safe_proposals:
            safe_proposals = proposals  # fallback: allow all if all blocked

        # active_inference_step charges the cognitive cost of evaluating proposals
        result = active_inference_step(
            self.state,
            safe_proposals,
            precision_weights=precision_report.weights,
            compute_load=self._compute_load,
        )
        selected = result.selected

        cost = result.inference_cost
        cost_str = (
            f"inference_cost=[E={cost.energy_cost:.3f} H={cost.heat_cost:.3f} "
            f"KL={cost.kl_complexity:.1f} prec={cost.precision_used:.2f}]"
            if cost else ""
        )

        self.diary.append(
            DiaryEntry(
                tick=self.state.entropy,
                role="thought",
                content=(
                    f"DECIDE: selected '{selected.name}' "
                    f"(EFE={result.efe_scores[selected.name]:.2f}). "
                    f"{result.reasoning} {cost_str}"
                ),
                metadata={
                    "mask": self.rotator.active.name,
                    "precision_regime": precision_report.regime,
                    "affect": self.state.affect,
                    "free_energy": precision_report.free_energy,
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
    # Signal handler                                                       #
    # ------------------------------------------------------------------ #

    def _handle_signal(self, signum: int, frame: object) -> None:
        print("\n[GhostMesh] Signal received — shutting down gracefully.", file=sys.stderr)
        self._running = False


# ------------------------------------------------------------------ #
# Module entry-point                                                   #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    mesh = GhostMesh()
    mesh.run()
