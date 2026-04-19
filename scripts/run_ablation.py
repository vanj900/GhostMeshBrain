#!/usr/bin/env python3
"""Controlled ablation grid for the Plasticity–Longevity Phase Transition.

Direction #1 of the breakthrough analysis: systematically vary three control
parameters and measure where the Dreamer→Guardian bifurcation occurs.

Parameters swept
----------------
allostatic_multiplier  : scales the allostatic-load gain constants in
                         MetabolicState (1.0 = baseline, >1 = faster loading)
precision_damping      : additional damping factor applied to the overload
                         precision regime (1.0 = no extra damping, <1 = harder
                         damping when FE > STRESS_UPPER)
affect_sensitivity     : scales the affect → cortisol accumulation rate
                         (1.0 = baseline, >1 = stronger cortisol response)

Outputs
-------
results/ablation_TIMESTAMP/
    grid_summary.jsonl     — one JSON line per experiment with key metrics
    {experiment_id}.jsonl  — per-tick TickRecord for each experiment (if
                             ABLATION_SAVE_TICKS=1 env var is set)

Usage
-----
    python scripts/run_ablation.py                    # default 3×3×3 grid
    python scripts/run_ablation.py --ticks 5000       # ticks per run
    python scripts/run_ablation.py --output /tmp/abl  # custom output dir
    python scripts/run_ablation.py --dry-run          # print plan, no exec
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

# Ensure the package root is importable when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from thermodynamic_agency.core.metabolic import (
    MetabolicState,
    _AL_ALPHA_T,
    _AL_ALPHA_W,
    _AL_ALPHA_FE,
    _CORTISOL_RATE,
)
from thermodynamic_agency.cognition.precision import (
    PrecisionEngine,
    STRESS_UPPER,
)
from thermodynamic_agency.cognition.collapse_probe import CollapseProbe
from thermodynamic_agency.run_logger import RunLogger, TickRecord
from thermodynamic_agency.pulse import GhostMesh


# ------------------------------------------------------------------ #
# Grid definition                                                      #
# ------------------------------------------------------------------ #

DEFAULT_AL_MULTIPLIERS = [0.5, 1.0, 2.0]
DEFAULT_PRECISION_DAMPINGS = [0.6, 1.0, 1.4]
DEFAULT_AFFECT_SENSITIVITIES = [0.5, 1.0, 2.0]
DEFAULT_TICKS = 2000
DEFAULT_SEED = 42


# ------------------------------------------------------------------ #
# Parameter patch helpers                                             #
# ------------------------------------------------------------------ #

def _patch_allostatic(multiplier: float) -> tuple[float, float, float]:
    """Return patched (alpha_T, alpha_W, alpha_FE) for a given multiplier."""
    import thermodynamic_agency.core.metabolic as _m
    _orig = (_m._AL_ALPHA_T, _m._AL_ALPHA_W, _m._AL_ALPHA_FE)
    _m._AL_ALPHA_T = _AL_ALPHA_T * multiplier
    _m._AL_ALPHA_W = _AL_ALPHA_W * multiplier
    _m._AL_ALPHA_FE = _AL_ALPHA_FE * multiplier
    return _orig


def _restore_allostatic(orig: tuple[float, float, float]) -> None:
    import thermodynamic_agency.core.metabolic as _m
    _m._AL_ALPHA_T, _m._AL_ALPHA_W, _m._AL_ALPHA_FE = orig


def _patch_precision_damping(damping: float) -> float:
    """Monkey-patch the overload damping factor in PrecisionEngine.

    The baseline ``_overload_weights`` method damps by up to 50% at full
    overload (damp = 1.0 − 0.5 × overload_factor).  We replace this with
    damp = 1.0 − (0.5 × damping) × overload_factor so damping > 1 damps
    harder and damping < 1 damps less.
    """
    import thermodynamic_agency.cognition.precision as _pe

    _orig_overload = PrecisionEngine._overload_weights

    _d = damping  # capture in closure

    def _patched_overload(self, state, fe, compute_load):
        overload_factor = min(1.0, (fe - STRESS_UPPER) / 45.0)
        damp = 1.0 - 0.5 * _d * overload_factor
        weights = {}
        for vital, base in self.base_precision.items():
            if vital in ("energy", "heat"):
                weights[vital] = min(6.0, base * (1.0 + overload_factor))
            else:
                weights[vital] = max(0.3, base * damp)
        return weights, 0.0, 0.0

    PrecisionEngine._overload_weights = _patched_overload  # type: ignore[assignment]
    return _orig_overload  # type: ignore[return-value]


def _restore_precision_damping(orig) -> None:
    PrecisionEngine._overload_weights = orig  # type: ignore[assignment]


def _patch_affect_sensitivity(sensitivity: float) -> float:
    """Scale the cortisol accumulation rate in MetabolicState.tick()."""
    import thermodynamic_agency.core.metabolic as _m
    _orig = _m._CORTISOL_RATE
    _m._CORTISOL_RATE = _CORTISOL_RATE * sensitivity
    return _orig


def _restore_affect_sensitivity(orig: float) -> None:
    import thermodynamic_agency.core.metabolic as _m
    _m._CORTISOL_RATE = orig


# ------------------------------------------------------------------ #
# Experiment dataclass                                                 #
# ------------------------------------------------------------------ #

@dataclass
class ExperimentSpec:
    experiment_id: str
    al_multiplier: float
    precision_damping: float
    affect_sensitivity: float
    seed: int
    ticks: int


@dataclass
class ExperimentResult:
    experiment_id: str
    al_multiplier: float
    precision_damping: float
    affect_sensitivity: float
    seed: int
    ticks_run: int
    death_tick: int | None          # None = survived all ticks
    death_cause: str                # "" = survived
    # Phase-transition metrics
    first_transition_tick: int | None  # tick when near_transition first = True
    mean_plasticity_index: float
    final_plasticity_index: float
    mean_guardian_fraction: float
    max_pre_collapse_score: float
    mean_pre_collapse_score: float
    # Survival metrics
    mean_health: float
    mean_allostatic_load: float
    mean_free_energy: float
    # Action distribution
    action_distribution: dict[str, int]


# ------------------------------------------------------------------ #
# Single experiment runner                                             #
# ------------------------------------------------------------------ #

def run_experiment(
    spec: ExperimentSpec,
    output_dir: Path | None = None,
    save_ticks: bool = False,
) -> ExperimentResult:
    """Run one experiment with the given parameter set.

    Patches module-level constants, runs GhostMesh for spec.ticks, restores
    constants, and returns an ExperimentResult.
    """
    # ---- Apply patches ---------------------------------------------------
    orig_al = _patch_allostatic(spec.al_multiplier)
    orig_pd = _patch_precision_damping(spec.precision_damping)
    orig_as = _patch_affect_sensitivity(spec.affect_sensitivity)

    # Redirect state file to /tmp so experiments don't clobber each other
    os.environ["GHOST_STATE_FILE"] = f"/tmp/_abl_{spec.experiment_id}.json"
    os.environ["GHOST_DIARY_PATH"] = f"/tmp/_abl_{spec.experiment_id}.db"
    os.environ["GHOST_HUD"] = "0"
    os.environ["GHOST_ENV_EVENTS"] = "1"
    os.environ["GHOST_VITALS_LOG"] = ""  # disable JSONL (we use run_logger)
    if save_ticks and output_dir is not None:
        tick_path = str(output_dir / f"{spec.experiment_id}.jsonl")
        os.environ["GHOST_LOG_FILE"] = tick_path
    else:
        os.environ["GHOST_LOG_FILE"] = ""

    try:
        mesh = GhostMesh(seed=spec.seed)
        mesh._pulse_seconds = 0.0  # no sleep between ticks

        ticks_run = 0
        death_tick: int | None = None
        death_cause = ""
        first_transition_tick: int | None = None

        # Track per-tick probe stats for summary metrics
        plasticity_vals: list[float] = []
        guardian_vals: list[float] = []
        pre_collapse_vals: list[float] = []
        health_vals: list[float] = []
        al_vals: list[float] = []
        fe_vals: list[float] = []
        action_counts: dict[str, int] = {}

        from thermodynamic_agency.core.exceptions import GhostDeathException

        mesh._running = True
        for _ in range(spec.ticks):
            try:
                mesh._pulse()
            except GhostDeathException as exc:
                death_tick = mesh.state.entropy
                death_cause = type(exc).__name__
                break
            ticks_run += 1

            # Read probe snapshot from mesh
            snap = mesh._last_collapse_snapshot
            if snap is not None:
                plasticity_vals.append(snap.plasticity_index)
                guardian_vals.append(snap.guardian_fraction)
                pre_collapse_vals.append(snap.pre_collapse_score)
                if snap.is_near_transition and first_transition_tick is None:
                    first_transition_tick = mesh.state.entropy

            health_vals.append(mesh.state.health_score())
            al_vals.append(mesh.state.allostatic_load)
            fe_vals.append(mesh.state.free_energy_estimate())
            act = mesh.state.last_action or "DECIDE"
            action_counts[act] = action_counts.get(act, 0) + 1

    finally:
        mesh.run_logger.close()
        # Restore patches
        _restore_allostatic(orig_al)
        _restore_precision_damping(orig_pd)
        _restore_affect_sensitivity(orig_as)
        # Clean up temp state files
        for tmp in [
            os.environ.get("GHOST_STATE_FILE", ""),
            os.environ.get("GHOST_DIARY_PATH", ""),
        ]:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return ExperimentResult(
        experiment_id=spec.experiment_id,
        al_multiplier=spec.al_multiplier,
        precision_damping=spec.precision_damping,
        affect_sensitivity=spec.affect_sensitivity,
        seed=spec.seed,
        ticks_run=ticks_run,
        death_tick=death_tick,
        death_cause=death_cause,
        first_transition_tick=first_transition_tick,
        mean_plasticity_index=_mean(plasticity_vals),
        final_plasticity_index=plasticity_vals[-1] if plasticity_vals else 0.0,
        mean_guardian_fraction=_mean(guardian_vals),
        max_pre_collapse_score=max(pre_collapse_vals) if pre_collapse_vals else 0.0,
        mean_pre_collapse_score=_mean(pre_collapse_vals),
        mean_health=_mean(health_vals),
        mean_allostatic_load=_mean(al_vals),
        mean_free_energy=_mean(fe_vals),
        action_distribution=action_counts,
    )


# ------------------------------------------------------------------ #
# Grid generator                                                       #
# ------------------------------------------------------------------ #

def iter_grid(
    al_multipliers: list[float],
    precision_dampings: list[float],
    affect_sensitivities: list[float],
    seed: int,
    ticks: int,
) -> Iterator[ExperimentSpec]:
    """Yield all (al × pd × as) combinations."""
    idx = 0
    for al in al_multipliers:
        for pd in precision_dampings:
            for af in affect_sensitivities:
                exp_id = f"exp_{idx:04d}_al{al:.1f}_pd{pd:.1f}_af{af:.1f}"
                yield ExperimentSpec(
                    experiment_id=exp_id,
                    al_multiplier=al,
                    precision_damping=pd,
                    affect_sensitivity=af,
                    seed=seed,
                    ticks=ticks,
                )
                idx += 1


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-transition ablation grid runner for GhostMesh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticks", type=int, default=DEFAULT_TICKS,
        help="Number of ticks to run per experiment.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Output directory for results (default: results/ablation_TIMESTAMP/).",
    )
    parser.add_argument(
        "--al-multipliers", type=float, nargs="+",
        default=DEFAULT_AL_MULTIPLIERS,
        help="Allostatic-load multiplier values to sweep.",
    )
    parser.add_argument(
        "--precision-dampings", type=float, nargs="+",
        default=DEFAULT_PRECISION_DAMPINGS,
        help="Precision-damping values to sweep.",
    )
    parser.add_argument(
        "--affect-sensitivities", type=float, nargs="+",
        default=DEFAULT_AFFECT_SENSITIVITIES,
        help="Affect-sensitivity values to sweep.",
    )
    parser.add_argument(
        "--save-ticks", action="store_true",
        help="Save per-tick TickRecord JSONL for each experiment.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the experiment plan without running anything.",
    )
    args = parser.parse_args()

    # Determine output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) if args.output else Path("results") / f"ablation_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = list(iter_grid(
        al_multipliers=args.al_multipliers,
        precision_dampings=args.precision_dampings,
        affect_sensitivities=args.affect_sensitivities,
        seed=args.seed,
        ticks=args.ticks,
    ))
    n = len(specs)

    print(
        f"[ablation] {n} experiments × {args.ticks} ticks "
        f"→ {output_dir}",
        flush=True,
    )
    if args.dry_run:
        for s in specs:
            print(
                f"  {s.experiment_id}  al={s.al_multiplier}  "
                f"pd={s.precision_damping}  af={s.affect_sensitivity}"
            )
        return

    summary_path = output_dir / "grid_summary.jsonl"
    with open(summary_path, "w") as fh:
        for i, spec in enumerate(specs, 1):
            print(
                f"  [{i}/{n}] {spec.experiment_id} ...",
                end=" ", flush=True,
            )
            t0 = time.time()
            result = run_experiment(
                spec, output_dir=output_dir, save_ticks=args.save_ticks,
            )
            elapsed = time.time() - t0
            status = (
                f"died tick={result.death_tick} ({result.death_cause})"
                if result.death_tick
                else f"survived {result.ticks_run} ticks"
            )
            print(
                f"{status}  "
                f"plasticity={result.mean_plasticity_index:.3f}  "
                f"pcs={result.mean_pre_collapse_score:.3f}  "
                f"[{elapsed:.1f}s]",
                flush=True,
            )
            fh.write(json.dumps(asdict(result)) + "\n")
            fh.flush()

    print(f"\n[ablation] Done. Summary → {summary_path}")


if __name__ == "__main__":
    main()
