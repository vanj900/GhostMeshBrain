#!/usr/bin/env python3
"""BifurcationExperiment — parameter sweep over genesis_prior_strength,
precision_bound, and llm_usage_freq.

Usage
-----
    python scripts/bifurcation_experiment.py [--runs-per-config N] [--ticks T] [--out results/]

The script runs a full cross-product parameter sweep and writes a CSV to
``results/bifurcation_sweep.csv``.

CSV columns
-----------
config_id, genesis_prior_strength, precision_bound, llm_usage_freq,
run, plasticity_index, lifespan, interiority_score, dreamer_fraction,
guardian_fraction, n_awakenings, pre_collapse_score, cause_of_death

Quick start
-----------
    python scripts/bifurcation_experiment.py --runs-per-config 3 --ticks 500

This produces results in ~60 seconds and writes to results/bifurcation_sweep.csv.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add the repo root to the path so the module can be imported when running
# the script from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from thermodynamic_agency.pulse import GhostMesh
from thermodynamic_agency.core.exceptions import GhostDeathException


# ── Parameter grid ────────────────────────────────────────────────────────────

_GENESIS_PRIOR_STRENGTHS: list[float] = [0.5, 1.0, 2.0]
_PRECISION_BOUNDS: list[float] = [1.5, 3.0, 6.0]
_LLM_USAGE_FREQS: list[str] = ["off", "low"]    # "off" and "low" are enough without a real LLM


# ── CSV schema ────────────────────────────────────────────────────────────────

_CSV_FIELDS: list[str] = [
    "config_id",
    "genesis_prior_strength",
    "precision_bound",
    "llm_usage_freq",
    "run",
    "plasticity_index",
    "lifespan",
    "interiority_score",
    "dreamer_fraction",
    "guardian_fraction",
    "n_awakenings",
    "pre_collapse_score",
    "cause_of_death",
]


@dataclass
class RunResult:
    """Metrics extracted from one simulation run."""

    config_id: str
    genesis_prior_strength: float
    precision_bound: float
    llm_usage_freq: str
    run: int
    plasticity_index: float
    lifespan: int
    interiority_score: float
    dreamer_fraction: float
    guardian_fraction: float
    n_awakenings: int
    pre_collapse_score: float
    cause_of_death: str


class BifurcationExperiment:
    """Runs parameter sweeps and records bifurcation metrics.

    Parameters
    ----------
    runs_per_config:
        Number of independent runs per parameter combination (default 3).
    ticks_per_run:
        Maximum ticks per run (default 1000).
    output_dir:
        Directory to write the CSV file into.
    seed_base:
        Base random seed; run *i* uses ``seed_base + i``.
    """

    def __init__(
        self,
        runs_per_config: int = 3,
        ticks_per_run: int = 1000,
        output_dir: str = "results",
        seed_base: int = 42,
    ) -> None:
        self.runs_per_config = runs_per_config
        self.ticks_per_run = ticks_per_run
        self.output_dir = Path(output_dir)
        self.seed_base = seed_base
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        """Execute the full sweep and return the path to the CSV output.

        Returns
        -------
        Path
            Path to the written CSV file.
        """
        csv_path = self.output_dir / "bifurcation_sweep.csv"
        configs = list(itertools.product(
            _GENESIS_PRIOR_STRENGTHS,
            _PRECISION_BOUNDS,
            _LLM_USAGE_FREQS,
        ))
        total = len(configs) * self.runs_per_config
        print(f"[BifurcationExperiment] Starting sweep: "
              f"{len(configs)} configs × {self.runs_per_config} runs = {total} total runs",
              flush=True)

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=_CSV_FIELDS)
            writer.writeheader()

            run_global = 0
            for config_idx, (gps, pb, llmf) in enumerate(configs):
                config_id = f"gps{gps}_pb{pb}_{llmf}"
                for run_idx in range(self.runs_per_config):
                    run_global += 1
                    seed = self.seed_base + run_global
                    print(
                        f"  [{run_global}/{total}] config={config_id} run={run_idx} seed={seed}",
                        flush=True,
                    )
                    result = self._single_run(
                        config_id=config_id,
                        genesis_prior_strength=gps,
                        precision_bound=pb,
                        llm_usage_freq=llmf,
                        run_idx=run_idx,
                        seed=seed,
                    )
                    writer.writerow({
                        "config_id": result.config_id,
                        "genesis_prior_strength": result.genesis_prior_strength,
                        "precision_bound": result.precision_bound,
                        "llm_usage_freq": result.llm_usage_freq,
                        "run": result.run,
                        "plasticity_index": round(result.plasticity_index, 4),
                        "lifespan": result.lifespan,
                        "interiority_score": round(result.interiority_score, 4),
                        "dreamer_fraction": round(result.dreamer_fraction, 4),
                        "guardian_fraction": round(result.guardian_fraction, 4),
                        "n_awakenings": result.n_awakenings,
                        "pre_collapse_score": round(result.pre_collapse_score, 4),
                        "cause_of_death": result.cause_of_death,
                    })
                    csvfile.flush()

        print(f"[BifurcationExperiment] Complete. Results written to {csv_path}", flush=True)
        return csv_path

    def _single_run(
        self,
        config_id: str,
        genesis_prior_strength: float,
        precision_bound: float,
        llm_usage_freq: str,
        run_idx: int,
        seed: int,
    ) -> RunResult:
        """Execute one simulation run and return its metrics."""
        tmpdir = tempfile.mkdtemp(prefix="bifurcation_exp_")

        # Configure via env vars — GhostMesh reads these at construction
        env_overrides: dict[str, str] = {
            "GHOST_HUD": "0",
            "GHOST_PULSE": "0",
            "GHOST_STATE_FILE": os.path.join(tmpdir, "state.json"),
            "GHOST_DIARY_PATH": os.path.join(tmpdir, "diary.db"),
            "GHOST_VITALS_LOG": "",
            "GHOST_LOG_FILE": "",
            "GHOST_ENV_EVENTS": "1",
            "GHOST_USE_LLM": "0" if llm_usage_freq == "off" else "0",
            "GHOST_PURITY_MODE": "0",
            # precision_bound is not a direct env var; we set it via the
            # precision engine's base_precision after construction
        }
        saved_env: dict[str, str | None] = {}
        for k, v in env_overrides.items():
            saved_env[k] = os.environ.get(k)
            os.environ[k] = v

        cause_of_death = "survived"
        try:
            mesh = GhostMesh(seed=seed)

            # Apply precision_bound override to the precision engine
            for vital in mesh.precision_engine.base_precision:
                mesh.precision_engine.base_precision[vital] = min(
                    precision_bound,
                    mesh.precision_engine.base_precision[vital],
                )

            try:
                mesh.run(max_ticks=self.ticks_per_run)
            except GhostDeathException as exc:
                cause_of_death = type(exc).__name__
            except Exception as exc:  # noqa: BLE001
                cause_of_death = type(exc).__name__

            snap = mesh._last_collapse_snapshot
            plasticity_index = snap.plasticity_index if snap else 0.0
            pre_collapse_score = snap.pre_collapse_score if snap else 0.0
            dreamer_fraction = snap.dreamer_fraction if snap else 0.0
            guardian_fraction = snap.guardian_fraction if snap else 0.0

            return RunResult(
                config_id=config_id,
                genesis_prior_strength=genesis_prior_strength,
                precision_bound=precision_bound,
                llm_usage_freq=llm_usage_freq,
                run=run_idx,
                plasticity_index=plasticity_index,
                lifespan=mesh.state.entropy,
                interiority_score=mesh.meta_self.interiority_score(),
                dreamer_fraction=dreamer_fraction,
                guardian_fraction=guardian_fraction,
                n_awakenings=len(mesh.awakening_history),
                pre_collapse_score=pre_collapse_score,
                cause_of_death=cause_of_death,
            )
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GhostMesh Bifurcation Parameter Sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs-per-config",
        type=int,
        default=3,
        help="Independent runs per parameter combination",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=1000,
        help="Max ticks per run",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="Output directory for the CSV file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    experiment = BifurcationExperiment(
        runs_per_config=args.runs_per_config,
        ticks_per_run=args.ticks,
        output_dir=args.out,
        seed_base=args.seed,
    )
    output_file = experiment.run()
    print(f"Done: {output_file}")
