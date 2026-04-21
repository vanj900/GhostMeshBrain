#!/usr/bin/env python3
"""MortalityLineageExperiment — multi-generational death-chain parameter sweep.

Chains GhostMesh lives together via real death → LineageTracker → Q-table
mutation → rebirth.  Sweeps stressor_prob and stressor_intensity to test
whether moderate mortality pressure rescues plasticity across generations or
just kills everything faster while the Guardian attractor deepens.

Includes an immortal baseline where death-causing vital collapses are
intercepted and the organism is revived in-place (same total tick budget,
no generational cycling), providing the control arm for the core claim:
*mortality creates the uncomfortable bifurcation*.

Usage
-----
    python scripts/mortality_lineage_experiment.py [OPTIONS]

Quick start (fast test, ~30 seconds)
-------------------------------------
    python scripts/mortality_lineage_experiment.py \\
        --n-generations 5 --ticks-per-life 300 --runs 2

Full overnight sweep
--------------------
    python scripts/mortality_lineage_experiment.py \\
        --n-generations 50 --ticks-per-life 1000 --runs 3

Output files
------------
results/lineage_experiment.jsonl  — one JSON record per generation
results/lineage_experiment.csv    — per-config aggregated summary

CSV columns
-----------
config_id, stressor_prob, stressor_intensity, immortal, run,
generation, lifespan, dreamer_fraction, guardian_fraction,
plasticity_index, interiority_score, n_awakenings, cause_of_death,
fitness_score, plasticity_selection_r
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from thermodynamic_agency.pulse import GhostMesh
from thermodynamic_agency.core.exceptions import GhostDeathException
from thermodynamic_agency.evolution.lineage import (
    LineageTracker,
    Lineage,
    extract_top_q_entries,
    _generate_lineage_id,
    DEFAULT_MUTATION_RATE,
)


# ── Parameter grid ────────────────────────────────────────────────────────────

_STRESSOR_PROBS: list[float] = [0.0, 0.1, 0.3, 0.5]
_STRESSOR_INTENSITIES: list[float] = [1.0, 2.0, 3.0]

# ── CSV schema ────────────────────────────────────────────────────────────────

_CSV_FIELDS: list[str] = [
    "config_id",
    "stressor_prob",
    "stressor_intensity",
    "immortal",
    "run",
    "generation",
    "lifespan",
    "dreamer_fraction",
    "guardian_fraction",
    "plasticity_index",
    "interiority_score",
    "n_awakenings",
    "cause_of_death",
    "fitness_score",
    "plasticity_selection_r",
]

_JSONL_FIELDS: set[str] = set(_CSV_FIELDS)

# Maximum energy injection per revival tick in immortal mode.
_IMMORTAL_REVIVAL_ENERGY: float = 50.0
_IMMORTAL_REVIVAL_HEAT_DRAIN: float = -30.0
_IMMORTAL_REVIVAL_WASTE_DRAIN: float = -30.0
_IMMORTAL_REVIVAL_INTEGRITY_BOOST: float = 20.0
_IMMORTAL_REVIVAL_STABILITY_BOOST: float = 20.0


@dataclass
class GenerationRecord:
    """Per-generation metrics for one life in a lineage chain."""

    config_id: str
    stressor_prob: float
    stressor_intensity: float
    immortal: bool
    run: int
    generation: int
    lifespan: int
    dreamer_fraction: float
    guardian_fraction: float
    plasticity_index: float
    interiority_score: float
    n_awakenings: int
    cause_of_death: str
    fitness_score: float
    plasticity_selection_r: float  # updated after each life

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Round floats for readability
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 6)
        return d


class MortalityLineageExperiment:
    """Multi-generation experiment: death-chain vs immortal baseline.

    Parameters
    ----------
    n_generations:
        Number of successive lives per configuration run (mortal mode only;
        immortal mode runs one "life" of equivalent total ticks).
    ticks_per_life:
        Maximum ticks per life (mortal) or total tick budget (immortal).
    runs_per_config:
        Independent runs per (stressor_prob, stressor_intensity, immortal)
        triple.
    output_dir:
        Directory for JSONL and CSV output files.
    seed_base:
        Base RNG seed; per-run seeds are ``seed_base + run_index * 1000``.
    stressor_probs:
        Override the default stressor probability grid.
    stressor_intensities:
        Override the default stressor intensity grid.
    include_immortal:
        Whether to also run immortal-baseline configurations.
    """

    def __init__(
        self,
        n_generations: int = 20,
        ticks_per_life: int = 1000,
        runs_per_config: int = 3,
        output_dir: str = "results",
        seed_base: int = 42,
        stressor_probs: list[float] | None = None,
        stressor_intensities: list[float] | None = None,
        include_immortal: bool = True,
    ) -> None:
        self.n_generations = n_generations
        self.ticks_per_life = ticks_per_life
        self.runs_per_config = runs_per_config
        self.output_dir = Path(output_dir)
        self.seed_base = seed_base
        self.stressor_probs = stressor_probs if stressor_probs is not None else _STRESSOR_PROBS
        self.stressor_intensities = (
            stressor_intensities if stressor_intensities is not None else _STRESSOR_INTENSITIES
        )
        self.include_immortal = include_immortal
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> tuple[Path, Path]:
        """Execute the full sweep.

        Returns
        -------
        (jsonl_path, csv_path)
        """
        jsonl_path = self.output_dir / "lineage_experiment.jsonl"
        csv_path = self.output_dir / "lineage_experiment.csv"

        # Build config list: all (prob, intensity, immortal) combos
        configs: list[tuple[float, float, bool]] = []
        for prob in self.stressor_probs:
            for intensity in self.stressor_intensities:
                configs.append((prob, intensity, False))
                if self.include_immortal:
                    configs.append((prob, intensity, True))

        total_runs = len(configs) * self.runs_per_config
        print(
            f"[MortalityLineageExperiment] "
            f"{len(configs)} configs × {self.runs_per_config} runs = {total_runs} runs "
            f"({self.n_generations} generations × {self.ticks_per_life} ticks/life each)",
            flush=True,
        )

        run_counter = 0
        with (
            open(jsonl_path, "w") as jsonl_fh,
            open(csv_path, "w", newline="") as csv_fh,
        ):
            writer = csv.DictWriter(csv_fh, fieldnames=_CSV_FIELDS)
            writer.writeheader()

            for prob, intensity, immortal in configs:
                config_id = f"sp{prob}_si{intensity}_{'immortal' if immortal else 'mortal'}"
                for run_idx in range(self.runs_per_config):
                    run_counter += 1
                    seed = self.seed_base + run_idx * 1000
                    print(
                        f"  [{run_counter}/{total_runs}] {config_id} run={run_idx} seed={seed}",
                        flush=True,
                    )
                    records = self._run_lineage_chain(
                        config_id=config_id,
                        stressor_prob=prob,
                        stressor_intensity=intensity,
                        immortal=immortal,
                        run_idx=run_idx,
                        seed=seed,
                    )
                    for rec in records:
                        d = rec.to_dict()
                        jsonl_fh.write(json.dumps(d) + "\n")
                        writer.writerow({k: d[k] for k in _CSV_FIELDS})
                    jsonl_fh.flush()
                    csv_fh.flush()

        print(
            f"[MortalityLineageExperiment] Done.\n"
            f"  JSONL → {jsonl_path}\n"
            f"  CSV  → {csv_path}",
            flush=True,
        )
        return jsonl_path, csv_path

    # ── Internal: run a full chain of N generations ───────────────────────────

    def _run_lineage_chain(
        self,
        *,
        config_id: str,
        stressor_prob: float,
        stressor_intensity: float,
        immortal: bool,
        run_idx: int,
        seed: int,
    ) -> list[GenerationRecord]:
        """Run N generations (mortal) or one equivalent immortal run.

        Returns a list of GenerationRecord, one per generation.
        """
        tmpdir = tempfile.mkdtemp(prefix="mle_")
        tracker = LineageTracker(
            path=os.path.join(tmpdir, "lineage.jsonl"),
            mutation_rate=DEFAULT_MUTATION_RATE,
        )
        rng = random.Random(seed)
        records: list[GenerationRecord] = []

        try:
            if immortal:
                records = self._run_immortal(
                    config_id=config_id,
                    stressor_prob=stressor_prob,
                    stressor_intensity=stressor_intensity,
                    run_idx=run_idx,
                    tracker=tracker,
                    rng=rng,
                    tmpdir=tmpdir,
                )
            else:
                records = self._run_mortal_chain(
                    config_id=config_id,
                    stressor_prob=stressor_prob,
                    stressor_intensity=stressor_intensity,
                    run_idx=run_idx,
                    tracker=tracker,
                    rng=rng,
                    tmpdir=tmpdir,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return records

    # ── Mortal chain: N successive lives with Q-table inheritance ─────────────

    def _run_mortal_chain(
        self,
        *,
        config_id: str,
        stressor_prob: float,
        stressor_intensity: float,
        run_idx: int,
        tracker: LineageTracker,
        rng: random.Random,
        tmpdir: str,
    ) -> list[GenerationRecord]:
        records: list[GenerationRecord] = []
        death_memory: str | None = None
        prev_lineage: Lineage | None = None

        for gen in range(self.n_generations):
            life_seed = rng.randint(0, 2**31)
            mesh = self._build_mesh(
                life_number=gen + 1,
                death_memory=death_memory,
                seed=life_seed,
                stressor_prob=stressor_prob,
                stressor_intensity=stressor_intensity,
                tmpdir=tmpdir,
            )

            # Seed Q-table from previous life (if any)
            if prev_lineage is not None and mesh.learner is not None:
                tracker.seed_q_table(mesh.learner._q_table, rng=rng)

            cause_of_death = "survived"
            try:
                mesh.run(max_ticks=self.ticks_per_life)
            except GhostDeathException as exc:
                cause_of_death = type(exc).__name__
            except Exception as exc:  # noqa: BLE001
                cause_of_death = type(exc).__name__
                print(
                    f"    [WARNING] Unexpected exception in life {gen} of {config_id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

            # Build lineage record from this life
            lineage = self._build_lineage(mesh, cause_of_death, prev_lineage, gen)
            tracker.record(lineage)

            rec = self._make_generation_record(
                config_id=config_id,
                stressor_prob=stressor_prob,
                stressor_intensity=stressor_intensity,
                immortal=False,
                run_idx=run_idx,
                generation=gen,
                mesh=mesh,
                cause_of_death=cause_of_death,
                tracker=tracker,
            )
            records.append(rec)

            # Carry death memory and lineage forward
            death_memory = f"{cause_of_death}: life {gen + 1}"
            prev_lineage = lineage

        return records

    # ── Immortal run: one long life with in-place revival on death ────────────

    def _run_immortal(
        self,
        *,
        config_id: str,
        stressor_prob: float,
        stressor_intensity: float,
        run_idx: int,
        tracker: LineageTracker,
        rng: random.Random,
        tmpdir: str,
    ) -> list[GenerationRecord]:
        """Run a single immortal organism for n_generations × ticks_per_life ticks.

        Whenever a GhostDeathException is raised the organism's vitals are
        revived in-place (no new organism created) and the run continues.
        This models a "no mortality pressure" baseline for comparison.

        The immortal run is sliced into n_generations equal windows for
        compatibility with the mortal output format, letting callers compare
        plasticity_index at equivalent generational depth / tick counts.
        """
        total_ticks = self.n_generations * self.ticks_per_life
        life_seed = rng.randint(0, 2**31)
        mesh = self._build_mesh(
            life_number=1,
            death_memory=None,
            seed=life_seed,
            stressor_prob=stressor_prob,
            stressor_intensity=stressor_intensity,
            tmpdir=tmpdir,
        )

        records: list[GenerationRecord] = []
        ticks_run = 0
        revival_count = 0

        # We drive the organism tick-by-tick directly so we can intercept
        # death exceptions and revive in-place without rebuilding the object.
        for gen in range(self.n_generations):
            window_start_tick = mesh.state.entropy
            window_ticks = 0
            while window_ticks < self.ticks_per_life:
                try:
                    mesh._pulse()  # type: ignore[attr-defined]
                    window_ticks += 1
                    ticks_run += 1
                except GhostDeathException:
                    # Immortal revival: inject vitals to pull back from death
                    revival_count += 1
                    mesh.state.apply_action_feedback(
                        delta_energy=_IMMORTAL_REVIVAL_ENERGY,
                        delta_heat=_IMMORTAL_REVIVAL_HEAT_DRAIN,
                        delta_waste=_IMMORTAL_REVIVAL_WASTE_DRAIN,
                        delta_integrity=_IMMORTAL_REVIVAL_INTEGRITY_BOOST,
                        delta_stability=_IMMORTAL_REVIVAL_STABILITY_BOOST,
                    )
                    window_ticks += 1
                    ticks_run += 1
                except Exception as exc:  # noqa: BLE001
                    # Non-death exception — treat as survived segment and log
                    print(
                        f"    [WARNING] Unexpected exception in immortal window {gen} "
                        f"of {config_id}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    window_ticks += 1
                    ticks_run += 1

            # Snapshot this window as a pseudo-generation record
            cause = "survived" if revival_count == 0 else f"revived×{revival_count}"
            rec = self._make_generation_record(
                config_id=config_id,
                stressor_prob=stressor_prob,
                stressor_intensity=stressor_intensity,
                immortal=True,
                run_idx=run_idx,
                generation=gen,
                mesh=mesh,
                cause_of_death=cause,
                tracker=tracker,
                lifespan_override=mesh.state.entropy - window_start_tick,
            )
            records.append(rec)
            # Add a dummy lineage record so plasticity_selection_r can be computed
            lineage = self._build_lineage(mesh, cause, None, gen)
            tracker.record(lineage)

        return records

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_mesh(
        self,
        *,
        life_number: int,
        death_memory: str | None,
        seed: int,
        stressor_prob: float,
        stressor_intensity: float,
        tmpdir: str,
    ) -> GhostMesh:
        """Construct a GhostMesh with experiment-specific env overrides."""
        env_overrides: dict[str, str] = {
            "GHOST_HUD": "0",
            "GHOST_PULSE": "0",
            "GHOST_STATE_FILE": os.path.join(tmpdir, f"state_{life_number}.json"),
            "GHOST_DIARY_PATH": os.path.join(tmpdir, f"diary_{life_number}.db"),
            "GHOST_VITALS_LOG": "",
            "GHOST_LOG_FILE": "",
            "GHOST_ENV_EVENTS": "1",
            "GHOST_USE_LLM": "0",
            "GHOST_PURITY_MODE": "0",
            "GHOST_STRESSOR_PROB": str(stressor_prob),
            "GHOST_STRESSOR_INTENSITY": str(stressor_intensity),
            "GHOST_STRESSOR_MODE": "hostile_windows" if stressor_prob > 0 else "flat",
        }
        saved: dict[str, str | None] = {}
        for k, v in env_overrides.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v

        try:
            mesh = GhostMesh(
                seed=seed,
                death_memory=death_memory,
                life_number=life_number,
            )
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        return mesh

    def _build_lineage(
        self,
        mesh: GhostMesh,
        cause_of_death: str,
        parent: Lineage | None,
        generation: int,
    ) -> Lineage:
        """Build a Lineage record from a completed (or windowed) GhostMesh run."""
        snap = mesh._last_collapse_snapshot  # type: ignore[attr-defined]
        dreamer_fraction = snap.dreamer_fraction if snap else 0.0
        guardian_fraction = snap.guardian_fraction if snap else 0.0
        plasticity_index = snap.plasticity_index if snap else 0.0

        q_table: dict = {}
        if mesh.learner is not None:
            q_table = mesh.learner._q_table  # type: ignore[attr-defined]

        return Lineage(
            lineage_id=_generate_lineage_id(),
            parent_id=parent.lineage_id if parent else None,
            life_number=generation + 1,
            lifespan=mesh.state.entropy,
            interiority_score=mesh.meta_self.interiority_score(),
            dreamer_fraction=dreamer_fraction,
            guardian_fraction=guardian_fraction,
            plasticity_index=plasticity_index,
            cause_of_death=cause_of_death,
            top_q_entries=extract_top_q_entries(q_table),
            mask_preferences={},
            mutation_rate=DEFAULT_MUTATION_RATE,
        )

    def _make_generation_record(
        self,
        *,
        config_id: str,
        stressor_prob: float,
        stressor_intensity: float,
        immortal: bool,
        run_idx: int,
        generation: int,
        mesh: GhostMesh,
        cause_of_death: str,
        tracker: LineageTracker,
        lifespan_override: int | None = None,
    ) -> GenerationRecord:
        """Assemble a GenerationRecord from a completed GhostMesh life."""
        snap = mesh._last_collapse_snapshot  # type: ignore[attr-defined]
        dreamer_fraction = snap.dreamer_fraction if snap else 0.0
        guardian_fraction = snap.guardian_fraction if snap else 0.0
        plasticity_index = snap.plasticity_index if snap else 0.0
        lifespan = lifespan_override if lifespan_override is not None else mesh.state.entropy

        # Fitness scores use the tracker's current full history
        fitness_scores = tracker.lineage_fitness()
        fitness_score = fitness_scores[-1] if fitness_scores else 0.0
        plasticity_r = tracker.plasticity_selection_signal()

        return GenerationRecord(
            config_id=config_id,
            stressor_prob=stressor_prob,
            stressor_intensity=stressor_intensity,
            immortal=immortal,
            run=run_idx,
            generation=generation,
            lifespan=lifespan,
            dreamer_fraction=dreamer_fraction,
            guardian_fraction=guardian_fraction,
            plasticity_index=plasticity_index,
            interiority_score=mesh.meta_self.interiority_score(),
            n_awakenings=len(mesh.awakening_history),  # type: ignore[attr-defined]
            cause_of_death=cause_of_death,
            fitness_score=fitness_score,
            plasticity_selection_r=plasticity_r,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GhostMesh Mortality Lineage Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-generations", type=int, default=20,
        help="Successive lives per configuration run (mortal); "
             "also sets the number of equal windows in immortal mode",
    )
    p.add_argument(
        "--ticks-per-life", type=int, default=1000,
        help="Max ticks per life (mortal) or per immortal window",
    )
    p.add_argument(
        "--runs", type=int, default=3,
        help="Independent runs per (stressor_prob, stressor_intensity, immortal) triple",
    )
    p.add_argument(
        "--out", default="results",
        help="Output directory for JSONL and CSV files",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed",
    )
    p.add_argument(
        "--stressor-probs", type=float, nargs="+", default=None,
        metavar="P",
        help="Override stressor probability grid (e.g. 0.0 0.1 0.3 0.5)",
    )
    p.add_argument(
        "--stressor-intensities", type=float, nargs="+", default=None,
        metavar="I",
        help="Override stressor intensity grid (e.g. 1.0 2.0 3.0)",
    )
    p.add_argument(
        "--no-immortal", action="store_true",
        help="Skip immortal baseline configurations",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exp = MortalityLineageExperiment(
        n_generations=args.n_generations,
        ticks_per_life=args.ticks_per_life,
        runs_per_config=args.runs,
        output_dir=args.out,
        seed_base=args.seed,
        stressor_probs=args.stressor_probs,
        stressor_intensities=args.stressor_intensities,
        include_immortal=not args.no_immortal,
    )
    jsonl_path, csv_path = exp.run()
    print(f"\nJSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")
