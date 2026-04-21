#!/usr/bin/env python3
"""plot_lineage_experiment.py — visualise mortality_lineage_experiment results.

Reads results/lineage_experiment.csv and produces four plots:

  (a) Plasticity erosion: dreamer_fraction vs generational depth,
      one curve per stressor_prob (mortal), with immortal overlay.
  (b) Mean lifespan curves: mortal vs immortal at each stressor level.
  (c) Sweet-spot heatmap: mean plasticity_index over
      (stressor_prob × stressor_intensity) for mortal runs.
  (d) Plasticity selection signal (Pearson r) vs stressor level —
      does plasticity in generation t predict lifespan in t+1?

Usage
-----
    python scripts/plot_lineage_experiment.py [--csv results/lineage_experiment.csv]
                                              [--out results/]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "matplotlib is not installed.  Install with:\n"
            "    pip install matplotlib",
            file=sys.stderr,
        )
        sys.exit(1)


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _cast(row: dict) -> dict:
    return {
        "config_id": row["config_id"],
        "stressor_prob": float(row["stressor_prob"]),
        "stressor_intensity": float(row["stressor_intensity"]),
        "immortal": row["immortal"].lower() in ("true", "1", "yes"),
        "run": int(row["run"]),
        "generation": int(row["generation"]),
        "lifespan": int(row["lifespan"]),
        "dreamer_fraction": float(row["dreamer_fraction"]),
        "guardian_fraction": float(row["guardian_fraction"]),
        "plasticity_index": float(row["plasticity_index"]),
        "interiority_score": float(row["interiority_score"]),
        "n_awakenings": int(row["n_awakenings"]),
        "cause_of_death": row["cause_of_death"],
        "fitness_score": float(row["fitness_score"]),
        "plasticity_selection_r": float(row["plasticity_selection_r"]),
    }


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def plot_lineage(csv_path: Path, output_dir: Path) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_cast(r) for r in _load_csv(csv_path)]
    if not rows:
        print("CSV is empty — nothing to plot.", file=sys.stderr)
        return

    mortal_rows = [r for r in rows if not r["immortal"]]
    immortal_rows = [r for r in rows if r["immortal"]]
    max_gen = max(r["generation"] for r in rows) if rows else 0

    probs = sorted({r["stressor_prob"] for r in rows})
    intensities = sorted({r["stressor_intensity"] for r in rows})

    # ── (a) Plasticity erosion: dreamer_fraction vs generation ───────────────
    fig_a, ax_a = plt.subplots(figsize=(11, 6))
    colours = cm.plasma(np.linspace(0.1, 0.9, len(probs)))

    for colour, prob in zip(colours, probs):
        # Mortal: mean dreamer_fraction per generation across runs and intensities
        gen_dreamer: dict[int, list[float]] = defaultdict(list)
        for r in mortal_rows:
            if r["stressor_prob"] == prob:
                gen_dreamer[r["generation"]].append(r["dreamer_fraction"])
        gens = sorted(gen_dreamer)
        if not gens:
            continue
        means = [_mean(gen_dreamer[g]) for g in gens]
        ax_a.plot(gens, means, "-o", color=colour, label=f"mortal p={prob}", linewidth=2,
                  markersize=5, alpha=0.9)

    # Immortal overlay (mean across all stressor settings)
    if immortal_rows:
        gen_dreamer_imm: dict[int, list[float]] = defaultdict(list)
        for r in immortal_rows:
            gen_dreamer_imm[r["generation"]].append(r["dreamer_fraction"])
        gens_imm = sorted(gen_dreamer_imm)
        means_imm = [_mean(gen_dreamer_imm[g]) for g in gens_imm]
        ax_a.plot(gens_imm, means_imm, "--k", linewidth=2.5, label="immortal (pooled)", alpha=0.7)

    ax_a.axhline(0.5, color="green", linestyle=":", alpha=0.5, label="plastic threshold (0.5)")
    ax_a.axhline(0.25, color="red", linestyle=":", alpha=0.5, label="guardian threshold (0.25)")
    ax_a.set_xlabel("Generation", fontsize=12)
    ax_a.set_ylabel("Mean Dreamer Fraction", fontsize=12)
    ax_a.set_title(
        "Plasticity Erosion (or Rescue) vs Generational Depth\n"
        "by Stressor Probability (mortal) vs Immortal Baseline",
        fontsize=12,
    )
    ax_a.legend(fontsize=9, loc="upper right")
    ax_a.grid(alpha=0.3)
    ax_a.set_xlim(left=0)
    ax_a.set_ylim(0, 1)
    fig_a.tight_layout()
    path_a = output_dir / "lineage_plasticity_erosion.png"
    fig_a.savefig(path_a, dpi=120)
    plt.close(fig_a)
    print(f"  Saved: {path_a}")

    # ── (b) Mean lifespan curves: mortal vs immortal at each stressor prob ────
    fig_b, ax_b = plt.subplots(figsize=(10, 6))
    for colour, prob in zip(colours, probs):
        # Mortal: mean lifespan per generation
        gen_ls: dict[int, list[float]] = defaultdict(list)
        for r in mortal_rows:
            if r["stressor_prob"] == prob:
                gen_ls[r["generation"]].append(r["lifespan"])
        gens = sorted(gen_ls)
        if not gens:
            continue
        means_ls = [_mean(gen_ls[g]) for g in gens]
        ax_b.plot(gens, means_ls, "-o", color=colour, label=f"mortal p={prob}",
                  linewidth=2, markersize=5, alpha=0.9)

    if immortal_rows:
        gen_ls_imm: dict[int, list[float]] = defaultdict(list)
        for r in immortal_rows:
            gen_ls_imm[r["generation"]].append(r["lifespan"])
        gens_imm = sorted(gen_ls_imm)
        means_ls_imm = [_mean(gen_ls_imm[g]) for g in gens_imm]
        ax_b.plot(gens_imm, means_ls_imm, "--k", linewidth=2.5,
                  label="immortal window length", alpha=0.7)

    ax_b.set_xlabel("Generation", fontsize=12)
    ax_b.set_ylabel("Mean Lifespan (ticks)", fontsize=12)
    ax_b.set_title(
        "Mean Lifespan per Generation\nMortal (natural death) vs Immortal (revival baseline)",
        fontsize=12,
    )
    ax_b.legend(fontsize=9)
    ax_b.grid(alpha=0.3)
    ax_b.set_xlim(left=0)
    fig_b.tight_layout()
    path_b = output_dir / "lineage_lifespan_curves.png"
    fig_b.savefig(path_b, dpi=120)
    plt.close(fig_b)
    print(f"  Saved: {path_b}")

    # ── (c) Sweet-spot heatmap: mean plasticity_index ─────────────────────────
    heatmap = np.full((len(intensities), len(probs)), np.nan)
    count_map = np.zeros_like(heatmap)
    for r in mortal_rows:
        i = intensities.index(r["stressor_intensity"])
        j = probs.index(r["stressor_prob"])
        if np.isnan(heatmap[i, j]):
            heatmap[i, j] = 0.0
        heatmap[i, j] += r["plasticity_index"]
        count_map[i, j] += 1

    with np.errstate(invalid="ignore"):
        heatmap = np.where(count_map > 0, heatmap / count_map, np.nan)

    fig_c, ax_c = plt.subplots(figsize=(9, 6))
    im = ax_c.imshow(heatmap, cmap="RdYlGn", vmin=0.0, vmax=2.0,
                     aspect="auto", origin="lower")
    ax_c.set_xticks(range(len(probs)))
    ax_c.set_xticklabels([str(p) for p in probs])
    ax_c.set_yticks(range(len(intensities)))
    ax_c.set_yticklabels([str(s) for s in intensities])
    ax_c.set_xlabel("Stressor Probability", fontsize=12)
    ax_c.set_ylabel("Stressor Intensity", fontsize=12)
    ax_c.set_title(
        "Mean Plasticity Index Heatmap (mortal runs)\n"
        "green = plastic (Dreamer); red = calcified (Guardian)",
        fontsize=12,
    )
    plt.colorbar(im, ax=ax_c, label="mean plasticity_index")
    for i in range(len(intensities)):
        for j in range(len(probs)):
            v = heatmap[i, j]
            if not np.isnan(v):
                ax_c.text(j, i, f"{v:.2f}", ha="center", va="center",
                          fontsize=9, color="white" if v < 0.4 or v > 1.6 else "black")
    fig_c.tight_layout()
    path_c = output_dir / "lineage_sweet_spot_heatmap.png"
    fig_c.savefig(path_c, dpi=120)
    plt.close(fig_c)
    print(f"  Saved: {path_c}")

    # ── (d) Plasticity selection signal vs stressor_prob ─────────────────────
    fig_d, ax_d = plt.subplots(figsize=(9, 5))
    # For each stressor_prob take the last recorded plasticity_selection_r per run
    prob_r: dict[float, list[float]] = defaultdict(list)
    for prob in probs:
        # Get the final-generation record from each mortal run
        for run_id in {r["run"] for r in mortal_rows if r["stressor_prob"] == prob}:
            run_records = [
                r for r in mortal_rows
                if r["stressor_prob"] == prob and r["run"] == run_id
            ]
            if run_records:
                last = max(run_records, key=lambda r: r["generation"])
                prob_r[prob].append(last["plasticity_selection_r"])

    prob_means = [_mean(prob_r[p]) for p in probs]
    # Error bars: std
    prob_stds = []
    for p in probs:
        vals = prob_r[p]
        if len(vals) > 1:
            mean = _mean(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            prob_stds.append(std)
        else:
            prob_stds.append(0.0)

    ax_d.bar(range(len(probs)), prob_means, yerr=prob_stds, capsize=5,
             color=[cm.plasma(0.3 + 0.5 * i / max(len(probs) - 1, 1)) for i in range(len(probs))],
             alpha=0.8, edgecolor="black", linewidth=0.8)
    ax_d.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax_d.set_xticks(range(len(probs)))
    ax_d.set_xticklabels([f"p={p}" for p in probs])
    ax_d.set_xlabel("Stressor Probability", fontsize=12)
    ax_d.set_ylabel("Plasticity Selection Signal\n(Pearson r: dreamer[t] → lifespan[t+1])",
                    fontsize=11)
    ax_d.set_title(
        "Does Plasticity in Generation t Predict Longer Survival in t+1?\n"
        "(positive r = plasticity is adaptive; negative r = plasticity is costly)",
        fontsize=11,
    )
    ax_d.grid(alpha=0.3, axis="y")
    ax_d.set_ylim(-1, 1)
    fig_d.tight_layout()
    path_d = output_dir / "lineage_selection_signal.png"
    fig_d.savefig(path_d, dpi=120)
    plt.close(fig_d)
    print(f"  Saved: {path_d}")

    print(f"\nAll plots written to {output_dir}/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot GhostMesh mortality lineage experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv", default="results/lineage_experiment.csv",
        help="Path to lineage_experiment.csv",
    )
    p.add_argument(
        "--out", default="results",
        help="Output directory for PNG plots",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        print(
            "Run mortality_lineage_experiment.py first:\n"
            "    python scripts/mortality_lineage_experiment.py --n-generations 10 --ticks-per-life 500",
            file=sys.stderr,
        )
        sys.exit(1)
    plot_lineage(csv_path, Path(args.out))
