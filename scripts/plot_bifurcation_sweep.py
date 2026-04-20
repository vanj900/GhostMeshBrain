#!/usr/bin/env python3
"""plot_bifurcation_sweep.py — visualise the results of bifurcation_experiment.py

Reads results/bifurcation_sweep.csv and produces three plots:

  (a) plasticity_index over lifespan for each parameter setting
  (b) lifespan distribution split by genesis_prior_strength type
  (c) guardian_fraction heatmap (precision_bound × genesis_prior_strength)

Usage
-----
    python scripts/plot_bifurcation_sweep.py [--csv results/bifurcation_sweep.csv]
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
    """Convert string CSV values to appropriate Python types."""
    return {
        "config_id": row["config_id"],
        "genesis_prior_strength": float(row["genesis_prior_strength"]),
        "precision_bound": float(row["precision_bound"]),
        "llm_usage_freq": row["llm_usage_freq"],
        "run": int(row["run"]),
        "plasticity_index": float(row["plasticity_index"]),
        "lifespan": int(row["lifespan"]),
        "interiority_score": float(row["interiority_score"]),
        "dreamer_fraction": float(row["dreamer_fraction"]),
        "guardian_fraction": float(row["guardian_fraction"]),
        "n_awakenings": int(row["n_awakenings"]),
        "pre_collapse_score": float(row["pre_collapse_score"]),
        "cause_of_death": row["cause_of_death"],
    }


def plot_sweep(csv_path: Path, output_dir: Path) -> None:
    """Generate all three plots from the sweep CSV.

    Parameters
    ----------
    csv_path:
        Path to the bifurcation_sweep.csv file.
    output_dir:
        Directory to write PNG files into.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_cast(r) for r in _load_csv(csv_path)]
    if not rows:
        print("CSV is empty — nothing to plot.", file=sys.stderr)
        return

    # ── (a) Plasticity Index vs Lifespan per config ───────────────────────
    fig_a, ax_a = plt.subplots(figsize=(10, 6))
    configs = sorted({r["config_id"] for r in rows})
    colours = cm.tab20(np.linspace(0, 1, len(configs)))
    for colour, config_id in zip(colours, configs):
        config_rows = [r for r in rows if r["config_id"] == config_id]
        xs = [r["lifespan"] for r in config_rows]
        ys = [r["plasticity_index"] for r in config_rows]
        ax_a.scatter(xs, ys, label=config_id, alpha=0.7, s=60, color=colour)
    ax_a.set_xlabel("Lifespan (ticks)", fontsize=12)
    ax_a.set_ylabel("Plasticity Index (dreamer / (guardian + ε))", fontsize=12)
    ax_a.set_title("Plasticity Index vs Lifespan by Parameter Configuration", fontsize=13)
    ax_a.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, ncol=2)
    ax_a.grid(alpha=0.3)
    # Add threshold lines
    ax_a.axhline(0.5, color="green", linestyle="--", alpha=0.5, label="Plastic threshold (0.5)")
    ax_a.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="Guardian threshold (0.3)")
    fig_a.tight_layout()
    path_a = output_dir / "bifurcation_plasticity_vs_lifespan.png"
    fig_a.savefig(path_a, dpi=120)
    plt.close(fig_a)
    print(f"  Saved: {path_a}")

    # ── (b) Lifespan distribution by genesis_prior_strength ──────────────
    fig_b, ax_b = plt.subplots(figsize=(10, 6))
    gps_values = sorted({r["genesis_prior_strength"] for r in rows})
    lifespan_by_gps: dict[float, list[int]] = defaultdict(list)
    for r in rows:
        lifespan_by_gps[r["genesis_prior_strength"]].append(r["lifespan"])

    for i, gps in enumerate(gps_values):
        ax_b.hist(
            lifespan_by_gps[gps],
            bins=20,
            alpha=0.6,
            label=f"genesis_prior_strength={gps}",
            density=True,
        )
    ax_b.set_xlabel("Lifespan (ticks)", fontsize=12)
    ax_b.set_ylabel("Density", fontsize=12)
    ax_b.set_title("Lifespan Distribution by Genesis Prior Strength", fontsize=13)
    ax_b.legend(fontsize=10)
    ax_b.grid(alpha=0.3)
    fig_b.tight_layout()
    path_b = output_dir / "bifurcation_lifespan_by_prior.png"
    fig_b.savefig(path_b, dpi=120)
    plt.close(fig_b)
    print(f"  Saved: {path_b}")

    # ── (c) Guardian fraction heatmap ────────────────────────────────────
    gps_sorted = sorted({r["genesis_prior_strength"] for r in rows})
    pb_sorted = sorted({r["precision_bound"] for r in rows})
    heatmap: np.ndarray = np.zeros((len(pb_sorted), len(gps_sorted)))
    count_map: np.ndarray = np.zeros_like(heatmap)

    for r in rows:
        i = pb_sorted.index(r["precision_bound"])
        j = gps_sorted.index(r["genesis_prior_strength"])
        heatmap[i, j] += r["guardian_fraction"]
        count_map[i, j] += 1

    with np.errstate(invalid="ignore"):
        heatmap = np.where(count_map > 0, heatmap / count_map, np.nan)

    fig_c, ax_c = plt.subplots(figsize=(8, 6))
    im = ax_c.imshow(
        heatmap,
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        origin="lower",
    )
    ax_c.set_xticks(range(len(gps_sorted)))
    ax_c.set_xticklabels([str(g) for g in gps_sorted])
    ax_c.set_yticks(range(len(pb_sorted)))
    ax_c.set_yticklabels([str(p) for p in pb_sorted])
    ax_c.set_xlabel("genesis_prior_strength", fontsize=12)
    ax_c.set_ylabel("precision_bound", fontsize=12)
    ax_c.set_title("Mean Guardian Fraction Heatmap\n(red = Guardian attractor; green = plastic)", fontsize=12)
    plt.colorbar(im, ax=ax_c, label="guardian_fraction")

    # Annotate cells
    for i in range(len(pb_sorted)):
        for j in range(len(gps_sorted)):
            v = heatmap[i, j]
            if not np.isnan(v):
                ax_c.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                          color="white" if v > 0.65 or v < 0.35 else "black")

    fig_c.tight_layout()
    path_c = output_dir / "bifurcation_guardian_heatmap.png"
    fig_c.savefig(path_c, dpi=120)
    plt.close(fig_c)
    print(f"  Saved: {path_c}")

    print(f"\nAll plots written to {output_dir}/")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GhostMesh bifurcation sweep results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default="results/bifurcation_sweep.csv",
        help="Path to the bifurcation_sweep.csv file",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="Output directory for PNG plots",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        print(
            "Run bifurcation_experiment.py first:\n"
            "    python scripts/bifurcation_experiment.py --ticks 500",
            file=sys.stderr,
        )
        sys.exit(1)
    plot_sweep(csv_path, Path(args.out))
