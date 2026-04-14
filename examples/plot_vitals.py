#!/usr/bin/env python3
"""plot_vitals.py — plot GhostMesh vital signs from a stress-test JSONL log.

Requires matplotlib (``pip install matplotlib``).

Usage
-----
    python examples/plot_vitals.py /tmp/ghost_runs/vitals_cl1.0.jsonl
    python examples/plot_vitals.py /tmp/ghost_runs/vitals_cl1.0.jsonl --output chart.png
    python examples/plot_vitals.py /tmp/ghost_runs/vitals_cl1.0.jsonl --show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot(records: list[dict], output: str | None, show: bool) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        sys.exit(1)

    ticks = [r["tick"] for r in records]
    energy = [r.get("energy", 0) for r in records]
    heat = [r.get("heat", 0) for r in records]
    waste = [r.get("waste", 0) for r in records]
    integrity = [r.get("integrity", 0) for r in records]
    stability = [r.get("stability", 0) for r in records]
    affect = [r.get("affect", 0) for r in records]
    free_energy = [r.get("free_energy", 0) for r in records]
    health = [r.get("health", 0) for r in records]

    actions = [r.get("action", "") for r in records]
    events = [r.get("env_event", "none") for r in records]

    # Colour bands for action tokens
    action_colours = {
        "FORAGE": "#e67e22",
        "REST": "#3498db",
        "REPAIR": "#9b59b6",
        "DECIDE": "#27ae60",
    }

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"GhostMesh Vital Signs  ·  {len(ticks)} ticks",
        fontsize=14, fontweight="bold",
    )

    def _shade_actions(ax):
        """Shade background by action token."""
        if not ticks:
            return
        prev_action = actions[0]
        start = ticks[0]
        for i, (t, a) in enumerate(zip(ticks, actions)):
            if a != prev_action:
                colour = action_colours.get(prev_action, "#cccccc")
                ax.axvspan(start, t, alpha=0.07, color=colour, linewidth=0)
                start = t
                prev_action = a
        # Close the final span
        colour = action_colours.get(prev_action, "#cccccc")
        ax.axvspan(start, ticks[-1], alpha=0.07, color=colour, linewidth=0)

    # ── Plot 1: Energy / Health ─────────────────────────────────────────
    ax = axes[0]
    _shade_actions(ax)
    ax.plot(ticks, energy, color="#27ae60", label="Energy (E)", linewidth=1.0)
    ax.plot(ticks, health, color="#1abc9c", label="Health Score", linewidth=1.0, linestyle="--")
    ax.axhline(25, color="#e67e22", linestyle=":", linewidth=0.8, alpha=0.6, label="FORAGE threshold")
    ax.axhline(0, color="#c0392b", linestyle="-", linewidth=1.0, alpha=0.5, label="Death (E=0)")
    ax.set_ylabel("Energy / Health")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # ── Plot 2: Heat / Waste ────────────────────────────────────────────
    ax = axes[1]
    _shade_actions(ax)
    ax.plot(ticks, heat, color="#e74c3c", label="Heat (T)", linewidth=1.0)
    ax.plot(ticks, waste, color="#f39c12", label="Waste (W)", linewidth=1.0)
    ax.axhline(80, color="#e74c3c", linestyle=":", linewidth=0.8, alpha=0.6, label="REST(heat) threshold")
    ax.axhline(75, color="#f39c12", linestyle=":", linewidth=0.8, alpha=0.6, label="REST(waste) threshold")
    ax.axhline(100, color="#c0392b", linestyle="-", linewidth=1.0, alpha=0.5, label="ThermalDeath")
    ax.set_ylabel("Heat / Waste")
    ax.set_ylim(-2, 115)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # ── Plot 3: Integrity / Stability ──────────────────────────────────
    ax = axes[2]
    _shade_actions(ax)
    ax.plot(ticks, integrity, color="#8e44ad", label="Integrity (M)", linewidth=1.0)
    ax.plot(ticks, stability, color="#2980b9", label="Stability (S)", linewidth=1.0)
    ax.axhline(45, color="#8e44ad", linestyle=":", linewidth=0.8, alpha=0.6, label="REPAIR(integrity)")
    ax.axhline(40, color="#2980b9", linestyle=":", linewidth=0.8, alpha=0.6, label="REPAIR(stability)")
    ax.axhline(10, color="#c0392b", linestyle="-", linewidth=1.0, alpha=0.5, label="MemoryCollapse")
    ax.axhline(0, color="#c0392b", linestyle="--", linewidth=1.0, alpha=0.5, label="EntropyDeath")
    ax.set_ylabel("Integrity / Stability")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # ── Plot 4: Affect / Free Energy ───────────────────────────────────
    ax = axes[3]
    _shade_actions(ax)
    ax_fe = ax.twinx()
    ax.fill_between(ticks, affect, 0, where=[a >= 0 for a in affect],
                    alpha=0.4, color="#27ae60", label="Positive affect")
    ax.fill_between(ticks, affect, 0, where=[a < 0 for a in affect],
                    alpha=0.4, color="#c0392b", label="Negative affect")
    ax.plot(ticks, affect, color="#555", linewidth=0.6, alpha=0.6)
    ax.axhline(0, color="#888", linewidth=0.5)
    ax.set_ylabel("Affect", color="#555")
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("Tick")

    ax_fe.plot(ticks, free_energy, color="#e67e22", linewidth=0.8, alpha=0.7, label="Free Energy (FE)")
    ax_fe.axhline(8, color="#3498db", linestyle=":", linewidth=0.7, alpha=0.7, label="Sweet-spot lower (FE=8)")
    ax_fe.axhline(45, color="#e74c3c", linestyle=":", linewidth=0.7, alpha=0.7, label="Overload threshold (FE=45)")
    ax_fe.set_ylabel("Free Energy", color="#e67e22")
    ax_fe.set_ylim(-2, max(50, max(free_energy) + 5) if free_energy else 55)

    lines_affect = [
        mpatches.Patch(color="#27ae60", alpha=0.6, label="Positive affect"),
        mpatches.Patch(color="#c0392b", alpha=0.6, label="Negative affect"),
    ]
    lines_fe = [
        mpatches.Patch(color="#e67e22", alpha=0.6, label="Free Energy"),
    ]
    ax.legend(handles=lines_affect, loc="upper left", fontsize=7)
    ax_fe.legend(handles=lines_fe, loc="upper right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # ── Legend for action-token shading ────────────────────────────────
    action_patches = [
        mpatches.Patch(color=c, alpha=0.25, label=a)
        for a, c in action_colours.items()
    ]
    fig.legend(
        handles=action_patches,
        loc="lower center",
        ncol=4,
        fontsize=8,
        title="Action token (background shading)",
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"[plot_vitals] Saved chart to {output}")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GhostMesh vitals from JSONL log.")
    parser.add_argument("log", help="Path to a JSONL vitals log from stress_test.py")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save chart to this path (PNG/PDF/SVG). Defaults to <log>.png",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the chart interactively (requires a display)",
    )
    args = parser.parse_args()

    if not Path(args.log).exists():
        print(f"Error: log file not found: {args.log}")
        sys.exit(1)

    records = load_jsonl(args.log)
    if not records:
        print("Error: log file is empty")
        sys.exit(1)

    output = args.output or str(Path(args.log).with_suffix(".png"))
    plot(records, output=output, show=args.show)


if __name__ == "__main__":
    main()
