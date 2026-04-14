#!/usr/bin/env python3
"""Analyze a GhostMesh run log and produce vital-sign plots / statistics.

Reads a JSONL file produced by ``examples/stress_test.py`` (one JSON
object per line, each representing one tick) and:

1. Always prints a text summary (action distribution, mask distribution,
   affect/FE/health stats, ethics blocks, stressor events).
2. If ``matplotlib`` is installed, renders a multi-panel plot showing:
   - Vital signs trajectory (energy, heat, waste, integrity, stability)
   - Affect signal over time
   - Free-energy trajectory
   - Action token distribution (bar chart)
   - Personality mask distribution (pie chart)

Usage
-----
    # Text summary only
    python examples/analyze_run.py /tmp/ghost_run.jsonl --no-plot

    # Text summary + interactive plot
    python examples/analyze_run.py /tmp/ghost_run.jsonl

    # Save plot to file
    python examples/analyze_run.py /tmp/ghost_run.jsonl --output /tmp/plots/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running directly without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ #
# Data loading                                                         #
# ------------------------------------------------------------------ #

def load_records(path: str) -> list[dict]:
    """Load all tick records from a JSONL file."""
    records: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ------------------------------------------------------------------ #
# Text summary                                                         #
# ------------------------------------------------------------------ #

def print_text_summary(records: list[dict]) -> None:
    n = len(records)
    if not n:
        print("No records found in log file.")
        return

    action_counts: dict[str, int] = {}
    mask_counts: dict[str, int] = {}
    regime_counts: dict[str, int] = {}
    for r in records:
        action_counts[r.get("action", "?")] = (
            action_counts.get(r.get("action", "?"), 0) + 1
        )
        mask_counts[r.get("mask", "?")] = (
            mask_counts.get(r.get("mask", "?"), 0) + 1
        )
        regime = r.get("precision_regime", "")
        if regime:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

    affects = [r.get("affect", 0.0) for r in records]
    fes = [r.get("free_energy", 0.0) for r in records]
    healths = [r.get("health_score", 0.0) for r in records]

    print(f"\n{'='*62}")
    print(f"  GhostMesh Run Analysis — {n} ticks")
    print(f"{'='*62}")
    print(f"  First tick : {records[0]['tick']}")
    print(f"  Final tick : {records[-1]['tick']}")
    print(f"  Final stage: {records[-1].get('stage', '?')}")

    print("\n  Action Token Distribution:")
    for action in ("DECIDE", "FORAGE", "REST", "REPAIR"):
        count = action_counts.get(action, 0)
        pct = count / n * 100
        bar = "█" * int(pct / 2)
        print(f"    {action:<8} {bar:<25} {count:5d}  ({pct:5.1f}%)")

    print("\n  Personality Mask Distribution:")
    for mask, count in sorted(mask_counts.items(), key=lambda x: -x[1]):
        pct = count / n * 100
        print(f"    {mask:<12} {count:5d}  ({pct:5.1f}%)")

    if regime_counts:
        print("\n  Precision Regime Distribution:")
        for regime, count in sorted(regime_counts.items()):
            pct = count / n * 100
            print(f"    {regime:<14} {count:5d}  ({pct:5.1f}%)")

    print(
        f"\n  Affect (pleasure signal) : "
        f"mean={sum(affects)/n:+.3f}  "
        f"min={min(affects):+.3f}  "
        f"max={max(affects):+.3f}"
    )
    print(
        f"  Free Energy (surprise)   : "
        f"mean={sum(fes)/n:.3f}  "
        f"min={min(fes):.3f}  "
        f"max={max(fes):.3f}"
    )
    print(
        f"  Health Score             : "
        f"mean={sum(healths)/n:.1f}  "
        f"min={min(healths):.1f}  "
        f"max={max(healths):.1f}"
    )

    ethics_blocks = sum(r.get("ethics_blocks", 0) for r in records)
    stressor_events = sum(1 for r in records if r.get("stressor_event"))
    stressor_pct = stressor_events / n * 100
    print(f"\n  Ethics blocks   : {ethics_blocks}")
    print(f"  Stressor events : {stressor_events}  ({stressor_pct:.1f}% of ticks)")

    # Sample a few interesting stressor events
    sample_events = [
        f"    tick {r['tick']:>5}: {r['stressor_event']}"
        for r in records
        if r.get("stressor_event")
    ][:8]
    if sample_events:
        print("  Stressor sample :")
        for ev in sample_events:
            print(ev)

    print()


# ------------------------------------------------------------------ #
# Matplotlib plots                                                     #
# ------------------------------------------------------------------ #

def plot_with_matplotlib(records: list[dict], output_dir: str | None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n = len(records)
    ticks = [r["tick"] for r in records]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"GhostMesh Run Analysis  ({n} ticks  ·  final stage: {records[-1].get('stage','?')})",
        fontsize=13,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)

    # ── Vital Signs ───────────────────────────────────────────────────
    ax_vitals = fig.add_subplot(gs[0, :])
    vital_cfg = [
        ("energy",    "#2ecc71", "Energy (E)"),
        ("heat",      "#e74c3c", "Heat (T)"),
        ("waste",     "#e67e22", "Waste (W)"),
        ("integrity", "#3498db", "Integrity (M)"),
        ("stability", "#9b59b6", "Stability (S)"),
    ]
    for key, color, label in vital_cfg:
        ax_vitals.plot(
            ticks, [r.get(key, 0.0) for r in records],
            label=label, color=color, alpha=0.85, linewidth=0.9,
        )
    ax_vitals.set_ylabel("Value (0–100)")
    ax_vitals.set_title("Vital Signs Trajectory")
    ax_vitals.legend(loc="upper right", fontsize=8, ncol=5)
    ax_vitals.set_ylim(-5, 115)
    ax_vitals.grid(True, alpha=0.25)

    # ── Affect signal ─────────────────────────────────────────────────
    ax_affect = fig.add_subplot(gs[1, 0])
    ax_affect.plot(
        ticks, [r.get("affect", 0.0) for r in records],
        color="#f39c12", alpha=0.85, linewidth=0.8,
    )
    ax_affect.axhline(y=0, color="gray", linestyle="--", linewidth=0.6)
    ax_affect.set_ylabel("Affect")
    ax_affect.set_title("Affect Signal  (+ = pleasure, − = stress)")
    ax_affect.set_ylim(-1.15, 1.15)
    ax_affect.grid(True, alpha=0.25)

    # ── Free Energy ───────────────────────────────────────────────────
    ax_fe = fig.add_subplot(gs[1, 1])
    ax_fe.plot(
        ticks, [r.get("free_energy", 0.0) for r in records],
        color="#1abc9c", alpha=0.85, linewidth=0.8,
    )
    ax_fe.set_ylabel("Free Energy")
    ax_fe.set_title("Free Energy Estimate  (surprise proxy)")
    ax_fe.grid(True, alpha=0.25)

    # ── Action distribution ───────────────────────────────────────────
    action_counts: dict[str, int] = {}
    for r in records:
        k = r.get("action", "?")
        action_counts[k] = action_counts.get(k, 0) + 1

    ax_actions = fig.add_subplot(gs[2, 0])
    action_colors = {
        "DECIDE": "#2ecc71",
        "FORAGE": "#e74c3c",
        "REST":   "#3498db",
        "REPAIR": "#f39c12",
    }
    keys = list(action_counts.keys())
    vals = [action_counts[k] for k in keys]
    colors = [action_colors.get(k, "#95a5a6") for k in keys]
    bars = ax_actions.bar(keys, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax_actions.set_title("Action Token Distribution")
    ax_actions.set_ylabel("Count")
    for bar in bars:
        ax_actions.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.01,
            str(bar.get_height()),
            ha="center", va="bottom", fontsize=9,
        )

    # ── Mask distribution (pie) ───────────────────────────────────────
    mask_counts: dict[str, int] = {}
    for r in records:
        k = r.get("mask", "?")
        mask_counts[k] = mask_counts.get(k, 0) + 1

    ax_masks = fig.add_subplot(gs[2, 1])
    mask_colors = {
        "Guardian": "#e74c3c",
        "Healer":   "#2ecc71",
        "Judge":    "#3498db",
        "Dreamer":  "#9b59b6",
        "Courier":  "#e67e22",
    }
    mk_labels = list(mask_counts.keys())
    mk_vals = [mask_counts[k] for k in mk_labels]
    mk_colors = [mask_colors.get(k, "#95a5a6") for k in mk_labels]
    ax_masks.pie(
        mk_vals,
        labels=mk_labels,
        colors=mk_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax_masks.set_title("Personality Mask Distribution")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "run_analysis.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {out_path}")
    else:
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------ #
# Entry-point                                                          #
# ------------------------------------------------------------------ #

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a GhostMesh JSONL run log",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("log_file", help="Path to JSONL log produced by stress_test.py")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Directory to save plot PNG (requires matplotlib)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plotting; show text summary only",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.log_file):
        print(f"Error: log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    records = load_records(args.log_file)
    print_text_summary(records)

    if not args.no_plot:
        try:
            plot_with_matplotlib(records, args.output)
        except ImportError:
            print("[Note] matplotlib not installed — skipping plots.")
            print("       pip install matplotlib")


if __name__ == "__main__":
    main()
