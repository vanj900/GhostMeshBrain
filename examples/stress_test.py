#!/usr/bin/env python3
"""GhostMesh stress test runner.

Runs the organism for N ticks with configurable compute load and stochastic
environmental surprises, then prints a statistical summary.  Optionally
writes a JSONL run log for analysis with ``analyze_run.py``.

Usage
-----
    # Quick 500-tick run with default settings
    python examples/stress_test.py

    # 2000-tick harsh environment run, log to file
    python examples/stress_test.py --ticks 2000 --load 1.5 \\
        --stressor-prob 0.08 --stressor-intensity 1.2 \\
        --log-file /tmp/ghost_run.jsonl

    # Reproducible run (fixed seed), no HUD output
    python examples/stress_test.py --ticks 5000 --seed 42 --no-hud \\
        --log-file /tmp/ghost_long.jsonl

Environment Variables (override CLI flags)
------------------------------------------
GHOST_COMPUTE_LOAD, GHOST_STRESSOR_PROB, GHOST_STRESSOR_INTENSITY,
GHOST_STRESSOR_SEED, GHOST_LOG_FILE, GHOST_HUD — same semantics as
described in the main README.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

# Allow running directly without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thermodynamic_agency.core.exceptions import GhostDeathException  # noqa: F401
from thermodynamic_agency.pulse import GhostMesh


def _print_summary(summary: dict) -> None:
    print(f"\n{'='*60}")
    print("Run Summary")
    print(f"{'='*60}")

    print(f"  Total ticks  : {summary.get('total_ticks', '?')}")
    print(f"  Final tick   : {summary.get('final_tick', '?')}")
    print(f"  Final stage  : {summary.get('final_stage', '?')}")

    action_dist = summary.get("action_distribution", {})
    if action_dist:
        n = summary["total_ticks"]
        print("\n  Action Token Distribution:")
        for action in ("DECIDE", "FORAGE", "REST", "REPAIR"):
            count = action_dist.get(action, 0)
            pct = count / n * 100 if n else 0.0
            bar = "█" * int(pct / 2)
            print(f"    {action:<8} {bar:<25} {count:5d}  ({pct:5.1f}%)")

    mask_dist = summary.get("mask_distribution", {})
    if mask_dist:
        n = summary["total_ticks"]
        print("\n  Personality Mask Distribution:")
        for mask, count in sorted(mask_dist.items(), key=lambda x: -x[1]):
            pct = count / n * 100 if n else 0.0
            print(f"    {mask:<12} {count:5d}  ({pct:5.1f}%)")

    regime_dist = summary.get("precision_regime_distribution", {})
    if regime_dist:
        n = summary["total_ticks"]
        print("\n  Precision Regime Distribution:")
        for regime, count in sorted(regime_dist.items()):
            pct = count / n * 100 if n else 0.0
            print(f"    {regime:<14} {count:5d}  ({pct:5.1f}%)")

    for metric, label in (
        ("affect",      "Affect (pleasure signal)"),
        ("free_energy", "Free Energy (surprise)  "),
        ("health",      "Health Score            "),
    ):
        d = summary.get(metric, {})
        if d:
            print(
                f"\n  {label}: "
                f"mean={d['mean']:+.3f}  "
                f"min={d['min']:+.3f}  "
                f"max={d['max']:+.3f}"
            )

    print(f"\n  Ethics blocks     : {summary.get('total_ethics_blocks', 0)}")
    n_stressor = summary.get("total_stressor_events", 0)
    n_total = summary.get("total_ticks", 0)
    stressor_pct = n_stressor / n_total * 100 if n_total else 0.0
    print(f"  Stressor events   : {n_stressor}  ({stressor_pct:.1f}% of ticks)")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="GhostMesh stress test runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticks", type=int, default=500,
        help="Number of heartbeat ticks to run",
    )
    parser.add_argument(
        "--load", type=float, default=1.0,
        help="GHOST_COMPUTE_LOAD — per-tick computational burden",
    )
    parser.add_argument(
        "--stressor-prob", type=float, default=0.05,
        help="Probability of a random environmental disturbance per tick",
    )
    parser.add_argument(
        "--stressor-intensity", type=float, default=1.0,
        help="Scale factor for disturbance magnitudes",
    )
    parser.add_argument(
        "--log-file", type=str, default="",
        help="Path to write JSONL run log (empty = no file log)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible stressor events",
    )
    parser.add_argument(
        "--no-hud", action="store_true",
        help="Suppress per-tick HUD output (faster, cleaner logs)",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, "ghost_metabolic.json")
        diary_path = os.path.join(tmpdir, "ghost_diary.db")

        os.environ["GHOST_STATE_FILE"] = state_file
        os.environ["GHOST_DIARY_PATH"] = diary_path
        os.environ["GHOST_HUD"] = "0" if args.no_hud else "1"
        os.environ["GHOST_PULSE"] = "0"          # no sleep between heartbeat ticks (run at full speed)
        os.environ["GHOST_COMPUTE_LOAD"] = str(args.load)
        os.environ["GHOST_STRESSOR_PROB"] = str(args.stressor_prob)
        os.environ["GHOST_STRESSOR_INTENSITY"] = str(args.stressor_intensity)
        if args.seed is not None:
            os.environ["GHOST_STRESSOR_SEED"] = str(args.seed)
        if args.log_file:
            os.environ["GHOST_LOG_FILE"] = args.log_file

        print(f"\n{'='*60}")
        print("GhostMesh Stress Test")
        print(f"  ticks={args.ticks}  load={args.load}  "
              f"stressor_prob={args.stressor_prob}  "
              f"intensity={args.stressor_intensity}")
        if args.seed is not None:
            print(f"  seed={args.seed}")
        if args.log_file:
            print(f"  log_file={args.log_file}")
        print(f"{'='*60}\n")

        mesh = GhostMesh()
        try:
            mesh.run(max_ticks=args.ticks)
        except SystemExit:
            pass  # natural death is expected in harsh environments

        # Close logger and print summary
        mesh.run_logger.close()
        summary = mesh.run_logger.summary()
        if summary:
            _print_summary(summary)
        else:
            # Fallback: print final state directly
            s = mesh.state
            print(f"\nFinal tick: {s.entropy}  stage: {s.stage}  "
                  f"health: {s.health_score():.1f}")

        if args.log_file:
            print(f"Run log written to: {args.log_file}")
            print("Analyze with:  python examples/analyze_run.py "
                  f"{args.log_file}\n")


if __name__ == "__main__":
    main()
