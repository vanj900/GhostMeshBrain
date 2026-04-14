#!/usr/bin/env python3
"""stress_test.py — GhostMesh long-run stress tester.

Runs one or more simulation sessions back-to-back with configurable
compute load and logs every tick's vital signs to a JSONL file for
later analysis or plotting.

Usage
-----
    # 2000-tick run at default load, logs to /tmp/run_cl1.0.jsonl
    python examples/stress_test.py

    # 5000-tick run at elevated load 1.8
    python examples/stress_test.py --ticks 5000 --compute-load 1.8

    # Three sessions with varied loads, save to a named output dir
    python examples/stress_test.py --multi --output-dir /tmp/ghost_runs

    # Disable environmental events (flat world)
    python examples/stress_test.py --no-env-events --ticks 1000

Output
------
Each session writes a JSONL file (one JSON record per tick) plus a
human-readable summary table to stdout.  If matplotlib is installed,
``--plot`` generates a PNG vitals chart.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Silence HUD and state-file writes during stress tests by default
os.environ.setdefault("GHOST_HUD", "0")


def _run_session(
    *,
    ticks: int,
    compute_load: float,
    log_path: str,
    env_events: bool,
    seed: int | None,
    state_file: str,
    diary_path: str,
) -> dict:
    """Run one simulation session and return a summary dict."""
    import os

    os.environ["GHOST_COMPUTE_LOAD"] = str(compute_load)
    os.environ["GHOST_VITALS_LOG"] = log_path
    os.environ["GHOST_ENV_EVENTS"] = "1" if env_events else "0"
    os.environ["GHOST_STATE_FILE"] = state_file
    os.environ["GHOST_DIARY_PATH"] = diary_path
    # No sleep between ticks in stress mode
    os.environ["GHOST_PULSE"] = "0"

    from thermodynamic_agency.pulse import GhostMesh

    mesh = GhostMesh(seed=seed)
    t0 = time.perf_counter()
    mesh.run(max_ticks=ticks)
    elapsed = time.perf_counter() - t0

    # Parse log to produce summary stats
    records = []
    if os.path.exists(log_path):
        with open(log_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    summary = _summarise(records, compute_load, elapsed, mesh.state)
    return summary


def _summarise(records: list[dict], compute_load: float, elapsed: float, final_state) -> dict:
    """Compute summary statistics from a list of per-tick records."""
    if not records:
        return {"error": "no records"}

    ticks = len(records)
    actions = [r["action"] for r in records]
    events = [r.get("env_event", "none") for r in records]
    masks = [r.get("mask", "?") for r in records]

    def mean(key: str) -> float:
        vals = [r[key] for r in records if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    def count(lst: list, val: str) -> int:
        return sum(1 for x in lst if x == val)

    action_dist = {
        a: count(actions, a) for a in ["FORAGE", "REST", "REPAIR", "DECIDE"]
    }
    event_dist: dict[str, int] = {}
    for e in events:
        event_dist[e] = event_dist.get(e, 0) + 1
    mask_dist: dict[str, int] = {}
    for m in masks:
        mask_dist[m] = mask_dist.get(m, 0) + 1

    # Near-death events: ticks where energy < 5 or heat > 95 or integrity < 20
    near_death = sum(
        1 for r in records
        if r.get("energy", 100) < 5
        or r.get("heat", 0) > 95
        or r.get("integrity", 100) < 20
        or r.get("stability", 100) < 5
    )

    final = records[-1]

    return {
        "ticks": ticks,
        "compute_load": compute_load,
        "elapsed_s": round(elapsed, 2),
        "ticks_per_sec": round(ticks / elapsed, 1) if elapsed > 0 else 0,
        "survived": True,  # if we got here, no death exception was fatal
        "final_stage": final.get("stage", "?"),
        "final_health": round(final.get("health", 0), 1),
        "final_energy": round(final.get("energy", 0), 1),
        "final_heat": round(final.get("heat", 0), 1),
        "final_integrity": round(final.get("integrity", 0), 1),
        "final_stability": round(final.get("stability", 0), 1),
        "avg_energy": round(mean("energy"), 1),
        "avg_heat": round(mean("heat"), 1),
        "avg_waste": round(mean("waste"), 1),
        "avg_integrity": round(mean("integrity"), 1),
        "avg_stability": round(mean("stability"), 1),
        "avg_affect": round(mean("affect"), 4),
        "avg_free_energy": round(mean("free_energy"), 2),
        "near_death_ticks": near_death,
        "action_distribution": action_dist,
        "event_distribution": event_dist,
        "mask_distribution": mask_dist,
    }


def _print_summary(summary: dict, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Session: {label}")
    print(f"{'='*60}")
    if "error" in summary:
        print(f"  ERROR: {summary['error']}")
        return
    print(f"  Ticks          : {summary['ticks']} ({summary['ticks_per_sec']} ticks/s)")
    print(f"  Compute load   : {summary['compute_load']}")
    print(f"  Survived       : {summary['survived']}")
    print(f"  Final stage    : {summary['final_stage']}")
    print(f"  Final health   : {summary['final_health']}")
    print(f"  Near-death tks : {summary['near_death_ticks']}")
    print(f"\n  Vital averages over run:")
    print(f"    Energy     {summary['avg_energy']:6.1f}  (final {summary['final_energy']:.1f})")
    print(f"    Heat       {summary['avg_heat']:6.1f}  (final {summary['final_heat']:.1f})")
    print(f"    Waste      {summary['avg_waste']:6.1f}")
    print(f"    Integrity  {summary['avg_integrity']:6.1f}  (final {summary['final_integrity']:.1f})")
    print(f"    Stability  {summary['avg_stability']:6.1f}  (final {summary['final_stability']:.1f})")
    print(f"    Affect     {summary['avg_affect']:+.4f}")
    print(f"    Free-E     {summary['avg_free_energy']:6.2f}")
    print(f"\n  Action distribution:")
    for a, n in sorted(summary["action_distribution"].items()):
        pct = 100.0 * n / summary["ticks"] if summary["ticks"] else 0
        print(f"    {a:<8} {n:>6}  ({pct:.1f}%)")
    print(f"\n  Mask distribution:")
    for m, n in sorted(summary["mask_distribution"].items(), key=lambda x: -x[1]):
        pct = 100.0 * n / summary["ticks"] if summary["ticks"] else 0
        print(f"    {m:<12} {n:>6}  ({pct:.1f}%)")
    top_events = sorted(summary["event_distribution"].items(), key=lambda x: -x[1])[:5]
    if top_events:
        print(f"\n  Top env events (by frequency):")
        for ev, n in top_events:
            pct = 100.0 * n / summary["ticks"] if summary["ticks"] else 0
            print(f"    {ev:<20} {n:>5}  ({pct:.1f}%)")


def _maybe_plot(log_path: str, output_path: str) -> None:
    """Generate a vitals chart if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "\n[plot] matplotlib not installed — skipping chart.\n"
            "       Install with: pip install matplotlib"
        )
        return

    # Delegate to plot_vitals.py if it exists alongside this script
    plot_script = Path(__file__).parent / "plot_vitals.py"
    if plot_script.exists():
        import subprocess
        subprocess.run(
            [sys.executable, str(plot_script), log_path, "--output", output_path],
            check=False,
        )
    else:
        print(f"[plot] plot_vitals.py not found alongside stress_test.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GhostMesh stress tester — long-run vital sign logger."
    )
    parser.add_argument(
        "--ticks", type=int, default=2000,
        help="Number of ticks per session (default: 2000)",
    )
    parser.add_argument(
        "--compute-load", type=float, default=1.0,
        help="GHOST_COMPUTE_LOAD for the session (default: 1.0)",
    )
    parser.add_argument(
        "--multi", action="store_true",
        help=(
            "Run multiple sessions with varied compute loads "
            "(0.5, 1.0, 1.5, 2.0) — ignores --compute-load"
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="/tmp/ghost_runs",
        help="Directory for JSONL logs and plots (default: /tmp/ghost_runs)",
    )
    parser.add_argument(
        "--no-env-events", action="store_true",
        help="Disable stochastic environmental events (flat-world run)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible runs",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate matplotlib vitals charts (requires matplotlib)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions: list[tuple[float, str]] = []  # (compute_load, label)
    if args.multi:
        for cl in [0.5, 1.0, 1.5, 2.0]:
            sessions.append((cl, f"cl{cl}"))
    else:
        sessions.append((args.compute_load, f"cl{args.compute_load}"))

    summaries = []
    for cl, label in sessions:
        log_path = str(out_dir / f"vitals_{label}.jsonl")
        state_file = str(out_dir / f"state_{label}.json")
        diary_path = str(out_dir / f"diary_{label}.db")

        # Clean up any previous run artifacts
        for p in [log_path, state_file, diary_path]:
            if os.path.exists(p):
                os.remove(p)

        print(f"\n[stress_test] Starting session '{label}' — {args.ticks} ticks, "
              f"compute_load={cl}, env_events={not args.no_env_events}")

        summary = _run_session(
            ticks=args.ticks,
            compute_load=cl,
            log_path=log_path,
            env_events=not args.no_env_events,
            seed=args.seed,
            state_file=state_file,
            diary_path=diary_path,
        )
        summary["label"] = label
        summary["log_path"] = log_path
        summaries.append(summary)
        _print_summary(summary, label)

        if args.plot:
            plot_out = str(out_dir / f"vitals_{label}.png")
            _maybe_plot(log_path, plot_out)

    # Write combined summary JSON
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"\n[stress_test] Summaries written to {summary_path}")
    print(f"[stress_test] JSONL logs in {out_dir}")


if __name__ == "__main__":
    main()
