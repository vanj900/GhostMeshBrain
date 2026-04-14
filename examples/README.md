# examples/

This directory contains stress-test scripts and analysis tools for
validating GhostMesh's long-run behaviour.

---

## stress_test.py

Runs one or more simulation sessions with configurable tick counts and
`compute_load` values.  Logs every tick's vital signs to a JSONL file and
prints a summary table at the end.

**Dependencies:** none beyond the base package.

```bash
# Install the package first
pip install -e ".[dev]"          # from repo root

# 2000-tick run at default load (results → /tmp/ghost_runs/)
python examples/stress_test.py

# 5000-tick elevated-stress run
python examples/stress_test.py --ticks 5000 --compute-load 1.8

# Four sequential sessions with different loads in one invocation (cl0.5, cl1.0, cl1.5, cl2.0)
python examples/stress_test.py --multi --ticks 2000

# Flat-world run (no stochastic events) — control condition
python examples/stress_test.py --ticks 2000 --no-env-events

# Reproducible run with fixed seed + matplotlib chart
python examples/stress_test.py --ticks 2000 --seed 42 --plot

# Custom output directory
python examples/stress_test.py --output-dir /tmp/my_runs
```

### What it measures

| Column | Description |
|--------|-------------|
| `ticks` | Total ticks completed |
| `final_stage` | Developmental stage at run end |
| `near_death_ticks` | Ticks within lethal threshold on any vital |
| `avg_*` | Mean vital signs over the run |
| `action_distribution` | How often each action token was chosen |
| `mask_distribution` | How often each personality mask was active |
| `event_distribution` | Environmental event frequency |

---

## plot_vitals.py

Generates a four-panel matplotlib chart from a JSONL vitals log:

1. **Energy / Health** — with FORAGE threshold and death line
2. **Heat / Waste** — with REST thresholds and ThermalDeath line
3. **Integrity / Stability** — with REPAIR thresholds and death lines
4. **Affect / Free Energy** — affect fill + FE overlay with sweet-spot bands

Background shading indicates the active action token each tick.

**Dependencies:** `matplotlib` (optional — install with `pip install matplotlib`).

```bash
# Generate chart from a log file
python examples/plot_vitals.py /tmp/ghost_runs/vitals_cl1.0.jsonl

# Save to a custom path
python examples/plot_vitals.py /tmp/ghost_runs/vitals_cl1.0.jsonl -o my_chart.png

# Display interactively (requires a display)
python examples/plot_vitals.py /tmp/ghost_runs/vitals_cl1.0.jsonl --show
```

---

## Interpreting results

### Healthy run (compute_load ≈ 1.0)
- Energy oscillates between ~30–90; FORAGE triggered occasionally
- Heat and waste stay below their thresholds most of the time
- DECIDE dominates the action distribution (> 60 %)
- Affect is slightly positive on average (surprise resolving)
- Free energy stays in the sweet-spot band (8–45) most of the time

### Stressed run (compute_load ≈ 1.5–1.8)
- More FORAGE and REPAIR tokens; DECIDE drops
- Near-death ticks increase; personality biases toward Guardian/Healer
- Affect swings more negative; free energy spikes toward overload
- Ethics blocks may appear in the diary if crisis events stack

### Collapsed run (compute_load ≥ 2.0 without recovery)
- Death exception logged before `max_ticks` — `survived=False`
- Final state shows one vital at lethal threshold
- Useful for testing the death-cascade is actually non-bypassable

### What "emergence" looks like
Look for **mask switching correlated with affect sign changes**: the organism
should shift toward Courier/Dreamer when affect turns positive and toward
Guardian/Healer when it turns negative — without manual tuning.  A healthy
long run shows rhythmic oscillation, not a flat line or monotonic collapse.
