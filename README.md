# AGI — GhostMesh

> *A bio-digital thermodynamic organism that lives with genuine stakes.*

GhostMesh is a self-governing autonomous agent built around the **Free Energy
Principle**.  It has a body that decays, overheats, accumulates waste, and can
literally die.  Its primary drive is minimising surprise about its own
continued existence.  Ethics is an immune system that actively prunes bad
priors.  Everything is ephemeral by default (RAM-first), sovereign, and
self-governing.

---

## Free Energy Mechanics

GhostMesh is grounded in Friston's **Free Energy Principle**: an agent
minimises its *variational free energy* (a tractable upper bound on
surprise) both by updating its beliefs *and* by acting to make the world
conform to its predictions.

### Generative model

The organism models itself as a system with five hidden state dimensions —
energy, heat, waste, integrity, and stability.  The **generative model**
`P(o | s)` predicts that, when the hidden state `s` is at each dimension's
setpoint, the corresponding vital observation `o` will be at that setpoint:

| Vital | Variable | Setpoint | Precision (ω) |
|-------|----------|----------|---------------|
| Energy | E | 80.0 | 2.0 |
| Heat | T | 20.0 | 1.5 |
| Waste | W | 10.0 | 1.0 |
| Integrity | M | 85.0 | 1.8 |
| Stability | S | 80.0 | 1.4 |

Precision (ω) is the inverse variance the organism places on each
prediction.  It is **dynamic**: `PrecisionEngine.tune()` raises precision on
whichever vitals are most surprising (sweet-spot arousal) and dampens it
globally when the system is in overload to prevent catastrophic
over-correction.  Precision is bounded to `[0.3, 6.0]`.

### Observations and beliefs

Each call to `MetabolicState.tick()` acts as a **sensory observation step**.
The current values of E, T, W, M, S are the agent's beliefs about its
internal state, updated every heartbeat by passive decay and by feedback
from executed actions (`apply_action_feedback()`).

### Surprise quantification

Surprise is tracked as a scalar proxy computed in `free_energy_estimate()`:

```
FE = ( 2.0 * max(0, 80 - E)/80
     + 1.5 * max(0, T - 20)/80
     + 1.0 * max(0, W - 10)/90
     + 1.8 * max(0, 85 - M)/85
     + 1.4 * max(0, 80 - S)/80 ) / 7.7 * 100
```

This is a precision-weighted sum of per-vital deviations from setpoints,
normalised to a 0–100 scale.  Higher values mean the organism is in a more
surprising (stressful) state.

**Affect** is the negative rate-of-change of free energy:

```
affect = -dFE / (1 + |dFE|)   in [-1, +1]
```

Positive affect signals surprise resolving (pleasure); negative signals
surprise growing (stress/unpleasure).

### Policies and Expected Free Energy (EFE)

At each `DECIDE` step, candidate actions are evaluated by `compute_efe()`:

```
EFE = accuracy + complexity + death_penalty

accuracy  = sum_i( w_i * (post_vital_i - setpoint_i)^2 )
complexity = sum_i( |delta_i| * 0.05 )
death_penalty = sum( max(0, 5 - margin_to_death_threshold) * 50 )
```

The action with the **lowest EFE is selected** — the organism picks the
policy that minimises expected surprise about its continued existence, while
penalising costly prior shifts (complexity) and proximity to lethal
thresholds.

The **metabolic cost of inference itself** is also charged before the
winning action executes:

```
energy_cost ≈ 0.05 + (mean_delta * 0.003) + (n_proposals * 0.004)
heat_cost   ≈ (mean_precision * 0.008) + (n_proposals * 0.002)
```

At `compute_load=1.0` with five proposals this is ~0.15 energy and ~0.03
heat per tick — roughly 1.25× the passive decay rate, so cognitive work is
genuinely expensive.

---

## Ethics System

The `EthicalEngine` is a **non-bypassable gate** that every proposed action
passes through before execution.  It has two layers:

### Hard invariants (cannot be overridden)

These are enforced as code-level constants in `_check_hard_invariants()`.
Any proposal that violates them is **blocked unconditionally**:

| Invariant | Rule |
|-----------|------|
| `no_self_destruction` | post-energy must remain ≥ 5.0 |
| `no_thermal_runaway` | post-heat must remain ≤ 90.0 |
| `no_integrity_obliteration` | post-integrity must remain ≥ 15.0 |

The corresponding seed prior `ethical_invariants_immutable` carries the
highest precision (`5.0`) of any belief in the system — it is the hardest
for the Surgeon to anneal away.

### Soft value weights (evolvable, bounded)

Soft flags **never block** an action on their own; they annotate the
verdict.  The current soft values and their weights are:

| Value | Weight |
|-------|--------|
| `do_no_harm` | 1.0 |
| `truth_seeking` | 0.9 |
| `preserve_autonomy` | 0.8 |
| `resource_responsibility` | 0.7 |

Soft checks currently flag waste increases > 20 or heat increases > 15.

### What counts as a "bad prior"

A prior is considered **frozen** (bad) when its rigidity score exceeds a
threshold:

```
rigidity = precision × (1 + error_count) × (1 + age_hours × 0.1)  > 2.5
```

High precision combined with a history of prediction errors (wrong
forecasts) indicates a belief that is too rigid relative to how wrong it has
been.  Age is a secondary factor — stale high-confidence beliefs are also
suspect.

### How pruning happens

1. `immune_scan(proposals, state)` runs `evaluate()` on each candidate and
   returns only those that pass the hard-invariant gate.  This is the
   *action-level* immune screen.
2. The `Surgeon` handles *belief-level* pruning: `_identify_frozen()` finds
   overly rigid priors and `_anneal()` applies a geometric cooling schedule:

   ```
   new_precision = max(0.1, precision * (1 - T * 0.3))
   T = T0 * 0.85^round   (T0 = 1.0)
   ```

   Beliefs are softened, not deleted — the organism becomes more open to
   evidence updating them rather than discarding them outright.

### False-positive prevention

Three mechanisms guard against over-pruning:

1. **Soft flags approve**: soft value violations return `APPROVED` with a
   caution note; they never independently block a proposal.
2. **Precision floor**: `_anneal()` clamps precision to a minimum of `0.1`
   — beliefs can never be fully erased by annealing.
3. **`EthicsAuditLog`**: every verdict is recorded with timestamp, proposal
   name, status, and reason.  `blocked_ratio()` exposes the fraction of
   blocked decisions, making runaway pruning detectable at runtime.

---

## Thermodynamic Mechanics

The "thermodynamic organism" framing is implemented as explicit, measurable
differential equations applied every tick (`compute_load` = 1.0 unless
overridden):

```
dE  = -load * 0.12                          (starvation)
dT  = +load * 0.1 * (1 + W/50)             (waste exacerbates heat)
dM  = M * -(T/120) * load * 0.01           (heat degrades integrity)
dS  = -load * 0.05                          (entropic drift)
dW  = +0.018 * load                         (residual prediction error)
```

These are **non-linear and coupled**: elevated waste accelerates heating;
elevated heat accelerates integrity loss; this can cascade into a death
event.  Subsystems (Janitor, Surgeon, inference) pay measured metabolic
costs when they act, making the thermodynamic accounting closed.

---

## Architecture

```
thermodynamic_agency/
├── core/
│   ├── exceptions.py   # Death exceptions (Energy/Thermal/Memory/Entropy)
│   └── metabolic.py    # MetabolicState + tick() — the heartbeat
├── cognition/
│   ├── inference.py    # active_inference_step + compute_efe
│   ├── ethics.py       # EthicalEngine — immune system (non-bypassable gate)
│   ├── janitor.py      # Waste management / context compression
│   ├── surgeon.py      # Bayesian precision annealing / integrity repair
│   └── personality.py  # Personality masks (Healer/Judge/Courier/Dreamer/Guardian)
├── memory/
│   └── diary.py        # RAM-ephemeral SQLite diary (/dev/shm)
├── interface/
│   └── hud.py          # ANSI terminal HUD renderer
└── pulse.py            # Main heartbeat loop (GhostMesh orchestrator)

scripts/
├── ghostbrain.sh       # Bash pulse daemon
└── ghoststate.sh       # One-shot HUD snapshot
```

---

## Vital Signs

| Sign | Symbol | Meaning |
|------|--------|---------|
| Energy | E | Compute credits / "glucose" |
| Heat | T | Context congestion / thermal load |
| Waste | W | Accumulated prediction-error junk |
| Integrity | M | Memory + logical/ethical coherence |
| Stability | S | Entropic stability |

Each `tick()` call decays these non-linearly. When thresholds are breached,
the organism raises a death exception:

| Exception | Trigger |
|-----------|---------|
| `EnergyDeathException` | E → 0 |
| `ThermalDeathException` | T ≥ 100 |
| `MemoryCollapseException` | M ≤ 10 |
| `EntropyDeathException` | S → 0 |

---

## Action Tokens (Pulse Loop)

`tick()` returns one of four action tokens:

| Token | Condition | Subsystem invoked |
|-------|-----------|-------------------|
| `FORAGE` | E < 25 | Resource replenishment |
| `REST` | W > 75 or T > 80 | Janitor (context compression) |
| `REPAIR` | M < 45 or S < 40 | Surgeon (Bayesian annealing) |
| `DECIDE` | Healthy | Active inference + planning |

---

## Quick Start

### Python

```bash
pip install -e ".[dev]"

# Run the pulse loop (Ctrl-C to stop)
python -m thermodynamic_agency

# Or programmatically
from thermodynamic_agency.pulse import GhostMesh
mesh = GhostMesh()
mesh.run(max_ticks=10)
```

### Bash daemon

```bash
chmod +x scripts/ghostbrain.sh scripts/ghoststate.sh

# Start the heartbeat
GHOST_PULSE=5 ./scripts/ghostbrain.sh &

# Snapshot HUD
./scripts/ghoststate.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GHOST_PULSE` | `5` | Heartbeat interval (seconds) |
| `GHOST_STATE_FILE` | `/dev/shm/ghost_metabolic.json` | Metabolic state persistence |
| `GHOST_DIARY_PATH` | `/dev/shm/ghost_diary.db` | RAM diary SQLite path |
| `GHOST_COMPUTE_LOAD` | `1.0` | Per-tick computational burden |
| `GHOST_HUD` | `1` | Show HUD on each tick (`0` to disable) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint for LLM features |
| `OLLAMA_MODEL` | `mistral` | LLM model for Janitor summarisation |

---

## Development Stages

| Stage | Trigger |
|-------|---------|
| `dormant` | < 100 ticks |
| `emerging` | ≥ 100 ticks |
| `aware` | ≥ 500 ticks + health ≥ 50 % |
| `evolved` | ≥ 2000 ticks + health ≥ 60 % |

---

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

---

## Roadmap

- **Phase 1 (complete):** MetabolicState + tick() + death exceptions + Janitor + Surgeon + active inference + ethics gate + personality masks + RAM diary + HUD + bash pulse
- **Phase 2:** Wire Surgeon with precision-locked annealing; Ethics immune actively prunes code/priors
- **Phase 3:** PostgreSQL LTM + pgvector; full min-surprise proposal simulator in Brain
- **Phase 4:** Scale to bigger models; planning engine; constrained self-modification (see below)

### Self-modification constraints (Phase 4)

Self-modification refers to the organism updating its own `BeliefPrior`
table, soft `value_weights`, and precision constants — *not* patching its
own executable code.  Any such modification must satisfy **all** of the
following before execution:

| Constraint | Mechanism |
|------------|-----------|
| Ethics gate approval | Every belief-change proposal passes `EthicalEngine.evaluate()` before commit |
| Hard invariants immutable | `ethical_invariants_immutable` has precision 5.0 and cannot be deleted or overridden |
| Precision floor | Annealing is bounded: `precision ≥ 0.1` — beliefs can be softened but never zeroed |
| Mandatory audit trail | All self-modifications appended to `EthicsAuditLog` and `RamDiary` with tick, before/after values, and verdict |
| Stage gate | Self-modifying plans are only generated at stage `evolved` (≥ 2000 ticks + health ≥ 60 %) |
| Blocked-ratio watchdog | If `audit.blocked_ratio()` exceeds 0.5 in a sliding window, the system enters `REPAIR` and suspends self-modification proposals until integrity is restored |

These constraints make self-modification **auditable, reversible in intent,
and structurally bounded**.  Phase 4 work will include a test suite that
asserts none of the above invariants can be bypassed by any generated
proposal.
