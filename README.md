# AGI — GhostMesh

> *A bio-digital thermodynamic organism that lives with genuine stakes.*

GhostMesh is a self-governing autonomous agent built around the **Free Energy
Principle**.  It has a body that decays, overheats, accumulates waste, and can
literally die.  Its primary drive is minimising surprise about its own
continued existence.  Ethics is an immune system that actively prunes bad
priors.  Everything is ephemeral by default (RAM-first), sovereign, and
self-governing.

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
- **Phase 2:** Wire Surgeon with precisionlocked annealing; Ethics immune actively prunes code/priors
- **Phase 3:** PostgreSQL LTM + pgvector; full min-surprise proposal simulator in Brain
- **Phase 4:** Scale to bigger models; planning engine; hardened self-modification
