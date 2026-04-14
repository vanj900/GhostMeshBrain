"""Stochastic environmental events — the world the organism must survive.

Every tick the pulse loop calls ``sample_event()`` to draw an environmental
shock (or calm) from the environment model.  This forces the organism to forage
proactively and adapt rather than simply oscillating at equilibrium.

Event distribution (approximate probabilities at default settings):

    ─────────────────────────────────────────────────────
    Category        p       Effect
    ─────────────────────────────────────────────────────
    calm            0.45    small energy bonus — reward for surviving
    energy_spike    0.12    larger energy injection (windfall)
    energy_drain    0.10    sudden energy loss (extra compute demand)
    thermal_spike   0.10    sudden heat surge (context flood)
    waste_flood     0.08    sudden waste surge (prediction-error burst)
    integrity_hit   0.07    minor integrity damage (external contradiction)
    stability_quake 0.05    sudden stability drop (entropic shock)
    crisis          0.03    multi-vital shock (rare; tests resilience)
    ─────────────────────────────────────────────────────

All magnitudes are randomised within configurable bounds so the organism
cannot learn a fixed schedule.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class EnvironmentalEvent:
    """A single tick's environmental signal."""

    name: str
    delta_energy: float = 0.0
    delta_heat: float = 0.0
    delta_waste: float = 0.0
    delta_integrity: float = 0.0
    delta_stability: float = 0.0

    def is_null(self) -> bool:
        return (
            self.delta_energy == 0.0
            and self.delta_heat == 0.0
            and self.delta_waste == 0.0
            and self.delta_integrity == 0.0
            and self.delta_stability == 0.0
        )


# ── Default event catalogue ───────────────────────────────────────────────────
# Each entry: (name, weight, delta_factory)
# delta_factory is a zero-arg callable that returns a dict of vital deltas.

def _uniform(lo: float, hi: float, rng: random.Random | None = None) -> float:
    r = rng if rng is not None else random
    return r.uniform(lo, hi)


_EVENT_CATALOGUE: list[tuple[str, float, object]] = [
    # Calm periods — small reward for continued existence
    (
        "calm",
        0.45,
        lambda rng: {"delta_energy": _uniform(0.5, 2.5, rng)},
    ),
    # Windfall energy (task completed, token budget released, etc.)
    (
        "energy_spike",
        0.12,
        lambda rng: {"delta_energy": _uniform(5.0, 18.0, rng)},
    ),
    # Sudden energy drain (heavy external request, context growth)
    (
        "energy_drain",
        0.10,
        lambda rng: {"delta_energy": -_uniform(3.0, 10.0, rng)},
    ),
    # Thermal spike (context flood, long chain-of-thought)
    (
        "thermal_spike",
        0.10,
        lambda rng: {"delta_heat": _uniform(4.0, 15.0, rng)},
    ),
    # Waste flood (large prediction-error burst)
    (
        "waste_flood",
        0.08,
        lambda rng: {"delta_waste": _uniform(5.0, 20.0, rng)},
    ),
    # Integrity hit (external contradiction, conflicting evidence)
    (
        "integrity_hit",
        0.07,
        lambda rng: {"delta_integrity": -_uniform(2.0, 8.0, rng)},
    ),
    # Stability quake (entropic shock)
    (
        "stability_quake",
        0.05,
        lambda rng: {"delta_stability": -_uniform(3.0, 12.0, rng)},
    ),
    # Rare multi-vital crisis
    (
        "crisis",
        0.03,
        lambda rng: {
            "delta_energy": -_uniform(5.0, 12.0, rng),
            "delta_heat": _uniform(6.0, 18.0, rng),
            "delta_waste": _uniform(8.0, 20.0, rng),
            "delta_integrity": -_uniform(3.0, 7.0, rng),
        },
    ),
]

# Pre-compute cumulative weights for O(1) sampling
_NAMES: list[str] = [e[0] for e in _EVENT_CATALOGUE]
_WEIGHTS: list[float] = [e[1] for e in _EVENT_CATALOGUE]
_FACTORIES: list[object] = [e[2] for e in _EVENT_CATALOGUE]


def sample_event(rng: random.Random | None = None) -> EnvironmentalEvent:
    """Draw one environmental event for this tick.

    Parameters
    ----------
    rng:
        Optional seeded ``random.Random`` instance for reproducible runs.
        Defaults to the module-level ``random`` functions.

    Returns
    -------
    EnvironmentalEvent
        The sampled event, ready to be fed to ``MetabolicState.apply_action_feedback()``.
    """
    r = rng if rng is not None else random
    (chosen_name, _weight, factory) = r.choices(
        list(zip(_NAMES, _WEIGHTS, _FACTORIES)),
        weights=_WEIGHTS,
        k=1,
    )[0]
    deltas: dict[str, float] = factory(r)  # type: ignore[operator]
    return EnvironmentalEvent(
        name=chosen_name,
        delta_energy=deltas.get("delta_energy", 0.0),
        delta_heat=deltas.get("delta_heat", 0.0),
        delta_waste=deltas.get("delta_waste", 0.0),
        delta_integrity=deltas.get("delta_integrity", 0.0),
        delta_stability=deltas.get("delta_stability", 0.0),
    )
