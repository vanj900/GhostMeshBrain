"""HomeostasisAdapter — hebbian setpoint drift with Genesis Doctrine bounds.

The five vital setpoints (energy=80, heat=20, waste=10, integrity=85,
stability=80) are currently compile-time constants shared by the inference
engine and the PredictiveHierarchy.  A healthy organism in a benign
environment should hover near those values, but a chronically stressed
organism may need to *adapt* its expectations — just as a human living at
altitude adapts their oxygen "setpoint" over weeks.

What this module does
---------------------
Maintains a per-vital exponential moving average (EMA) of observed vital
values.  On each observation (one call to ``observe()`` per metabolic tick),
the EMA is updated with a very slow learning rate (α = 0.001 by default,
roughly "1/1000 ticks of influence per observation").  The EMA represents
the organism's long-run expectation of each vital.

``adapted_setpoints()`` returns setpoints that are gently pulled toward that
long-run expectation, bounded within ±``max_drift_fraction`` (default 15%)
of the original compile-time setpoints.  This prevents wire-heading: the
organism can never drift so far from its designed baseline that a vital
collapse becomes "expected" and therefore unpunished.

Genesis Doctrine enforcement
-----------------------------
The bound ±15% is a hard invariant, not a soft preference.  It cannot be
modified by the SelfModEngine, the Surgeon, or the EFE optimiser.  It is
the analogue of the "Paperclip Maximiser" lock: the organism is allowed to
adapt, but not to rewrite its own survival targets out of existence.

Integration
-----------
    # In pulse.py __init__:
    self.homeostasis_adapter = HomeostasisAdapter()
    self.hierarchy = PredictiveHierarchy(homeostasis=self.homeostasis_adapter)

    # After each metabolic tick (in _pulse):
    self.homeostasis_adapter.observe(self.state)

    # In _decide(), pass adapted setpoints to active_inference_step:
    adapted_sp = self.homeostasis_adapter.adapted_setpoints()
    result = active_inference_step(..., setpoints=adapted_sp)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from thermodynamic_agency.core.metabolic import MetabolicState


# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

# Original compile-time setpoints (mirrored from inference.py / predictive_hierarchy.py).
# These are the "designed baseline" the organism was born with.
_INITIAL_SETPOINTS: dict[str, float] = {
    "energy":    80.0,
    "heat":      20.0,
    "waste":     10.0,
    "integrity": 85.0,
    "stability": 80.0,
}

# EMA learning rate: how fast the long-run expectation adapts.
# At α=0.001, 1000 ticks of sustained deviation shifts the EMA by ~63%
# of the total deviation.  This is intentionally slow — adaptation should
# take days (thousands of ticks) not minutes.
_EMA_ALPHA: float = 0.001

# Maximum allowed setpoint drift as a fraction of the initial setpoint.
# Hard Genesis Doctrine bound — cannot be overridden by any subsystem.
# ±15% means energy can drift 68–92, heat 17–23, waste 8.5–11.5, etc.
_MAX_DRIFT_FRACTION: float = 0.15

_VITALS: tuple[str, ...] = ("energy", "heat", "waste", "integrity", "stability")


# ------------------------------------------------------------------ #
# Status snapshot                                                      #
# ------------------------------------------------------------------ #

@dataclass
class HomeostasisStatus:
    """A read-only snapshot of the adapter's current state."""

    ema: dict[str, float]               # current long-run expectations
    adapted_setpoints: dict[str, float] # bounded adapted setpoints
    drift: dict[str, float]             # current drift from initial (signed)
    ticks_observed: int                 # total number of observe() calls


# ------------------------------------------------------------------ #
# HomeostasisAdapter                                                   #
# ------------------------------------------------------------------ #

class HomeostasisAdapter:
    """Slow hebbian setpoint adaptation with hard Genesis Doctrine bounds.

    Parameters
    ----------
    alpha:
        EMA learning rate (default 0.001 — very slow, ~1 000 ticks to
        reach 63% of sustained deviation).
    max_drift_fraction:
        Maximum setpoint drift as a fraction of each vital's initial value
        (default 0.15 = ±15%).  This is the Genesis Doctrine bound.
    """

    def __init__(
        self,
        alpha: float = _EMA_ALPHA,
        max_drift_fraction: float = _MAX_DRIFT_FRACTION,
    ) -> None:
        self._alpha = alpha
        self._max_drift_fraction = max_drift_fraction
        # EMA initialised at the designed setpoints (no initial bias)
        self._ema: dict[str, float] = dict(_INITIAL_SETPOINTS)
        self._ticks_observed: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def observe(self, state: MetabolicState) -> None:
        """Update the EMA from the current metabolic state.

        Should be called once per tick, after ``state.tick()`` returns but
        before inference scoring so that the adaptation is available this tick.
        """
        for v in _VITALS:
            obs = getattr(state, v)
            self._ema[v] = (1.0 - self._alpha) * self._ema[v] + self._alpha * obs
        self._ticks_observed += 1

    def adapted_setpoints(self) -> dict[str, float]:
        """Return the Genesis-bounded adapted setpoints.

        Each vital's setpoint is pulled toward the long-run EMA but clamped
        within ±``max_drift_fraction`` of the original compiled setpoint.

        Returns
        -------
        dict[str, float]
            Adapted setpoints for all five vitals.
        """
        result: dict[str, float] = {}
        for v in _VITALS:
            initial = _INITIAL_SETPOINTS[v]
            max_drift = abs(initial) * self._max_drift_fraction
            # Pull toward EMA, then clamp to Genesis Doctrine bound.
            raw = self._ema[v]
            adapted = min(max(raw, initial - max_drift), initial + max_drift)
            result[v] = adapted
        return result

    def status(self) -> HomeostasisStatus:
        """Return a read-only status snapshot for logging / debugging."""
        adapted = self.adapted_setpoints()
        drift = {
            v: adapted[v] - _INITIAL_SETPOINTS[v]
            for v in _VITALS
        }
        return HomeostasisStatus(
            ema=dict(self._ema),
            adapted_setpoints=adapted,
            drift=drift,
            ticks_observed=self._ticks_observed,
        )

    def reset(self) -> None:
        """Reset the EMA to the original setpoints (useful for tests)."""
        self._ema = dict(_INITIAL_SETPOINTS)
        self._ticks_observed = 0
