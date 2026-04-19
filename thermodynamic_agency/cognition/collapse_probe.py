"""CollapseProbe — rolling-window phase-transition detector.

Tracks the pre-collapse behavioral signature of the Dreamer→Guardian
bifurcation described in Direction #1 of the breakthrough analysis.

The probe maintains a rolling buffer of the last ``window`` ticks and
computes, on every ``update()`` call:

- **action_entropy**   Shannon entropy over the action-token distribution
                       in the window (FORAGE/REST/REPAIR/DECIDE).  Falls
                       sharply as the organism locks into repetitive
                       FORAGE/REPAIR cycles before collapse.
- **mask_entropy**     Shannon entropy over the mask distribution.  Collapses
                       when Guardian / SalienceNet becomes the exclusive mode.
- **guardian_fraction** Fraction of ticks where the active mask was
                       Guardian or SalienceNet (conservation / threat modes).
- **dreamer_fraction** Fraction of ticks where the active mask was
                       Dreamer, DefaultMode, or CentralExec (exploratory /
                       plastic modes).
- **plasticity_index** dreamer_fraction / (guardian_fraction + ε) — the key
                       bifurcation signal.  Values > 1 → plastic regime;
                       values < 0.3 → guardian attractor regime.
- **mean_free_energy** Rolling mean variational free energy.
- **d_allostatic**     EMA-based derivative of allostatic load (how fast the
                       load is accumulating).  A sustained positive d_AL
                       predicts imminent precision saturation.
- **d_energy**         EMA derivative of energy (negative → resource crisis
                       approaching).
- **d_heat**           EMA derivative of heat.
- **pre_collapse_score** Composite (0–1) signal synthesising guardian_fraction,
                       low plasticity_index, rising d_allostatic, and falling
                       action_entropy.  Rises toward 1 in the pre-collapse
                       window.
- **is_near_transition** ``True`` when pre_collapse_score exceeds the
                       detection threshold (default 0.38), or when the lagged
                       detector fires (d_AL spike within last 300 ticks *and*
                       plasticity_index is still falling below its slow EMA).

Usage
-----
    probe = CollapseProbe(window=500)
    # inside _pulse() after mask rotation:
    snapshot = probe.update(
        action=action,
        mask=self.rotator.active.name,
        free_energy=self.state.free_energy_estimate(),
        allostatic_load=self.state.allostatic_load,
        energy=self.state.energy,
        heat=self.state.heat,
        precision_weights=precision_weights,
        efe_accuracy=efe_accuracy,
        efe_complexity=efe_complexity,
    )
    # snapshot.pre_collapse_score, snapshot.is_near_transition, etc.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


# ------------------------------------------------------------------ #
# Mask classification helpers                                          #
# ------------------------------------------------------------------ #

_GUARDIAN_MASKS: frozenset[str] = frozenset({"Guardian", "SalienceNet"})
_DREAMER_MASKS: frozenset[str] = frozenset({"Dreamer", "DefaultMode", "CentralExec"})

# Minimum d_allostatic value (EMA derivative, per tick) that counts as an
# allostatic-load spike for the lagged transition detector.
_AL_SPIKE_THRESHOLD: float = 0.5


# ------------------------------------------------------------------ #
# Snapshot dataclass                                                   #
# ------------------------------------------------------------------ #

@dataclass
class CollapseSnapshot:
    """Read-only statistics over the current rolling window."""

    window: int                    # configured window size
    ticks_in_window: int           # actual ticks accumulated so far
    action_entropy: float          # Shannon H over action distribution [0, ~2]
    mask_entropy: float            # Shannon H over mask distribution [0, ~3]
    guardian_fraction: float       # fraction of ticks in guardian-mode masks
    dreamer_fraction: float        # fraction of ticks in dreamer-mode masks
    plasticity_index: float        # dreamer / (guardian + ε)
    mean_free_energy: float        # rolling mean FE
    d_allostatic: float            # EMA derivative of allostatic load (per tick)
    d_energy: float                # EMA derivative of energy (per tick)
    d_heat: float                  # EMA derivative of heat (per tick)
    mean_precision_energy: float   # rolling mean precision weight for energy vital
    mean_precision_heat: float     # rolling mean precision weight for heat vital
    mean_precision_waste: float    # rolling mean precision weight for waste vital
    mean_precision_integrity: float
    mean_precision_stability: float
    mean_efe_accuracy: float       # rolling mean EFE accuracy component
    mean_efe_complexity: float     # rolling mean EFE complexity component
    pre_collapse_score: float      # composite 0-1 transition signal
    is_near_transition: bool       # True when pre_collapse_score ≥ threshold
                                   # OR lagged d_AL spike + falling plasticity


# ------------------------------------------------------------------ #
# Per-tick slot stored in the rolling buffer                           #
# ------------------------------------------------------------------ #

@dataclass
class _TickSlot:
    action: str
    mask: str
    free_energy: float
    allostatic_load: float
    energy: float
    heat: float
    precision_energy: float
    precision_heat: float
    precision_waste: float
    precision_integrity: float
    precision_stability: float
    efe_accuracy: float
    efe_complexity: float


# ------------------------------------------------------------------ #
# CollapseProbe                                                        #
# ------------------------------------------------------------------ #

class CollapseProbe:
    """Rolling-window phase-transition detector for the plasticity collapse.

    Parameters
    ----------
    window:
        Rolling buffer size in ticks (default 500).
    detection_threshold:
        pre_collapse_score above which ``is_near_transition`` is True
        (default 0.38).  The probe also fires via a lagged condition: if
        ``d_allostatic`` exceeded ``_AL_SPIKE_THRESHOLD`` (0.5) at any point
        in the last 300 ticks *and* the current ``plasticity_index`` is below
        its slow EMA (i.e. still falling), ``is_near_transition`` is set even
        when ``pre_collapse_score`` is below the threshold.
    ema_alpha:
        Exponential moving-average decay rate for derivative estimation
        (default 0.05 — ~20-tick smoothing).
    min_fill_fraction:
        Minimum fraction of the window that must be filled before
        ``is_near_transition`` can fire.  Prevents cold-start false
        positives when the rolling buffer has only a handful of ticks
        and statistics are not yet meaningful (default 0.2 = 100 ticks
        out of 500).
    """

    # Slow EMA alpha for plasticity-index trend tracking used by the lagged
    # transition detector (~50-tick smoothing; kept separate from _alpha so
    # the two smoothing rates can be tuned independently).
    _PLASTICITY_EMA_ALPHA: float = 0.02

    def __init__(
        self,
        window: int = 500,
        detection_threshold: float = 0.38,
        ema_alpha: float = 0.05,
        min_fill_fraction: float = 0.2,
    ) -> None:
        self._window = window
        self._threshold = detection_threshold
        self._alpha = ema_alpha
        self._min_fill = max(1, int(window * min_fill_fraction))
        self._buf: deque[_TickSlot] = deque(maxlen=window)
        # EMA state for derivatives
        self._ema_al: float | None = None
        self._ema_energy: float | None = None
        self._ema_heat: float | None = None
        self._d_allostatic: float = 0.0
        self._d_energy: float = 0.0
        self._d_heat: float = 0.0
        # Lagged-transition state
        self._d_al_history: deque[float] = deque(maxlen=300)
        self._d_al_spike_count: int = 0   # O(1) count of values > _AL_SPIKE_THRESHOLD
        self._plasticity_ema: float | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(
        self,
        *,
        action: str,
        mask: str,
        free_energy: float,
        allostatic_load: float,
        energy: float,
        heat: float,
        precision_weights: dict[str, float] | None = None,
        efe_accuracy: float = 0.0,
        efe_complexity: float = 0.0,
    ) -> CollapseSnapshot:
        """Record one tick and return updated rolling statistics.

        Parameters
        ----------
        action:
            Action token: "FORAGE" / "REST" / "REPAIR" / "DECIDE".
        mask:
            Active mask name (e.g., "Guardian", "Dreamer").
        free_energy:
            Scalar free-energy estimate from ``MetabolicState``.
        allostatic_load:
            Current allostatic load [0, 100].
        energy:
            Current energy vital.
        heat:
            Current heat vital.
        precision_weights:
            Dict of per-vital precision weights from the last DECIDE step.
            Pass ``None`` or ``{}`` for non-DECIDE ticks; previous values are
            carried forward in the summary statistics.
        efe_accuracy:
            Accuracy component of the last-computed multistep EFE.
        efe_complexity:
            Complexity component of the last-computed multistep EFE.

        Returns
        -------
        CollapseSnapshot
            Current rolling-window statistics.
        """
        pw = precision_weights or {}
        slot = _TickSlot(
            action=action,
            mask=mask,
            free_energy=free_energy,
            allostatic_load=allostatic_load,
            energy=energy,
            heat=heat,
            precision_energy=pw.get("energy", 0.0),
            precision_heat=pw.get("heat", 0.0),
            precision_waste=pw.get("waste", 0.0),
            precision_integrity=pw.get("integrity", 0.0),
            precision_stability=pw.get("stability", 0.0),
            efe_accuracy=efe_accuracy,
            efe_complexity=efe_complexity,
        )
        self._buf.append(slot)
        self._update_derivatives(allostatic_load, energy, heat)
        # Maintain the lagged-spike history and its O(1) spike counter.
        # Before appending, check whether the oldest element (about to be
        # evicted when the deque is full) was a spike so we can decrement.
        if len(self._d_al_history) == self._d_al_history.maxlen:
            if self._d_al_history[0] > _AL_SPIKE_THRESHOLD:
                self._d_al_spike_count -= 1
        self._d_al_history.append(self._d_allostatic)
        if self._d_allostatic > _AL_SPIKE_THRESHOLD:
            self._d_al_spike_count += 1
        snap = self._compute_snapshot()
        # Update slow plasticity EMA *after* snapshot so _compute_snapshot()
        # compares the *current* plasticity against the previous-tick trend.
        _a = self._PLASTICITY_EMA_ALPHA
        if self._plasticity_ema is None:
            self._plasticity_ema = snap.plasticity_index
        else:
            self._plasticity_ema = (
                (1.0 - _a) * self._plasticity_ema + _a * snap.plasticity_index
            )
        return snap

    @property
    def window(self) -> int:
        return self._window

    def reset(self) -> None:
        """Clear the buffer and reset derivative state."""
        self._buf.clear()
        self._ema_al = None
        self._ema_energy = None
        self._ema_heat = None
        self._d_allostatic = 0.0
        self._d_energy = 0.0
        self._d_heat = 0.0
        self._d_al_history.clear()
        self._d_al_spike_count = 0
        self._plasticity_ema = None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _update_derivatives(
        self,
        allostatic_load: float,
        energy: float,
        heat: float,
    ) -> None:
        """Update EMA-based derivative estimates for three key vitals."""
        a = self._alpha
        if self._ema_al is None:
            self._ema_al = allostatic_load
            self._ema_energy = energy
            self._ema_heat = heat
        else:
            new_al = (1.0 - a) * self._ema_al + a * allostatic_load
            new_e = (1.0 - a) * self._ema_energy + a * energy  # type: ignore[operator]
            new_h = (1.0 - a) * self._ema_heat + a * heat  # type: ignore[operator]
            # Derivative ≈ (new EMA − old EMA) per tick
            self._d_allostatic = new_al - self._ema_al
            self._d_energy = new_e - self._ema_energy  # type: ignore[operator]
            self._d_heat = new_h - self._ema_heat  # type: ignore[operator]
            self._ema_al = new_al
            self._ema_energy = new_e
            self._ema_heat = new_h

    def _compute_snapshot(self) -> CollapseSnapshot:
        buf = self._buf
        n = len(buf)
        if n == 0:
            return self._empty_snapshot()

        # ---- Action distribution -----------------------------------------
        action_counts: dict[str, int] = {}
        for s in buf:
            action_counts[s.action] = action_counts.get(s.action, 0) + 1
        action_entropy = _shannon_entropy(list(action_counts.values()), n)

        # ---- Mask distribution -------------------------------------------
        mask_counts: dict[str, int] = {}
        for s in buf:
            mask_counts[s.mask] = mask_counts.get(s.mask, 0) + 1
        mask_entropy = _shannon_entropy(list(mask_counts.values()), n)

        guardian_count = sum(1 for s in buf if s.mask in _GUARDIAN_MASKS)
        dreamer_count = sum(1 for s in buf if s.mask in _DREAMER_MASKS)
        guardian_fraction = guardian_count / n
        dreamer_fraction = dreamer_count / n
        plasticity_index = dreamer_fraction / (guardian_fraction + 1e-6)

        # ---- Free energy ------------------------------------------------
        mean_fe = sum(s.free_energy for s in buf) / n

        # ---- Precision weights (only slots with non-zero weights) --------
        def _mean_prec(attr: str) -> float:
            vals = [getattr(s, attr) for s in buf if getattr(s, attr) > 0.0]
            return sum(vals) / len(vals) if vals else 0.0

        # ---- EFE components ---------------------------------------------
        efe_acc_vals = [s.efe_accuracy for s in buf if s.efe_accuracy > 0.0]
        efe_cmp_vals = [s.efe_complexity for s in buf if s.efe_complexity > 0.0]
        mean_efe_accuracy = sum(efe_acc_vals) / len(efe_acc_vals) if efe_acc_vals else 0.0
        mean_efe_complexity = sum(efe_cmp_vals) / len(efe_cmp_vals) if efe_cmp_vals else 0.0

        # ---- Pre-collapse composite score --------------------------------
        # Four sub-signals, each normalised to [0, 1]:
        #
        # 1. guardian dominance — rises as Guardian/SalienceNet takes over
        s_guardian = guardian_fraction  # already [0, 1]
        #
        # 2. plasticity collapse — plasticity_index < 1 = conservative; norm to 0-1
        #    index of 0 → score 1.0; index of 2+ → score 0.0
        s_plasticity = max(0.0, 1.0 - plasticity_index / 2.0)
        #
        # 3. allostatic load rising — d_allostatic > 0 increasingly → score 1.0
        #    saturates at d_AL = 0.5/tick (aggressive loading)
        s_d_al = max(0.0, min(1.0, self._d_allostatic / 0.5))
        #
        # 4. action entropy collapse — max theoretical H for 4 actions = log2(4)=2
        #    score rises as entropy falls below 0.8 (very narrow action set)
        s_entropy = max(0.0, 1.0 - action_entropy / 0.8) if action_entropy < 0.8 else 0.0
        #
        # Weighted composite
        pre_collapse_score = (
            0.35 * s_guardian
            + 0.30 * s_plasticity
            + 0.20 * s_d_al
            + 0.15 * s_entropy
        )
        pre_collapse_score = max(0.0, min(1.0, pre_collapse_score))

        # Only flag a transition if the buffer has been filled to at least
        # ``min_fill_fraction`` of the configured window.  This prevents
        # cold-start false positives when the rolling window has only a
        # handful of ticks and statistical estimates are not yet meaningful.
        #
        # Primary condition: composite pre_collapse_score exceeds threshold.
        _primary = pre_collapse_score >= self._threshold and n >= self._min_fill
        #
        # Lagged condition: an allostatic-load spike (d_AL > _AL_SPIKE_THRESHOLD)
        # was recorded within the last 300 ticks *and* plasticity_index is
        # currently below its slow EMA (still falling).  This catches the
        # temporally desynchronized pattern where d_AL fires early during the
        # loading event but guardian dominance (and thus low pre_collapse_score)
        # only emerges hundreds of ticks later.
        _al_spiked = self._d_al_spike_count > 0
        _plasticity_falling = (
            self._plasticity_ema is not None
            and plasticity_index < self._plasticity_ema
        )
        _lagged = _al_spiked and _plasticity_falling and n >= self._min_fill
        #
        is_near_transition = _primary or _lagged

        return CollapseSnapshot(
            window=self._window,
            ticks_in_window=n,
            action_entropy=action_entropy,
            mask_entropy=mask_entropy,
            guardian_fraction=guardian_fraction,
            dreamer_fraction=dreamer_fraction,
            plasticity_index=plasticity_index,
            mean_free_energy=mean_fe,
            d_allostatic=self._d_allostatic,
            d_energy=self._d_energy,
            d_heat=self._d_heat,
            mean_precision_energy=_mean_prec("precision_energy"),
            mean_precision_heat=_mean_prec("precision_heat"),
            mean_precision_waste=_mean_prec("precision_waste"),
            mean_precision_integrity=_mean_prec("precision_integrity"),
            mean_precision_stability=_mean_prec("precision_stability"),
            mean_efe_accuracy=mean_efe_accuracy,
            mean_efe_complexity=mean_efe_complexity,
            pre_collapse_score=pre_collapse_score,
            is_near_transition=is_near_transition,
        )

    def _empty_snapshot(self) -> CollapseSnapshot:
        return CollapseSnapshot(
            window=self._window,
            ticks_in_window=0,
            action_entropy=0.0,
            mask_entropy=0.0,
            guardian_fraction=0.0,
            dreamer_fraction=0.0,
            plasticity_index=0.0,
            mean_free_energy=0.0,
            d_allostatic=0.0,
            d_energy=0.0,
            d_heat=0.0,
            mean_precision_energy=0.0,
            mean_precision_heat=0.0,
            mean_precision_waste=0.0,
            mean_precision_integrity=0.0,
            mean_precision_stability=0.0,
            mean_efe_accuracy=0.0,
            mean_efe_complexity=0.0,
            pre_collapse_score=0.0,
            is_near_transition=False,
        )


# ------------------------------------------------------------------ #
# Utility                                                              #
# ------------------------------------------------------------------ #

def _shannon_entropy(counts: list[int], total: int) -> float:
    """Compute Shannon entropy (bits) from a list of integer counts."""
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h
