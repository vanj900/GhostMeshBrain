"""HUD renderer — terminal display for the organism's vital signs.

Renders a compact, colour-coded dashboard to stdout.  Designed to be called
from ``ghoststate.sh`` or directly from Python.

Colour coding
-------------
- Green  (≥ 70 % of healthy range)
- Yellow (35–70 %)
- Red    (< 35 %)

Bifurcation Status row (added in v2.1)
--------------------------------------
Shows: plasticity_index | pre_collapse_score | ticks_since_calcification | awakening_count

Colour coding for Bifurcation:
- Green  = plastic regime (plasticity_index ≥ 0.5)
- Yellow = transitioning (0.3 ≤ plasticity_index < 0.5)
- Red    = guardian attractor (plasticity_index < 0.3)
"""

from __future__ import annotations

import os
import sys


# ANSI colour codes (disabled if not a TTY)
_IS_TTY = sys.stdout.isatty() or os.environ.get("GHOST_FORCE_COLOR") == "1"

_RESET = "\033[0m" if _IS_TTY else ""
_GREEN = "\033[32m" if _IS_TTY else ""
_YELLOW = "\033[33m" if _IS_TTY else ""
_RED = "\033[31m" if _IS_TTY else ""
_CYAN = "\033[36m" if _IS_TTY else ""
_BOLD = "\033[1m" if _IS_TTY else ""
_DIM = "\033[2m" if _IS_TTY else ""


def _colour(value: float, low: float, high: float, invert: bool = False) -> str:
    """Return ANSI colour for value in [low, high] range.

    invert=True means HIGH is bad (e.g., heat, waste).
    """
    ratio = (value - low) / max(high - low, 1e-9)
    if invert:
        ratio = 1.0 - ratio
    if ratio >= 0.70:
        return _GREEN
    if ratio >= 0.35:
        return _YELLOW
    return _RED


def _bifurcation_colour(plasticity_index: float) -> str:
    """Return ANSI colour for the Bifurcation Status row.

    Green  = plastic regime (plasticity_index ≥ 0.5)
    Yellow = transitioning (0.3 ≤ plasticity_index < 0.5)
    Red    = guardian attractor (plasticity_index < 0.3)
    """
    if plasticity_index >= 0.5:
        return _GREEN
    if plasticity_index >= 0.3:
        return _YELLOW
    return _RED


def _bar(value: float, max_val: float = 100.0, width: int = 20) -> str:
    filled = int(round((value / max(max_val, 1e-9)) * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def render_hud(
    state_dict: dict,
    mask_status: dict | None = None,
    bifurcation_status: dict | None = None,
) -> str:
    """Render a full HUD string from a MetabolicState dict.

    Parameters
    ----------
    state_dict:
        Output of MetabolicState.to_dict().
    mask_status:
        Optional dict from MaskRotator.status().
    bifurcation_status:
        Optional dict with bifurcation probe data.  Expected keys:
        ``plasticity_index``, ``pre_collapse_score``,
        ``ticks_since_calcification`` (int or None),
        ``awakening_count`` (int).

    Returns
    -------
    str
        Multi-line HUD string ready to print.
    """
    e = state_dict.get("energy", 0.0)
    h = state_dict.get("heat", 0.0)
    w = state_dict.get("waste", 0.0)
    i = state_dict.get("integrity", 0.0)
    s = state_dict.get("stability", 0.0)
    tick = state_dict.get("entropy", 0)
    stage = state_dict.get("stage", "dormant")

    lines: list[str] = [
        f"{_BOLD}{_CYAN}╔══════════════════════════════════════════╗{_RESET}",
        f"{_BOLD}{_CYAN}║          GhostMesh  ·  Vital Signs       ║{_RESET}",
        f"{_BOLD}{_CYAN}╠══════════════════════════════════════════╣{_RESET}",
        _vital_line("Energy    (E)", e,   0.0, 100.0, invert=False),
        _vital_line("Heat      (T)", h, 0.0, 100.0, invert=True),
        _vital_line("Waste     (W)", w,  0.0, 100.0, invert=True),
        _vital_line("Integrity (M)", i,  0.0, 100.0, invert=False),
        _vital_line("Stability (S)", s, 0.0, 100.0, invert=False),
        f"{_BOLD}{_CYAN}╠══════════════════════════════════════════╣{_RESET}",
        f"{_CYAN}║{_RESET}  Stage : {_BOLD}{stage:<10}{_RESET}     Tick : {_DIM}{tick:>8}{_RESET}  {_CYAN}║{_RESET}",
    ]

    if mask_status:
        mask_name = mask_status.get("active_mask", "—")
        mask_ticks = mask_status.get("ticks_active", 0)
        lines.append(
            f"{_CYAN}║{_RESET}  Mask  : {_BOLD}{mask_name:<10}{_RESET}   Active: {_DIM}{mask_ticks:>4} ticks{_RESET}   {_CYAN}║{_RESET}"
        )

    # ── Bifurcation Status row ─────────────────────────────────────────────
    if bifurcation_status:
        plasticity = float(bifurcation_status.get("plasticity_index", 0.0))
        pre_collapse = float(bifurcation_status.get("pre_collapse_score", 0.0))
        ticks_calc = bifurcation_status.get("ticks_since_calcification")
        n_awakenings = int(bifurcation_status.get("awakening_count", 0))
        bif_col = _bifurcation_colour(plasticity)
        if plasticity >= 0.5:
            regime_label = "PLASTIC"
        elif plasticity >= 0.3:
            regime_label = "TRANS. "
        else:
            regime_label = "GUARDIAN"
        tsc_str = f"{ticks_calc:>5}t" if ticks_calc is not None else "  n/a "
        lines.append(f"{_BOLD}{_CYAN}╠══════════════════════════════════════════╣{_RESET}")
        lines.append(
            f"{_CYAN}║{_RESET}  {_BOLD}Bifurcation Status{_RESET}                          {_CYAN}║{_RESET}"
        )
        lines.append(
            f"{_CYAN}║{_RESET}  Plasticity : {bif_col}{plasticity:.3f}{_RESET}"
            f"  ({bif_col}{regime_label}{_RESET})"
            f"   PreCollapse: {_YELLOW}{pre_collapse:.3f}{_RESET}  {_CYAN}║{_RESET}"
        )
        lines.append(
            f"{_CYAN}║{_RESET}  Calcification: {_DIM}{tsc_str}{_RESET}"
            f"  Awakenings: {_RED if n_awakenings > 0 else _DIM}{n_awakenings:>3}{_RESET}       {_CYAN}║{_RESET}"
        )

    lines.append(f"{_BOLD}{_CYAN}╚══════════════════════════════════════════╝{_RESET}")
    return "\n".join(lines)


def _vital_line(label: str, value: float, low: float, high: float, invert: bool) -> str:
    col = _colour(value, low, high, invert=invert)
    bar = _bar(value, max_val=high)
    return (
        f"{_CYAN}║{_RESET}  {label:<14} "
        f"{col}{bar}{_RESET} "
        f"{col}{value:6.1f}{_RESET} {_CYAN}║{_RESET}"
    )


def print_hud(
    state_dict: dict,
    mask_status: dict | None = None,
    bifurcation_status: dict | None = None,
) -> None:
    """Print the HUD to stdout."""
    print(render_hud(state_dict, mask_status, bifurcation_status=bifurcation_status), flush=True)

