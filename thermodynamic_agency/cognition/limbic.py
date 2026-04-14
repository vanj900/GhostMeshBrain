"""Limbic system — amygdala, nucleus accumbens, and episodic buffer.

Layer 2 of the GhostMeshBrain architecture.  The limbic system mediates
affect-driven responses, reward-based policy modulation, and short-term
episodic memory consolidation.

Components
----------
AmygdalaModule
    Threat-detection unit.  When free energy exceeds the danger threshold,
    it generates a precision-override signal that amplifies attention on
    survival-critical vitals and deepens negative affect.  This is the
    organism's equivalent of a startle reflex — metabolically expensive
    (it adds heat) and impossible to suppress once triggered.

NucleusAccumbens
    Reward-modulation unit.  When affect is positive (surprise resolving),
    it applies an EFE discount — making exploratory policies cheaper in the
    active-inference sense.  This biases the organism toward Dreamer/Courier
    masks when things are going well.

EpisodicBuffer
    Short-term memory consolidation buffer.  Holds up to ``capacity`` recent
    episodic slots as live working memories.  Slots that overflow capacity
    without being consolidated accumulate integrity cost (the organism "pays"
    for unprocessed experience).  Consolidation is triggered during REST
    (Janitor pass) via ``flush_oldest()``.

LimbicLayer
    Orchestrates the three components each heartbeat and returns a
    ``LimbicSignal`` with combined precision overrides, EFE discount, and
    metabolic costs for the pulse loop to apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from thermodynamic_agency.core.metabolic import MetabolicState


# ------------------------------------------------------------------ #
# Amygdala                                                            #
# ------------------------------------------------------------------ #

# Free-energy threshold above which the amygdala fires a threat response.
# Below this level the organism is in its normal operating range.
AMYGDALA_THREAT_FE: float = 35.0

# Maximum precision boost added to survival-critical vitals under peak threat.
AMYGDALA_MAX_BOOST: float = 2.5

# Heat cost per unit of threat level — neural activation is expensive.
AMYGDALA_HEAT_COST_SCALE: float = 0.15


@dataclass
class AmygdalaSignal:
    """Output of a single amygdala evaluation tick."""

    threat_level: float                              # 0-1 proportional to FE overshoot
    precision_overrides: dict[str, float] = field(default_factory=dict)
    affect_modifier: float = 0.0                    # additional negative affect push
    heat_cost: float = 0.0                          # metabolic cost of firing


class AmygdalaModule:
    """Threat-detection: spikes precision on survival vitals under high FE."""

    def evaluate(self, state: MetabolicState) -> AmygdalaSignal:
        """Evaluate current free energy and return a threat-graded signal.

        Parameters
        ----------
        state:
            Current metabolic state (read-only).

        Returns
        -------
        AmygdalaSignal
            Zero-valued if FE is within normal range.  Otherwise carries
            precision boosts on energy, heat, and integrity, plus a
            negative affect modifier and heat cost.
        """
        fe = state.free_energy_estimate()
        if fe <= AMYGDALA_THREAT_FE:
            return AmygdalaSignal(threat_level=0.0)

        # Threat level: 0 at threshold → 1 at FE = 100
        threat = min(1.0, (fe - AMYGDALA_THREAT_FE) / (100.0 - AMYGDALA_THREAT_FE))

        # Precision boosts: energy and heat (immediate survival), integrity (coherence)
        boost = threat * AMYGDALA_MAX_BOOST
        overrides = {
            "energy": boost * 1.2,
            "heat": boost * 1.0,
            "integrity": boost * 0.8,
        }

        # Negative affect pressure: amygdala deepens stress signal
        affect_modifier = -threat * 0.3

        # Heat cost of neural activation
        heat_cost = threat * AMYGDALA_HEAT_COST_SCALE

        return AmygdalaSignal(
            threat_level=threat,
            precision_overrides=overrides,
            affect_modifier=affect_modifier,
            heat_cost=heat_cost,
        )


# ------------------------------------------------------------------ #
# Nucleus Accumbens                                                   #
# ------------------------------------------------------------------ #

# Affect threshold above which the accumbens grants a reward discount.
ACCUMBENS_AFFECT_THRESHOLD: float = 0.25

# Maximum fractional discount on EFE scores (lower EFE = more attractive policy).
ACCUMBENS_MAX_DISCOUNT: float = 0.20


class NucleusAccumbens:
    """Reward-modulation: positive affect reduces EFE scores."""

    def efe_discount(self, state: MetabolicState) -> float:
        """Return a fractional discount (0 – MAX_DISCOUNT) to apply to EFE scores.

        When affect is positive (surprise resolving), exploratory policies
        become relatively cheaper, biasing the organism toward Dreamer/Courier
        behavior.  At peak positive affect (1.0) the discount is at maximum.

        Parameters
        ----------
        state:
            Current metabolic state (affect field is read).

        Returns
        -------
        float
            Fractional EFE reduction in [0, ACCUMBENS_MAX_DISCOUNT].
        """
        affect = state.affect
        if affect <= ACCUMBENS_AFFECT_THRESHOLD:
            return 0.0
        scale = (affect - ACCUMBENS_AFFECT_THRESHOLD) / (1.0 - ACCUMBENS_AFFECT_THRESHOLD)
        return min(ACCUMBENS_MAX_DISCOUNT, scale * ACCUMBENS_MAX_DISCOUNT)


# ------------------------------------------------------------------ #
# Episodic Buffer                                                     #
# ------------------------------------------------------------------ #

# Default buffer capacity (number of live episodic slots before overflow cost kicks in)
DEFAULT_EPISODIC_CAPACITY: int = 20

# Integrity cost per unconsolidated slot beyond capacity
CONSOLIDATION_COST_PER_SLOT: float = 0.02


@dataclass
class EpisodicSlot:
    """One short-term episodic memory slot."""

    tick: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    consolidated: bool = False


class EpisodicBuffer:
    """Short-term episodic memory with consolidation overflow cost.

    While the buffer is within capacity, there is no cost.  When the number
    of unconsolidated slots exceeds ``capacity``, each excess slot charges
    ``CONSOLIDATION_COST_PER_SLOT`` integrity per tick — the organism pays
    for unprocessed experience.

    Consolidation is triggered externally (e.g. during REST) via
    ``flush_oldest()``.  Flushed slots are marked consolidated and removed
    from the live buffer once total size exceeds 2× capacity.
    """

    def __init__(self, capacity: int = DEFAULT_EPISODIC_CAPACITY) -> None:
        self.capacity = capacity
        self._slots: list[EpisodicSlot] = []

    def push(self, slot: EpisodicSlot) -> None:
        """Add a new episodic slot to the buffer.

        If unconsolidated count exceeds ``capacity * 2``, the oldest
        unconsolidated slots are silently discarded (lost memories from
        failing to consolidate during REST).  This bounds the overflow cost
        and keeps the buffer from growing without limit.
        """
        self._slots.append(slot)

        # Auto-trim: discard oldest unconsolidated when buffer is too full.
        # Single-pass: count how many to drop, then rebuild in one go.
        max_unconsolidated = self.capacity * 2
        n_active = sum(1 for s in self._slots if not s.consolidated)
        trim = max(0, n_active - max_unconsolidated)
        if trim:
            to_remove = 0
            kept: list[EpisodicSlot] = []
            for s in self._slots:
                if not s.consolidated and to_remove < trim:
                    to_remove += 1  # discard oldest unconsolidated
                else:
                    kept.append(s)
            self._slots = kept

    def consolidation_cost(self) -> float:
        """Return integrity cost for unconsolidated slots beyond capacity."""
        active = self.active_count()
        overflow = max(0, active - self.capacity)
        return overflow * CONSOLIDATION_COST_PER_SLOT

    def flush_oldest(self, n: int = 5) -> list[EpisodicSlot]:
        """Consolidate the oldest unconsolidated slots.

        Marks up to ``n`` slots as consolidated and returns them for
        optional diary insertion.  Trims the internal list once it grows
        beyond 2× capacity to prevent unbounded memory growth.

        Parameters
        ----------
        n:
            Maximum number of slots to consolidate in one pass.

        Returns
        -------
        list[EpisodicSlot]
            The slots that were consolidated this call.
        """
        flushed: list[EpisodicSlot] = []
        count = 0
        for slot in self._slots:
            if not slot.consolidated and count < n:
                slot.consolidated = True
                flushed.append(slot)
                count += 1

        # Prune when buffer exceeds 2× capacity: single-pass partition into
        # (recent-consolidated, unconsolidated) keeping most-recent consolidated.
        if len(self._slots) > self.capacity * 2:
            consolidated: list[EpisodicSlot] = []
            unconsolidated: list[EpisodicSlot] = []
            for s in self._slots:
                (consolidated if s.consolidated else unconsolidated).append(s)
            self._slots = consolidated[-self.capacity:] + unconsolidated

        return flushed

    def active_count(self) -> int:
        """Number of unconsolidated (live) episodic slots."""
        return sum(1 for s in self._slots if not s.consolidated)

    def total_count(self) -> int:
        """Total slots in buffer (consolidated + unconsolidated)."""
        return len(self._slots)


# ------------------------------------------------------------------ #
# LimbicLayer orchestrator                                            #
# ------------------------------------------------------------------ #


@dataclass
class LimbicSignal:
    """Combined output from all limbic components for a single tick.

    Returned by ``LimbicLayer.process()``; the caller (pulse loop) is
    responsible for applying the contained costs to MetabolicState and
    forwarding precision overrides to the inference / precision engines.
    """

    threat_level: float                              # amygdala threat 0-1
    precision_overrides: dict[str, float]            # additive precision boosts
    efe_discount: float                              # fractional EFE reduction (accumbens)
    integrity_cost: float                            # consolidation overflow cost
    heat_cost: float                                 # amygdala activation heat
    amygdala_affect_modifier: float                  # additional affect adjustment


class LimbicLayer:
    """Orchestrates amygdala, nucleus accumbens, and episodic buffer.

    Usage
    -----
        limbic = LimbicLayer()

        # Each heartbeat
        signal = limbic.process(state)
        state.apply_action_feedback(
            delta_heat=-signal.heat_cost,
            delta_integrity=-signal.integrity_cost,
        )

        # During REST consolidation
        flushed = limbic.consolidate(n=10)
    """

    def __init__(self, episodic_capacity: int = DEFAULT_EPISODIC_CAPACITY) -> None:
        self.amygdala = AmygdalaModule()
        self.accumbens = NucleusAccumbens()
        self.episodic = EpisodicBuffer(capacity=episodic_capacity)

    def process(self, state: MetabolicState) -> LimbicSignal:
        """Evaluate all limbic components and return a combined signal.

        This is read-only with respect to ``state``; all costs are returned
        in the signal for the caller to apply explicitly.

        Parameters
        ----------
        state:
            Current metabolic state (read-only within this method).

        Returns
        -------
        LimbicSignal
            Aggregated precision overrides, EFE discount, and metabolic
            costs to apply this tick.
        """
        amyg = self.amygdala.evaluate(state)
        discount = self.accumbens.efe_discount(state)
        integrity_cost = self.episodic.consolidation_cost()

        return LimbicSignal(
            threat_level=amyg.threat_level,
            precision_overrides=amyg.precision_overrides,
            efe_discount=discount,
            integrity_cost=integrity_cost,
            heat_cost=amyg.heat_cost,
            amygdala_affect_modifier=amyg.affect_modifier,
        )

    def push_episode(
        self,
        tick: int,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a new episodic memory slot to the buffer.

        Parameters
        ----------
        tick:
            Current entropy tick.
        content:
            Human-readable summary of the episode.
        metadata:
            Optional additional data (action, affect, etc.).
        """
        self.episodic.push(
            EpisodicSlot(
                tick=tick,
                content=content,
                metadata=metadata or {},
            )
        )

    def consolidate(self, n: int = 5) -> list[EpisodicSlot]:
        """Consolidate oldest unconsolidated episodes.

        Call during REST / Janitor pass to drain the episodic overflow cost.

        Parameters
        ----------
        n:
            Maximum number of slots to consolidate.

        Returns
        -------
        list[EpisodicSlot]
            The slots that were consolidated, ready for diary insertion.
        """
        return self.episodic.flush_oldest(n)

    def status(self) -> dict[str, Any]:
        """Return a summary dict for HUD / logging."""
        return {
            "episodic_active": self.episodic.active_count(),
            "episodic_total": self.episodic.total_count(),
            "episodic_cost": self.episodic.consolidation_cost(),
        }
