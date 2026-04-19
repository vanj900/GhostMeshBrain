"""GridWorld — a 2-D environment for embodied GhostMesh.

A 10×10 grid containing resources (FOOD, WATER, MEDICINE) and hazards
(RADIATION, TOXIN) behind wall obstacles.  The agent navigates, gathers
resources, and avoids hazards.  Resources respawn after a configurable
number of ticks so the environment supports indefinite repeated interaction.

Cell effects
------------
FOOD      : GATHER → +15 energy         (consumed; respawns after 20 ticks)
WATER     : GATHER → -12 heat           (consumed; respawns after 20 ticks)
MEDICINE  : GATHER → +10 integrity      (consumed; respawns after 25 ticks)
RADIATION : on entry → +8 heat          (passive hazard, not consumed)
TOXIN     : on entry → +10 waste        (passive hazard, not consumed)
WALL      : impassable
EMPTY     : no effect

The agent sees a 5×5 neighbourhood (radius 2) — partial observability ensures
that exploration is necessary to discover the full layout.

Usage
-----
    world = GridWorld(seed=42)
    obs = world.reset()
    result = world.step(WorldAction.GATHER)
    print(result.metabolic_delta)   # e.g. {"delta_energy": 15.0}
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Cell and action enumerations
# ─────────────────────────────────────────────────────────────────────────────

class CellType(str, Enum):
    EMPTY = "empty"
    FOOD = "food"
    WATER = "water"
    MEDICINE = "medicine"
    RADIATION = "radiation"
    TOXIN = "toxin"
    WALL = "wall"


class WorldAction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    GATHER = "gather"
    WAIT = "wait"
    BROADCAST = "broadcast"  # costly communication; energy + heat charged by caller


# ─────────────────────────────────────────────────────────────────────────────
# Metabolic effects lookup tables
# ─────────────────────────────────────────────────────────────────────────────

# Applied once when agent *enters* a cell (passive hazards only)
_ENTRY_EFFECTS: dict[str, dict[str, float]] = {
    CellType.RADIATION.value: {"delta_heat": 8.0},
    CellType.TOXIN.value: {"delta_waste": 10.0},
}

# Applied when agent executes GATHER on the cell (resources only)
_GATHER_EFFECTS: dict[str, dict[str, float]] = {
    CellType.FOOD.value: {"delta_energy": 15.0},
    CellType.WATER.value: {"delta_heat": -12.0},
    CellType.MEDICINE.value: {"delta_integrity": 10.0},
}

# Respawn delay in ticks for each resource type
_RESPAWN_TICKS: dict[str, int] = {
    CellType.FOOD.value: 20,
    CellType.WATER.value: 20,
    CellType.MEDICINE.value: 25,
}

_GATHERABLE: frozenset[str] = frozenset(
    {CellType.FOOD.value, CellType.WATER.value, CellType.MEDICINE.value}
)

_MOVEMENT_DELTA: dict[str, tuple[int, int]] = {
    WorldAction.NORTH.value: (0, -1),
    WorldAction.SOUTH.value: (0, 1),
    WorldAction.EAST.value: (1, 0),
    WorldAction.WEST.value: (-1, 0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorldObservation:
    """Partial observable state returned to the agent each step.

    The agent sees a 5×5 neighbourhood centred on its current position.
    Cells outside the grid boundary appear as WALL.
    In multi-agent worlds, other agents visible in the window are listed in
    ``nearby_agents`` (relative positions) and flagged as social stressors.
    """

    position: tuple[int, int]
    current_cell: str                           # CellType.value of agent's cell
    visible_cells: dict[tuple[int, int], str]   # (Δx, Δy) → CellType.value
    nearby_resources: list[tuple[int, int]]     # relative positions of resources
    nearby_hazards: list[tuple[int, int]]       # relative positions of hazards
    resource_density: float                     # fraction of visible cells w/ resources
    hazard_density: float                       # fraction of visible cells w/ hazards
    nearby_agents: list[tuple[int, int]] = field(default_factory=list)
    social_stress: float = 0.0                  # 0–1 social stressor level

    def has_resource_here(self) -> bool:
        """True if the agent is standing on a gatherable resource."""
        return self.current_cell in _GATHERABLE

    def has_hazard_here(self) -> bool:
        """True if the agent is standing on a passive hazard."""
        return self.current_cell in (CellType.RADIATION.value, CellType.TOXIN.value)

    def nearest_resource_direction(self) -> Optional["WorldAction"]:
        """Return the cardinal direction toward the nearest visible resource.

        Returns None when no resources are in sight.
        """
        if not self.nearby_resources:
            return None
        nearest = min(self.nearby_resources, key=lambda p: abs(p[0]) + abs(p[1]))
        dx, dy = nearest
        if abs(dx) >= abs(dy):
            return WorldAction.EAST if dx > 0 else WorldAction.WEST
        return WorldAction.SOUTH if dy > 0 else WorldAction.NORTH

    def nearest_hazard_direction(self) -> Optional["WorldAction"]:
        """Return the cardinal direction toward the nearest visible hazard."""
        if not self.nearby_hazards:
            return None
        nearest = min(self.nearby_hazards, key=lambda p: abs(p[0]) + abs(p[1]))
        dx, dy = nearest
        if abs(dx) >= abs(dy):
            return WorldAction.EAST if dx > 0 else WorldAction.WEST
        return WorldAction.SOUTH if dy > 0 else WorldAction.NORTH


@dataclass
class WorldStepResult:
    """Result of one world step (action execution)."""

    action: WorldAction
    new_position: tuple[int, int]
    cell_type: str               # CellType.value of new position
    gathered: bool               # True if a resource was gathered and consumed
    metabolic_delta: dict[str, float]  # feed into MetabolicState.apply_action_feedback()
    observation: WorldObservation | None = None   # observation from new position
    contested: bool = False      # True if another agent grabbed the resource first


# ─────────────────────────────────────────────────────────────────────────────
# Internal respawn timer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _RespawnTimer:
    cell_type: str
    position: tuple[int, int]
    ticks_remaining: int


# ─────────────────────────────────────────────────────────────────────────────
# GridWorld
# ─────────────────────────────────────────────────────────────────────────────

class GridWorld:
    """A 10×10 navigable grid with resources, hazards, and repeatable episodes.

    Parameters
    ----------
    width, height:
        Grid dimensions (default 10×10).
    seed:
        Optional RNG seed for reproducible layouts.  Each call to ``reset()``
        generates a fresh random layout using this seed base + episode count.
    wall_density:
        Fraction of interior cells that become walls (default 0.05).
    vision_radius:
        Half-width of the agent's square observation window (default 2,
        giving a 5×5 view).
    allow_respawn:
        Whether consumed resources respawn after their timer expires.  Set to
        ``False`` to disable respawning entirely (e.g., "Lifeboat Scenario").
    resource_decay_rate:
        When > 0, respawn delays grow linearly with ``world_tick``:
        ``actual_delay = base_delay × (1 + resource_decay_rate × world_tick)``.
        This makes the world progressively more hostile over a long run,
        rewarding anticipatory resource management.  Default 0.0 (no decay).
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        seed: int | None = None,
        wall_density: float = 0.05,
        vision_radius: int = 2,
        allow_respawn: bool = True,
        resource_decay_rate: float = 0.0,
    ) -> None:
        self.width = width
        self.height = height
        self._wall_density = wall_density
        self._vision_radius = vision_radius
        self._base_seed = seed
        self._rng = random.Random(seed)
        self._grid: list[list[str]] = []
        self._agent_pos: tuple[int, int] = (1, 1)
        self._world_tick: int = 0
        self._episode: int = 0
        self._respawn_timers: list[_RespawnTimer] = []
        self.allow_respawn: bool = allow_respawn
        self.resource_decay_rate: float = max(0.0, resource_decay_rate)
        self.reset()

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> WorldObservation:
        """Reset to a new random layout and return the initial observation."""
        self._world_tick = 0
        self._episode += 1
        self._respawn_timers.clear()
        # Re-seed per episode for varied but reproducible layouts
        if self._base_seed is not None:
            self._rng = random.Random(self._base_seed + self._episode)
        self._generate_grid()
        self._agent_pos = self._random_empty_cell()
        return self._make_observation()

    def step(self, action: WorldAction) -> WorldStepResult:
        """Execute one world step and return the result.

        The returned ``metabolic_delta`` should be fed to
        ``MetabolicState.apply_action_feedback()`` by the caller.
        """
        self._world_tick += 1
        self._advance_respawn_timers()

        metabolic_delta: dict[str, float] = {}
        gathered = False

        action_value = action.value if isinstance(action, WorldAction) else action

        if action_value in _MOVEMENT_DELTA:
            dx, dy = _MOVEMENT_DELTA[action_value]
            cx, cy = self._agent_pos
            candidate = (cx + dx, cy + dy)
            if self._is_valid_move(candidate):
                self._agent_pos = candidate
                # Passive entry effects for hazards
                cell = self._cell_at(self._agent_pos)
                _merge_delta(metabolic_delta, _ENTRY_EFFECTS.get(cell, {}))
        elif action_value == WorldAction.GATHER.value:
            cell = self._cell_at(self._agent_pos)
            if cell in _GATHERABLE:
                _merge_delta(metabolic_delta, _GATHER_EFFECTS.get(cell, {}))
                # Consume resource; only schedule respawn when it is enabled
                x, y = self._agent_pos
                self._grid[y][x] = CellType.EMPTY.value
                if self.allow_respawn:
                    base_delay = _RESPAWN_TICKS.get(cell, 20)
                    # Resource decay: respawn takes longer as the world ages,
                    # creating increasing survival pressure over long runs.
                    decay_factor = 1.0 + self.resource_decay_rate * self._world_tick
                    actual_delay = max(1, int(base_delay * decay_factor))
                    self._respawn_timers.append(
                        _RespawnTimer(
                            cell_type=cell,
                            position=self._agent_pos,
                            ticks_remaining=actual_delay,
                        )
                    )
                gathered = True
        # WAIT: no movement, no metabolic effect
        # BROADCAST: no movement; metabolic cost is charged by the caller
        #            (multi-agent runner or agent code). Here we just skip.

        obs = self._make_observation()
        return WorldStepResult(
            action=action if isinstance(action, WorldAction) else WorldAction(action_value),
            new_position=self._agent_pos,
            cell_type=self._cell_at(self._agent_pos),
            gathered=gathered,
            metabolic_delta=metabolic_delta,
            observation=obs,
        )

    def get_observation(
        self,
        other_agent_positions: list[tuple[int, int]] | None = None,
    ) -> WorldObservation:
        """Return the current observation without advancing the world."""
        return self._make_observation(other_agent_positions)

    @property
    def agent_position(self) -> tuple[int, int]:
        return self._agent_pos

    @property
    def world_tick(self) -> int:
        return self._world_tick

    @property
    def episode(self) -> int:
        return self._episode

    @property
    def vision_radius(self) -> int:
        """Half-width of the agent's square observation window."""
        return self._vision_radius

    def cell_at(self, pos: tuple[int, int]) -> str:
        return self._cell_at(pos)

    def set_cell(self, pos: tuple[int, int], cell_type: str) -> None:
        """Set the cell at *pos* to *cell_type*.

        Only interior (non-border) positions may be modified; border cells are
        always walls and cannot be changed.  Raises ``ValueError`` for
        out-of-bounds positions or attempts to overwrite a border wall.

        Parameters
        ----------
        pos:
            ``(x, y)`` grid coordinate.
        cell_type:
            A :class:`CellType` value string (e.g. ``CellType.RADIATION.value``).
        """
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Position {pos} is outside the grid bounds.")
        if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
            raise ValueError(f"Position {pos} is a border wall and cannot be modified.")
        self._grid[y][x] = cell_type

    def available_actions(self) -> list[WorldAction]:
        """Return valid actions from the current position."""
        actions = [WorldAction.WAIT]
        x, y = self._agent_pos
        for action, (dx, dy) in _MOVEMENT_DELTA.items():
            candidate = (x + dx, y + dy)
            if self._is_valid_move(candidate):
                actions.append(WorldAction(action))
        if self._cell_at(self._agent_pos) in _GATHERABLE:
            actions.append(WorldAction.GATHER)
        return actions

    def resource_count(self) -> dict[str, int]:
        """Count each resource type currently on the grid."""
        counts: dict[str, int] = {}
        for row in self._grid:
            for cell in row:
                if cell in _GATHERABLE:
                    counts[cell] = counts.get(cell, 0) + 1
        return counts

    @property
    def world_pressure(self) -> float:
        """Normalised environmental pressure in [0, 1].

        Combines current resource scarcity with the accumulated decay effect
        to produce a single hostility score.  A freshly-initialised world
        with full resources scores near 0; a depleted world with high
        ``resource_decay_rate`` late in a long run scores near 1.

        Use this to monitor whether the world is becoming "alive enough" in
        the sense of creating genuine survival challenge.
        """
        total_interior = (self.width - 2) * (self.height - 2)
        if total_interior <= 0:
            return 0.0
        counts = self.resource_count()
        resources = sum(counts.values())
        # Scarcity fraction: 0 when well-resourced, approaches 1 when empty.
        # Scale so that ~12.5% resource coverage maps to maximum scarcity.
        resource_fraction = resources / total_interior
        scarcity = max(0.0, min(1.0, 1.0 - resource_fraction * 8.0))
        # Decay pressure grows with resource_decay_rate × elapsed world ticks.
        decay_pressure = min(1.0, self.resource_decay_rate * self._world_tick * 0.1)
        return min(1.0, (scarcity + decay_pressure) / 2.0)

    # ── Grid generation ───────────────────────────────────────────────────────

    def _generate_grid(self) -> None:
        self._grid = [
            [CellType.EMPTY.value] * self.width for _ in range(self.height)
        ]

        # Border walls
        for x in range(self.width):
            self._grid[0][x] = CellType.WALL.value
            self._grid[self.height - 1][x] = CellType.WALL.value
        for y in range(self.height):
            self._grid[y][0] = CellType.WALL.value
            self._grid[y][self.width - 1] = CellType.WALL.value

        interior = [
            (x, y)
            for x in range(1, self.width - 1)
            for y in range(1, self.height - 1)
        ]
        self._rng.shuffle(interior)
        n = len(interior)

        counts = {
            CellType.WALL.value:      max(2, int(n * self._wall_density)),
            CellType.FOOD.value:      max(4, int(n * 0.10)),
            CellType.WATER.value:     max(3, int(n * 0.07)),
            CellType.MEDICINE.value:  max(2, int(n * 0.05)),
            CellType.RADIATION.value: max(2, int(n * 0.06)),
            CellType.TOXIN.value:     max(2, int(n * 0.05)),
        }

        idx = 0
        for cell_type, count in counts.items():
            for _ in range(count):
                if idx < len(interior):
                    self._grid[interior[idx][1]][interior[idx][0]] = cell_type
                    idx += 1

    def _random_empty_cell(self) -> tuple[int, int]:
        empties = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self._grid[y][x] == CellType.EMPTY.value
        ]
        return self._rng.choice(empties) if empties else (1, 1)

    def _cell_at(self, pos: tuple[int, int]) -> str:
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height:
            return self._grid[y][x]
        return CellType.WALL.value

    def _is_valid_move(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return self._grid[y][x] != CellType.WALL.value

    def _advance_respawn_timers(self) -> None:
        remaining = []
        for timer in self._respawn_timers:
            timer.ticks_remaining -= 1
            if timer.ticks_remaining <= 0:
                px, py = timer.position
                if self._grid[py][px] == CellType.EMPTY.value:
                    self._grid[py][px] = timer.cell_type
            else:
                remaining.append(timer)
        self._respawn_timers = remaining

    def _make_observation(
        self,
        other_agent_positions: list[tuple[int, int]] | None = None,
    ) -> WorldObservation:
        x, y = self._agent_pos
        r = self._vision_radius
        visible: dict[tuple[int, int], str] = {}
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cell = self._cell_at((x + dx, y + dy))
                visible[(dx, dy)] = cell

        nearby_resources = [
            (dx, dy)
            for (dx, dy), c in visible.items()
            if c in _GATHERABLE
        ]
        nearby_hazards = [
            (dx, dy)
            for (dx, dy), c in visible.items()
            if c in (CellType.RADIATION.value, CellType.TOXIN.value)
        ]
        n_visible = len(visible)
        resource_density = len(nearby_resources) / n_visible if n_visible else 0.0
        hazard_density = len(nearby_hazards) / n_visible if n_visible else 0.0

        # Detect other agents within the vision window
        nearby_agents: list[tuple[int, int]] = []
        if other_agent_positions:
            for ax, ay in other_agent_positions:
                rel = (ax - x, ay - y)
                if rel in visible and rel != (0, 0):
                    nearby_agents.append(rel)
        social_stress = min(1.0, len(nearby_agents) / 4.0) if nearby_agents else 0.0

        return WorldObservation(
            position=self._agent_pos,
            current_cell=self._cell_at(self._agent_pos),
            visible_cells=visible,
            nearby_resources=nearby_resources,
            nearby_hazards=nearby_hazards,
            resource_density=resource_density,
            hazard_density=hazard_density,
            nearby_agents=nearby_agents,
            social_stress=social_stress,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_delta(target: dict[str, float], source: dict[str, float]) -> None:
    for k, v in source.items():
        target[k] = target.get(k, 0.0) + v
