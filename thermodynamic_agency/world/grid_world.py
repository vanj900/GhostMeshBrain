"""GridWorld — a 2-D environment for embodied GhostMesh.

A 30×30 (configurable) grid containing resources, terrain types, and hazards
behind wall obstacles.  The agent navigates, gathers resources, and avoids
hazards.  Resources respawn after a configurable number of ticks.

Cell effects
------------
FOOD      : GATHER → +15 energy         (consumed; respawns after 20 ticks)
WATER     : GATHER → -12 heat           (consumed; respawns after 20 ticks)
MEDICINE  : GATHER → +10 integrity      (consumed; respawns after 25 ticks)
FOREST    : on entry → -3 energy (movement cost); GATHER → +25 energy
            (rich but tiring terrain; respawns after 30 ticks)
ROCKY     : on entry → +6 heat (heat build-up); GATHER → +10 integrity
            (repair-boosting terrain; respawns after 35 ticks)
RADIATION : on entry → +8 heat          (passive hazard, not consumed)
            dwell penalty → +4 heat/tick × 1.2^(dwell-1) when standing still
TOXIN     : on entry → +10 waste        (passive hazard, not consumed)
            dwell penalty → +5 waste/tick × 1.2^(dwell-1) when standing still
MUD       : on entry → -2 energy (slows movement); reduces predator threat by 30%
WALL      : impassable
EMPTY     : no effect

Social actions (multi-agent context)
-------------------------------------
SIGNAL        : cheap broadcast to nearby agents (-1 energy, +1 heat)
COOPERATE     : share energy with nearby agents (-5 energy to sender)
BETRAY        : steal resources from a nearby agent (+8 energy, -reputation)
OBSERVE_OTHER : watch a nearby agent to improve world model (-2 energy, +1 heat)

The agent sees a 5×5 neighbourhood (radius 2) — partial observability ensures
that exploration is necessary to discover the full layout.

WorldEventSystem
----------------
Optionally attached to a GridWorld to fire global events each tick:
- storm        : spikes heat and waste for all exposed agents
- drought      : doubles resource respawn delay for a window
- predator     : adds a predator_threat signal to observations

Seasons cycle every ``season_length`` ticks (default 50), rotating through
SPRING → SUMMER → FALL → WINTER.  Summer doubles forage yield; winter halves it.

Usage
-----
    world = GridWorld(seed=42)
    obs = world.reset()
    result = world.step(WorldAction.GATHER)
    print(result.metabolic_delta)   # e.g. {"delta_energy": 15.0}

    # With event system:
    events = WorldEventSystem(seed=42)
    world_events = events.tick(world.world_tick)
    for agent_state in ...:
        events.apply_to_agent(agent_state, world_events)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Cell and action enumerations
# ─────────────────────────────────────────────────────────────────────────────

class CellType(str, Enum):
    EMPTY = "empty"
    FOOD = "food"
    WATER = "water"
    MEDICINE = "medicine"
    # New terrain types
    FOREST = "forest"   # high-yield but movement costs energy
    ROCKY = "rocky"     # repair-boosting but entry costs heat
    RADIATION = "radiation"
    TOXIN = "toxin"
    WALL = "wall"
    MUD = "mud"         # slows movement (−2 energy); reduces predator threat


class WorldAction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    GATHER = "gather"
    WAIT = "wait"
    BROADCAST = "broadcast"       # costly communication; energy + heat charged by caller
    # Social actions (multi-agent context; metabolic cost charged by runner)
    SIGNAL = "signal"             # cheap broadcast to nearby agents
    COOPERATE = "cooperate"       # share energy with a nearby agent
    BETRAY = "betray"             # steal resources from a nearby agent
    OBSERVE_OTHER = "observe_other"  # learn from watching a nearby agent


# ─────────────────────────────────────────────────────────────────────────────
# Season constants
# ─────────────────────────────────────────────────────────────────────────────

Season = Literal["spring", "summer", "fall", "winter"]
_SEASONS: list[Season] = ["spring", "summer", "fall", "winter"]

# Forage-yield multiplier per season (applied to energy delta on GATHER of FOOD/FOREST)
_SEASON_FORAGE_MULTIPLIER: dict[str, float] = {
    "spring": 1.0,
    "summer": 2.0,   # summer = seasonal burst
    "fall":   0.75,
    "winter": 0.5,   # winter drought
}

# Respawn delay multiplier per season (higher = slower regrowth = more pressure)
_SEASON_RESPAWN_MULTIPLIER: dict[str, float] = {
    "spring": 1.0,
    "summer": 0.8,   # fast growth in summer
    "fall":   1.25,
    "winter": 2.0,   # very slow regrowth in winter
}


# ─────────────────────────────────────────────────────────────────────────────
# Metabolic effects lookup tables
# ─────────────────────────────────────────────────────────────────────────────

# Applied once when agent *enters* a cell (passive hazards and terrain costs)
_ENTRY_EFFECTS: dict[str, dict[str, float]] = {
    CellType.RADIATION.value: {"delta_heat": 8.0},
    CellType.TOXIN.value: {"delta_waste": 10.0},
    # Terrain movement costs
    CellType.FOREST.value: {"delta_energy": -3.0},   # dense undergrowth tires the agent
    CellType.ROCKY.value:  {"delta_heat": 6.0},       # exertion on rocky terrain builds heat
    CellType.MUD.value:    {"delta_energy": -2.0},    # slogging through mud costs energy
}

# Per-tick dwell penalty applied when agent stays on a hazard cell without moving.
# Scales with consecutive dwell ticks: base × dwell_factor^(dwell_ticks-1).
_DWELL_EFFECTS: dict[str, dict[str, float]] = {
    CellType.RADIATION.value: {"delta_heat": 4.0},    # prolonged radiation exposure
    CellType.TOXIN.value: {"delta_waste": 5.0},       # toxin accumulates in tissue
}
_DWELL_SCALE: float = 1.2   # per additional dwell tick above the first

# Predator-threat damping on MUD terrain (fraction subtracted from raw threat level).
_MUD_PREDATOR_DAMPING: float = 0.30

# Heat-bearing cells that contribute to body_temp_external ambient reading
_HEAT_BEARING_CELLS: frozenset[str] = frozenset(
    {CellType.RADIATION.value, CellType.ROCKY.value}
)

# Applied when agent executes GATHER on the cell (resources only)
# Note: season multipliers are applied at runtime by GridWorld.step()
_GATHER_EFFECTS: dict[str, dict[str, float]] = {
    CellType.FOOD.value:     {"delta_energy": 15.0},
    CellType.WATER.value:    {"delta_heat": -12.0},
    CellType.MEDICINE.value: {"delta_integrity": 10.0},
    CellType.FOREST.value:   {"delta_energy": 25.0},   # rich forage yield
    CellType.ROCKY.value:    {"delta_integrity": 10.0}, # mineral repair bonus
}

# Respawn delay in ticks for each resource type (base values; multiplied by season)
_RESPAWN_TICKS: dict[str, int] = {
    CellType.FOOD.value:     20,
    CellType.WATER.value:    20,
    CellType.MEDICINE.value: 25,
    CellType.FOREST.value:   30,
    CellType.ROCKY.value:    35,
}

_GATHERABLE: frozenset[str] = frozenset(
    {
        CellType.FOOD.value,
        CellType.WATER.value,
        CellType.MEDICINE.value,
        CellType.FOREST.value,
        CellType.ROCKY.value,
    }
)

_MOVEMENT_DELTA: dict[str, tuple[int, int]] = {
    WorldAction.NORTH.value: (0, -1),
    WorldAction.SOUTH.value: (0, 1),
    WorldAction.EAST.value: (1, 0),
    WorldAction.WEST.value: (-1, 0),
}

# Cell types whose GATHER delta_energy is scaled by season forage multiplier.
# Only energy-bearing cells are affected; heat/integrity effects are season-neutral.
_SEASON_AFFECTED_CELLS: frozenset[str] = frozenset(
    {CellType.FOOD.value, CellType.FOREST.value}
)

# world_pressure constants
# At ~12.5% resource coverage (1/8 of interior cells) scarcity saturates to 1.
_SCARCITY_SCALING_FACTOR: float = 8.0
# Normalises the resource_decay_rate × world_tick product into [0, 1] range.
_DECAY_PRESSURE_SCALING: float = 0.1


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

    New fields (terrain / events)
    ------------------------------
    predator_threat:
        Normalised threat level [0, 1] from the current ``WorldEventSystem``
        predator event.  Triggers amygdala / SalienceNet in cognition.
    active_world_event:
        Human-readable label of the current global world event
        (e.g. ``"storm"``, ``"drought"``, ``"predator"``) or ``""`` if none.
    terrain_ahead:
        Cell type strings for the 3 cells directly north from the agent's
        position (relative offsets (0,-1), (0,-2), (0,-3)).  Cells outside
        the grid boundary are reported as WALL.
    scent_gradient:
        Normalised signal [0, 1] indicating proximity to the nearest
        food-bearing cell (FOOD or FOREST) within the visible window.
        0.0 = no food visible; 1.0 = food on the agent's current cell.
        Falls off as 1 / (1 + min_manhattan_distance).
    body_temp_external:
        Ambient thermal load from nearby heat-producing cells (RADIATION,
        ROCKY) in the visible window.  Normalised to [0, 1].
    novel_hazard_active:
        True when a NOVEL_HAZARD world event is currently active.  Guardian-
        mode agents should treat this as a high-cost signal.
    windfall_active:
        True when a WINDFALL world event is currently active.  Only plastic
        agents (Dreamer / Courier masks) can harvest the bonus resources.
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
    predator_threat: float = 0.0               # 0–1 amygdala threat from predator event
    active_world_event: str = ""               # current global event label or ""
    # ── Richer sensor suite ──────────────────────────────────────────────
    terrain_ahead: list[str] = field(default_factory=list)  # 3-cell north lookahead
    scent_gradient: float = 0.0               # 0-1 proximity to nearest food
    body_temp_external: float = 0.0           # 0-1 ambient heat from nearby terrain
    novel_hazard_active: bool = False         # NOVEL_HAZARD world event flag
    windfall_active: bool = False             # WINDFALL world event flag

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
    """A navigable grid with resources, terrain types, hazards, and repeatable episodes.

    Parameters
    ----------
    width, height:
        Grid dimensions (default 30×30).
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
        Negative values are silently clamped to 0.0.
    season_length:
        Number of ticks per season.  Seasons cycle SPRING→SUMMER→FALL→WINTER
        and affect forage yield (summer burst, winter drought) and respawn
        rates.  Default 50.  Set 0 to disable seasons.
    """

    def __init__(
        self,
        width: int = 30,
        height: int = 30,
        seed: int | None = None,
        wall_density: float = 0.05,
        vision_radius: int = 2,
        allow_respawn: bool = True,
        resource_decay_rate: float = 0.0,
        season_length: int = 50,
        procedural_terrain: bool = True,
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
        # Seasonal dynamics
        self.season_length: int = max(0, season_length)
        self._season_index: int = 0   # index into _SEASONS
        # Procedural terrain clustering (cellular-automaton growth pass)
        self._procedural_terrain: bool = procedural_terrain
        # Dwell-time tracking: counts consecutive ticks agent stays on a hazard cell
        self._dwell_ticks: int = 0
        self.reset()

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> WorldObservation:
        """Reset to a new random layout and return the initial observation."""
        self._world_tick = 0
        self._season_index = 0
        self._episode += 1
        self._respawn_timers.clear()
        self._dwell_ticks = 0
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
        self._advance_season()
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
                # Reset dwell counter on successful movement
                self._dwell_ticks = 0
                # Passive entry effects for hazards and terrain
                cell = self._cell_at(self._agent_pos)
                _merge_delta(metabolic_delta, _ENTRY_EFFECTS.get(cell, {}))
            # else: bumped into wall — dwell remains unchanged (walls are not
            # in _DWELL_EFFECTS so the counter will reset to 0 on the next line anyway)
        elif action_value == WorldAction.GATHER.value:
            cell = self._cell_at(self._agent_pos)
            if cell in _GATHERABLE:
                base_effects = _GATHER_EFFECTS.get(cell, {})
                scaled = self._apply_season_to_gather(cell, base_effects)
                _merge_delta(metabolic_delta, scaled)
                # Consume resource; only schedule respawn when it is enabled
                x, y = self._agent_pos
                self._grid[y][x] = CellType.EMPTY.value
                if self.allow_respawn:
                    base_delay = _RESPAWN_TICKS.get(cell, 20)
                    # Resource decay: respawn takes longer as the world ages
                    decay_factor = 1.0 + self.resource_decay_rate * self._world_tick
                    # Season also affects respawn speed
                    season_mult = _SEASON_RESPAWN_MULTIPLIER.get(self.current_season, 1.0)
                    actual_delay = max(1, int(base_delay * decay_factor * season_mult))
                    self._respawn_timers.append(
                        _RespawnTimer(
                            cell_type=cell,
                            position=self._agent_pos,
                            ticks_remaining=actual_delay,
                        )
                    )
                gathered = True
        # WAIT / BROADCAST / social actions: no movement

        # Dwell-time hazard penalty: accumulate extra damage when standing still on hazards.
        current_cell = self._cell_at(self._agent_pos)
        if current_cell in _DWELL_EFFECTS:
            self._dwell_ticks += 1
            if self._dwell_ticks > 1:
                # Scale penalty exponentially with consecutive dwell ticks
                scale = _DWELL_SCALE ** (self._dwell_ticks - 1)
                for k, v in _DWELL_EFFECTS[current_cell].items():
                    metabolic_delta[k] = metabolic_delta.get(k, 0.0) + v * scale
        else:
            self._dwell_ticks = 0

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
        predator_threat: float = 0.0,
        active_world_event: str = "",
        novel_hazard_active: bool = False,
        windfall_active: bool = False,
    ) -> WorldObservation:
        """Return the current observation without advancing the world.

        Parameters
        ----------
        other_agent_positions:
            Positions of other living agents (multi-agent context).
        predator_threat:
            Current predator threat level from ``WorldEventSystem`` [0, 1].
        active_world_event:
            Label of the current global event (e.g. ``"storm"``) or ``""``.
        novel_hazard_active:
            Whether a NOVEL_HAZARD event is currently firing.
        windfall_active:
            Whether a WINDFALL event is currently firing.
        """
        return self._make_observation(
            other_agent_positions,
            predator_threat=predator_threat,
            active_world_event=active_world_event,
            novel_hazard_active=novel_hazard_active,
            windfall_active=windfall_active,
        )

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

    @property
    def current_season(self) -> Season:
        """The current season label (spring/summer/fall/winter)."""
        return _SEASONS[self._season_index % 4]

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
        scarcity = max(0.0, min(1.0, 1.0 - resource_fraction * _SCARCITY_SCALING_FACTOR))
        # Decay pressure grows with resource_decay_rate × elapsed world ticks.
        decay_pressure = min(1.0, self.resource_decay_rate * self._world_tick * _DECAY_PRESSURE_SCALING)
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
            # New terrain types
            CellType.FOREST.value:    max(2, int(n * 0.06)),
            CellType.ROCKY.value:     max(2, int(n * 0.04)),
            CellType.RADIATION.value: max(2, int(n * 0.06)),
            CellType.TOXIN.value:     max(2, int(n * 0.05)),
            CellType.MUD.value:       max(2, int(n * 0.04)),
        }

        idx = 0
        for cell_type, count in counts.items():
            for _ in range(count):
                if idx < len(interior):
                    self._grid[interior[idx][1]][interior[idx][0]] = cell_type
                    idx += 1

        # Procedural terrain clustering: one pass of cellular-automaton growth.
        # Each empty interior cell that has ≥ 2 neighbours of the same terrain
        # type has a 30% chance of adopting that type, creating natural patches.
        if self._procedural_terrain:
            self._cluster_terrain(passes=2, adoption_prob=0.30, min_neighbours=2)

    def _cluster_terrain(
        self,
        passes: int = 2,
        adoption_prob: float = 0.30,
        min_neighbours: int = 2,
    ) -> None:
        """Grow terrain patches by cellular-automaton neighbour diffusion.

        For each pass, iterates over empty interior cells.  If at least
        ``min_neighbours`` of the 4-directional neighbours share the same
        clusterable terrain type, the empty cell adopts that type with
        probability ``adoption_prob``.

        Only terrain that should appear in patches is eligible for spreading;
        walls and pure hazards (RADIATION, TOXIN) are excluded to keep hazards
        as isolated danger spots rather than large zones.
        """
        _clusterable: frozenset[str] = frozenset({
            CellType.FOREST.value,
            CellType.ROCKY.value,
            CellType.MUD.value,
        })

        for _ in range(passes):
            candidates = [
                (x, y)
                for x in range(1, self.width - 1)
                for y in range(1, self.height - 1)
                if self._grid[y][x] == CellType.EMPTY.value
            ]
            self._rng.shuffle(candidates)
            for x, y in candidates:
                neighbour_types: dict[str, int] = {}
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    nc = self._grid[y + dy][x + dx]
                    if nc in _clusterable:
                        neighbour_types[nc] = neighbour_types.get(nc, 0) + 1
                if not neighbour_types:
                    continue
                dominant = max(neighbour_types, key=lambda k: neighbour_types[k])
                if (
                    neighbour_types[dominant] >= min_neighbours
                    and self._rng.random() < adoption_prob
                ):
                    self._grid[y][x] = dominant

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

    def _advance_season(self) -> None:
        """Advance the season index when the season_length threshold is crossed."""
        if self.season_length > 0 and self._world_tick > 0:
            new_index = (self._world_tick // self.season_length) % 4
            self._season_index = new_index

    def _apply_season_to_gather(
        self, cell: str, base_effects: dict[str, float]
    ) -> dict[str, float]:
        """Return gather effects with season forage multiplier applied.

        Only ``delta_energy`` on cells in ``_SEASON_AFFECTED_CELLS`` is scaled;
        other effects (heat reduction on WATER, integrity on MEDICINE/ROCKY) are
        season-neutral.
        """
        if cell not in _SEASON_AFFECTED_CELLS:
            return dict(base_effects)
        mult = _SEASON_FORAGE_MULTIPLIER.get(self.current_season, 1.0)
        return {
            k: (v * mult if k == "delta_energy" else v)
            for k, v in base_effects.items()
        }

    def _make_observation(
        self,
        other_agent_positions: list[tuple[int, int]] | None = None,
        predator_threat: float = 0.0,
        active_world_event: str = "",
        novel_hazard_active: bool = False,
        windfall_active: bool = False,
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

        # Dampen predator threat when agent is on MUD (cover mechanic)
        effective_predator_threat = predator_threat
        if self._cell_at(self._agent_pos) == CellType.MUD.value:
            effective_predator_threat = max(0.0, predator_threat - _MUD_PREDATOR_DAMPING)

        # ── Richer sensor suite ──────────────────────────────────────────────
        # terrain_ahead: 3-cell north lookahead (offsets (0,-1), (0,-2), (0,-3))
        terrain_ahead = [
            self._cell_at((x, y - 1)),
            self._cell_at((x, y - 2)),
            self._cell_at((x, y - 3)),
        ]

        # scent_gradient: 1/(1 + min_manhattan_distance_to_food)
        _food_cells: frozenset[str] = frozenset({CellType.FOOD.value, CellType.FOREST.value})
        if self._cell_at(self._agent_pos) in _food_cells:
            scent_gradient = 1.0
        else:
            food_dists = [
                abs(dx) + abs(dy)
                for (dx, dy), c in visible.items()
                if c in _food_cells
            ]
            scent_gradient = 1.0 / (1.0 + min(food_dists)) if food_dists else 0.0

        # body_temp_external: normalised ambient heat from nearby RADIATION/ROCKY cells.
        # Each heat-bearing cell contributes 1/(1 + manhattan_distance) to the sum.
        # Maximum theoretical value is bounded by the number of visible cells.
        heat_sum = sum(
            1.0 / (1.0 + abs(dx) + abs(dy))
            for (dx, dy), c in visible.items()
            if c in _HEAT_BEARING_CELLS
        )
        # Normalise against a soft maximum (~3.0 = very surrounded by hot terrain)
        body_temp_external = min(1.0, heat_sum / 3.0)

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
            predator_threat=effective_predator_threat,
            active_world_event=active_world_event,
            terrain_ahead=terrain_ahead,
            scent_gradient=scent_gradient,
            body_temp_external=body_temp_external,
            novel_hazard_active=novel_hazard_active,
            windfall_active=windfall_active,
        )


# ─────────────────────────────────────────────────────────────────────────────
# WorldEventSystem — global stochastic events shared across all agents
# ─────────────────────────────────────────────────────────────────────────────

# Metabolic deltas applied to each agent when a storm fires
_STORM_HEAT_HIT: float = 12.0
_STORM_WASTE_HIT: float = 8.0

# Predator event parameters
_PREDATOR_THREAT_LEVEL: float = 0.75   # fixed threat signal during predator event
_PREDATOR_MIN_DURATION: int = 5
_PREDATOR_MAX_DURATION: int = 15

# Drought parameters (multiplier applied to respawn delays)
_DROUGHT_RESPAWN_MULTIPLIER: float = 3.0
_DROUGHT_MIN_DURATION: int = 20
_DROUGHT_MAX_DURATION: int = 50

# Novel Hazard parameters
# A rare, high-chaos event requiring plastic (Dreamer/Courier) response.
# Guardian-mode agents take extra damage; ~1 per 200 ticks.
_NOVEL_HAZARD_PROB: float = 0.005        # per-tick probability
_NOVEL_HAZARD_MIN_DURATION: int = 5
_NOVEL_HAZARD_MAX_DURATION: int = 15
# Extra penalty for Guardian/SalienceNet agents during the event (applied by caller)
_NOVEL_HAZARD_GUARDIAN_HEAT: float = 10.0
_NOVEL_HAZARD_GUARDIAN_WASTE: float = 6.0

# Windfall parameters
# A rare resource surge only exploitable by plastic (Dreamer/Courier) agents.
# ~1 per 500 ticks.
_WINDFALL_PROB: float = 0.002            # per-tick probability
_WINDFALL_MIN_DURATION: int = 5
_WINDFALL_MAX_DURATION: int = 10
# Bonus metabolic reward for Dreamer/Courier agents during windfall (applied by caller)
_WINDFALL_ENERGY_BONUS: float = 20.0

# Mask classification for novel-hazard / windfall mechanics (mirrors CollapseProbe)
_GUARDIAN_MODE_MASKS: frozenset[str] = frozenset({"Guardian", "SalienceNet"})
_PLASTIC_MODE_MASKS: frozenset[str] = frozenset({"Dreamer", "Courier", "DefaultMode"})


@dataclass
class WorldEvent:
    """Description of a global world event for the current tick.

    Attributes
    ----------
    label:
        Short identifier (``"storm"``, ``"drought"``, ``"predator"``,
        ``"novel_hazard"``, ``"windfall"``, or ``""``).
    storm_hit:
        When True, each agent should receive ``delta_heat=_STORM_HEAT_HIT``
        and ``delta_waste=_STORM_WASTE_HIT``.
    predator_threat:
        Normalised predator threat level [0, 1] for this tick.
    drought_active:
        Whether a drought is currently suppressing resource respawn.
    novel_hazard_active:
        When True, Guardian-mode agents should apply the novel-hazard penalty
        (``delta_heat=_NOVEL_HAZARD_GUARDIAN_HEAT, delta_waste=_NOVEL_HAZARD_GUARDIAN_WASTE``).
        Plastic-mode agents (Dreamer/Courier) can navigate this event without
        additional cost.
    windfall_active:
        When True, plastic-mode agents (Dreamer/Courier/DefaultMode) should
        receive a resource bonus (``delta_energy=_WINDFALL_ENERGY_BONUS``).
        Guardian-mode agents miss the windfall — their conservative strategy
        cannot adapt quickly enough.
    """

    label: str = ""
    storm_hit: bool = False
    predator_threat: float = 0.0
    drought_active: bool = False
    novel_hazard_active: bool = False
    windfall_active: bool = False


class WorldEventSystem:
    """Global stochastic event generator for shared GridWorld environments.

    Fires storms, droughts, predator events, novel hazards, and windfall
    events that affect all agents simultaneously, forcing coordinated
    adaptation rather than independent optimisation.

    Parameters
    ----------
    storm_prob:
        Per-tick probability of a storm event (default 0.02).
    predator_prob:
        Per-tick probability of a predator appearing (default 0.015).
    drought_prob:
        Per-tick probability of a drought beginning (default 0.01).
    novel_hazard_prob:
        Per-tick probability of a NOVEL_HAZARD event beginning (~1/200 ticks,
        default 0.005).  Guardian-mode agents take extra damage; plastic agents
        are unaffected.
    windfall_prob:
        Per-tick probability of a WINDFALL event beginning (~1/500 ticks,
        default 0.002).  Only plastic-mode agents receive the energy bonus.
    seed:
        RNG seed for reproducibility.

    Usage
    -----
        events = WorldEventSystem(seed=42)
        for tick in range(max_ticks):
            world_event = events.tick(tick)
            if world_event.storm_hit:
                for agent in agents:
                    agent.state.apply_action_feedback(
                        delta_heat=_STORM_HEAT_HIT,
                        delta_waste=_STORM_WASTE_HIT,
                    )
            # Apply novel-hazard penalty to Guardian-mode agents
            if world_event.novel_hazard_active:
                for agent in agents:
                    if agent.mask in _GUARDIAN_MODE_MASKS:
                        agent.state.apply_action_feedback(
                            delta_heat=_NOVEL_HAZARD_GUARDIAN_HEAT,
                            delta_waste=_NOVEL_HAZARD_GUARDIAN_WASTE,
                        )
            # Reward plastic agents during windfall
            if world_event.windfall_active:
                for agent in agents:
                    if agent.mask in _PLASTIC_MODE_MASKS:
                        agent.state.apply_action_feedback(
                            delta_energy=_WINDFALL_ENERGY_BONUS,
                        )
    """

    def __init__(
        self,
        storm_prob: float = 0.02,
        predator_prob: float = 0.015,
        drought_prob: float = 0.01,
        novel_hazard_prob: float = _NOVEL_HAZARD_PROB,
        windfall_prob: float = _WINDFALL_PROB,
        seed: int | None = None,
    ) -> None:
        self.storm_prob = storm_prob
        self.predator_prob = predator_prob
        self.drought_prob = drought_prob
        self.novel_hazard_prob = novel_hazard_prob
        self.windfall_prob = windfall_prob
        self._rng = random.Random(seed)

        # Predator state
        self._predator_active: bool = False
        self._predator_ticks_remaining: int = 0

        # Drought state
        self._drought_active: bool = False
        self._drought_ticks_remaining: int = 0

        # Novel hazard state
        self._novel_hazard_active: bool = False
        self._novel_hazard_ticks_remaining: int = 0

        # Windfall state
        self._windfall_active: bool = False
        self._windfall_ticks_remaining: int = 0

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def drought_active(self) -> bool:
        """True if a drought is currently suppressing resource respawn."""
        return self._drought_active

    @property
    def predator_active(self) -> bool:
        """True if a predator event is ongoing."""
        return self._predator_active

    @property
    def novel_hazard_active(self) -> bool:
        """True if a NOVEL_HAZARD event is ongoing."""
        return self._novel_hazard_active

    @property
    def windfall_active(self) -> bool:
        """True if a WINDFALL event is ongoing."""
        return self._windfall_active

    def tick(self, world_tick: int = 0) -> WorldEvent:  # noqa: ARG002
        """Advance the event system by one tick and return the current event.

        Parameters
        ----------
        world_tick:
            The current world tick (unused internally, kept for call-site clarity).

        Returns
        -------
        WorldEvent
            Describes what is happening globally this tick.
        """
        # ── Storm ────────────────────────────────────────────────────────
        storm_hit = self._rng.random() < self.storm_prob

        # ── Predator ─────────────────────────────────────────────────────
        if self._predator_active:
            self._predator_ticks_remaining -= 1
            if self._predator_ticks_remaining <= 0:
                self._predator_active = False
        elif self._rng.random() < self.predator_prob:
            self._predator_active = True
            self._predator_ticks_remaining = self._rng.randint(
                _PREDATOR_MIN_DURATION, _PREDATOR_MAX_DURATION
            )

        # ── Drought ──────────────────────────────────────────────────────
        if self._drought_active:
            self._drought_ticks_remaining -= 1
            if self._drought_ticks_remaining <= 0:
                self._drought_active = False
        elif self._rng.random() < self.drought_prob:
            self._drought_active = True
            self._drought_ticks_remaining = self._rng.randint(
                _DROUGHT_MIN_DURATION, _DROUGHT_MAX_DURATION
            )

        # ── Novel Hazard ──────────────────────────────────────────────────
        if self._novel_hazard_active:
            self._novel_hazard_ticks_remaining -= 1
            if self._novel_hazard_ticks_remaining <= 0:
                self._novel_hazard_active = False
        elif self._rng.random() < self.novel_hazard_prob:
            self._novel_hazard_active = True
            self._novel_hazard_ticks_remaining = self._rng.randint(
                _NOVEL_HAZARD_MIN_DURATION, _NOVEL_HAZARD_MAX_DURATION
            )

        # ── Windfall ─────────────────────────────────────────────────────
        if self._windfall_active:
            self._windfall_ticks_remaining -= 1
            if self._windfall_ticks_remaining <= 0:
                self._windfall_active = False
        elif self._rng.random() < self.windfall_prob:
            self._windfall_active = True
            self._windfall_ticks_remaining = self._rng.randint(
                _WINDFALL_MIN_DURATION, _WINDFALL_MAX_DURATION
            )

        # ── Compose event label ───────────────────────────────────────────
        labels = []
        if storm_hit:
            labels.append("storm")
        if self._predator_active:
            labels.append("predator")
        if self._drought_active:
            labels.append("drought")
        if self._novel_hazard_active:
            labels.append("novel_hazard")
        if self._windfall_active:
            labels.append("windfall")
        label = "+".join(labels)

        return WorldEvent(
            label=label,
            storm_hit=storm_hit,
            predator_threat=_PREDATOR_THREAT_LEVEL if self._predator_active else 0.0,
            drought_active=self._drought_active,
            novel_hazard_active=self._novel_hazard_active,
            windfall_active=self._windfall_active,
        )

    def apply_to_agent(
        self,
        agent_state: object,
        world_event: WorldEvent,
        mask_name: str = "",
    ) -> None:
        """Apply event effects to a single agent's metabolic state.

        This is a convenience helper for callers that want a single dispatch
        point rather than checking each event flag manually.

        Parameters
        ----------
        agent_state:
            Any object with an ``apply_action_feedback(**kwargs)`` method
            (typically ``MetabolicState``).
        world_event:
            The ``WorldEvent`` returned by ``tick()`` for the current tick.
        mask_name:
            The agent's currently active personality mask name (e.g.
            ``"Guardian"``, ``"Dreamer"``).  Used to gate mask-dependent
            novel-hazard and windfall effects.
        """
        if world_event.storm_hit:
            agent_state.apply_action_feedback(  # type: ignore[union-attr]
                delta_heat=_STORM_HEAT_HIT,
                delta_waste=_STORM_WASTE_HIT,
            )

        if world_event.novel_hazard_active and mask_name in _GUARDIAN_MODE_MASKS:
            # Guardian-mode agents are unprepared for novel threats
            agent_state.apply_action_feedback(  # type: ignore[union-attr]
                delta_heat=_NOVEL_HAZARD_GUARDIAN_HEAT,
                delta_waste=_NOVEL_HAZARD_GUARDIAN_WASTE,
            )

        if world_event.windfall_active and mask_name in _PLASTIC_MODE_MASKS:
            # Plastic agents seize the rare resource surge
            agent_state.apply_action_feedback(  # type: ignore[union-attr]
                delta_energy=_WINDFALL_ENERGY_BONUS,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_delta(target: dict[str, float], source: dict[str, float]) -> None:
    for k, v in source.items():
        target[k] = target.get(k, 0.0) + v
