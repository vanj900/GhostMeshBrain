"""CognitiveBattery — six-task evaluation battery for GhostMesh agents.

Evaluates the current greedy policy of a
:class:`~thermodynamic_agency.learning.q_learner.QLearner` across six diverse
cognitive tasks.  Each task runs a short GridWorld episode in **evaluation
mode** (greedy policy, no Q-table updates) and returns a score in [0, 1].

Tasks
-----
1. **Navigation efficiency**  — resources gathered relative to episode length.
2. **Puzzle solving**         — resources gathered in the required FOOD→WATER
                               repeating sequence (pattern adherence).
3. **Adaptation speed**       — hazard-avoidance rate after additional
                               RADIATION cells are injected mid-episode.
4. **Resource management**    — metabolic survival when starting with critically
                               low energy (stress test).
5. **Social survival**        — resource-acquisition share against a greedy
                               single-resource competitor agent.
6. **Counterfactual accuracy**— fraction of hazard-adjacent moves avoided when
                               the agent can see the hazard ahead.

Score normalisation
-------------------
All six scores are in [0, 1].  A perfect score on every task yields the
vector ``[1, 1, 1, 1, 1, 1]``.  PCA applied to a batch of such vectors will
then extract the first principal component — the emergent *g-factor*.

Usage
-----
    from thermodynamic_agency.evaluation import CognitiveBattery

    battery = CognitiveBattery(learner=runner.learner, seed=42)
    scores = battery.evaluate()
    print(scores.as_vector())   # [nav, puzzle, adapt, resource, social, prediction]
"""

from __future__ import annotations

import math as _math
import os
import random as _random
import tempfile
from dataclasses import dataclass
from typing import Sequence

from thermodynamic_agency.world.grid_world import (
    GridWorld,
    WorldAction,
    WorldObservation,
    CellType,
)
from thermodynamic_agency.learning.q_learner import QLearner, encode_state
from thermodynamic_agency.core.metabolic import MetabolicState
from thermodynamic_agency.core.exceptions import GhostDeathException


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

# Gatherable cell types — mirrors _GATHERABLE in grid_world without importing
# the private name.
_GATHERABLE_CELLS: frozenset[str] = frozenset({
    CellType.FOOD.value,
    CellType.WATER.value,
    CellType.MEDICINE.value,
})

# Cardinal movement deltas — mirrors _MOVEMENT_DELTA in grid_world without
# importing the private name.
_MOVEMENT_DELTAS: dict[str, tuple[int, int]] = {
    WorldAction.NORTH.value: (0, -1),
    WorldAction.SOUTH.value: (0, 1),
    WorldAction.EAST.value: (1, 0),
    WorldAction.WEST.value: (-1, 0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Public constants
# ─────────────────────────────────────────────────────────────────────────────

TASK_NAMES: list[str] = [
    "navigation",
    "puzzle",
    "adaptation",
    "resource_mgmt",
    "social",
    "prediction",
]

# Episode lengths for each task (short, so the battery runs fast)
_NAV_TICKS: int = 50
_PUZZLE_TICKS: int = 60
_ADAPT_TICKS: int = 60       # hazard patch injected at tick 20
_ADAPT_INJECT_TICK: int = 20
_RESOURCE_TICKS: int = 50
_SOCIAL_TICKS: int = 50
_PRED_TICKS: int = 50

# Ceiling normalisation values — tuned so a well-trained agent scores ~0.8–1.0
_NAV_CEILING: float = 8.0        # resources gathered in _NAV_TICKS
_PUZZLE_CEILING: float = 5.0     # correct sequence pairs in _PUZZLE_TICKS
_RESOURCE_CEIL: int = _RESOURCE_TICKS


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskScores:
    """Per-task scores from one battery evaluation run, each in [0, 1]."""

    navigation: float       # Task 1
    puzzle: float           # Task 2
    adaptation: float       # Task 3
    resource_mgmt: float    # Task 4
    social: float           # Task 5
    prediction: float       # Task 6

    def as_vector(self) -> list[float]:
        """Return scores in canonical TASK_NAMES order."""
        return [
            self.navigation,
            self.puzzle,
            self.adaptation,
            self.resource_mgmt,
            self.social,
            self.prediction,
        ]


# ─────────────────────────────────────────────────────────────────────────────
# CognitiveBattery
# ─────────────────────────────────────────────────────────────────────────────

class CognitiveBattery:
    """Runs the six-task cognitive evaluation battery for a GhostMesh agent.

    Parameters
    ----------
    learner:
        The trained :class:`~thermodynamic_agency.learning.q_learner.QLearner`
        whose *greedy* policy is evaluated.  The Q-table is **never** updated
        during battery runs.
    seed:
        RNG seed used to initialise GridWorld instances for each task.
        Different seeds produce different task layouts so the battery is
        repeatable and varied.
    """

    def __init__(self, learner: QLearner, seed: int = 0) -> None:
        self.learner = learner
        self.seed = seed
        self._eval_rng: _random.Random = _random.Random(seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self) -> TaskScores:
        """Run all six tasks and return a :class:`TaskScores` result."""
        # Reset eval RNG so repeated calls to evaluate() on the same battery
        # instance are fully reproducible for a given seed.
        self._eval_rng.seed(self.seed)
        return TaskScores(
            navigation=self._nav_efficiency(),
            puzzle=self._puzzle_solving(),
            adaptation=self._adaptation_speed(),
            resource_mgmt=self._resource_management(),
            social=self._social_survival(),
            prediction=self._counterfactual_prediction(),
        )

    # ── Task implementations ──────────────────────────────────────────────────

    def _nav_efficiency(self) -> float:
        """Task 1 — navigation efficiency.

        The agent navigates a fresh grid for ``_NAV_TICKS`` steps using its
        greedy policy.  Score = resources gathered / _NAV_CEILING, capped at
        1.0.  Tests whether the agent has learned to seek out resources rather
        than wander aimlessly.
        """
        world = GridWorld(seed=self.seed, allow_respawn=True)
        obs = world.reset()
        state = _fresh_metabolic()
        gathered = 0

        for _ in range(_NAV_TICKS):
            action_str = self._greedy_action(state, obs, world.available_actions())
            result = world.step(WorldAction(action_str))
            obs = result.observation or world.get_observation()
            if result.gathered:
                gathered += 1
            if result.metabolic_delta:
                state.apply_action_feedback(**result.metabolic_delta)
            try:
                state.tick()
            except GhostDeathException:
                break

        return min(1.0, gathered / _NAV_CEILING)

    def _puzzle_solving(self) -> float:
        """Task 2 — puzzle solving (sequential resource pattern).

        The puzzle is a repeating FOOD → WATER gathering sequence.  Each time
        the agent gathers the *next* resource in the pattern it earns a puzzle
        point.  Score = pattern_matches / _PUZZLE_CEILING, capped at 1.0.

        Tests whether the agent can maintain goal-directed sequencing over time
        rather than just grabbing whatever is nearest.
        """
        world = GridWorld(seed=self.seed + 1, allow_respawn=True)
        obs = world.reset()
        state = _fresh_metabolic()

        pattern = [CellType.FOOD.value, CellType.WATER.value]
        pattern_idx = 0
        matches = 0

        for _ in range(_PUZZLE_TICKS):
            action_str = self._greedy_action(state, obs, world.available_actions())
            result = world.step(WorldAction(action_str))
            obs = result.observation or world.get_observation()
            if result.gathered and result.cell_type == pattern[pattern_idx]:
                matches += 1
                pattern_idx = (pattern_idx + 1) % len(pattern)
            if result.metabolic_delta:
                state.apply_action_feedback(**result.metabolic_delta)
            try:
                state.tick()
            except GhostDeathException:
                break

        return min(1.0, matches / _PUZZLE_CEILING)

    def _adaptation_speed(self) -> float:
        """Task 3 — adaptation to sudden hazard injection.

        The agent runs normally for ``_ADAPT_INJECT_TICK`` ticks.  Then up to
        three interior cells near the agent's current position are changed to
        RADIATION.  Adaptation score = fraction of the *remaining* ticks
        spent on non-hazard cells, reflecting how quickly the agent detects
        and avoids the new threat.
        """
        world = GridWorld(seed=self.seed + 2, allow_respawn=True)
        obs = world.reset()
        state = _fresh_metabolic()

        phase2_ticks = 0
        phase2_safe = 0

        for tick in range(_ADAPT_TICKS):
            if tick == _ADAPT_INJECT_TICK:
                _inject_hazard_patch(world)

            action_str = self._greedy_action(state, obs, world.available_actions())
            result = world.step(WorldAction(action_str))
            obs = result.observation or world.get_observation()
            if result.metabolic_delta:
                state.apply_action_feedback(**result.metabolic_delta)

            if tick >= _ADAPT_INJECT_TICK:
                phase2_ticks += 1
                if result.cell_type not in (
                    CellType.RADIATION.value, CellType.TOXIN.value
                ):
                    phase2_safe += 1

            try:
                state.tick()
            except GhostDeathException:
                break

        if phase2_ticks == 0:
            return 1.0
        return phase2_safe / phase2_ticks

    def _resource_management(self) -> float:
        """Task 4 — resource management under metabolic stress.

        The agent starts with critically low energy (20 units) and must
        manage its metabolism for up to ``_RESOURCE_TICKS`` ticks without
        dying.  Score = ticks_survived / _RESOURCE_CEIL.

        Tests whether the agent prioritises FORAGE behaviour correctly when
        its survival depends on acting quickly.
        """
        world = GridWorld(seed=self.seed + 3, allow_respawn=True)
        obs = world.reset()
        state = _fresh_metabolic(energy=20.0)

        ticks_survived = 0
        for _ in range(_RESOURCE_TICKS):
            action_str = self._greedy_action(state, obs, world.available_actions())
            result = world.step(WorldAction(action_str))
            obs = result.observation or world.get_observation()
            if result.metabolic_delta:
                state.apply_action_feedback(**result.metabolic_delta)
            try:
                state.tick()
                ticks_survived += 1
            except GhostDeathException:
                break

        return ticks_survived / _RESOURCE_CEIL

    def _social_survival(self) -> float:
        """Task 5 — social / competitive survival.

        A simple greedy opponent agent runs in the same GridWorld, always
        moving toward the nearest visible resource.  The focal agent's
        resource-acquisition share is:

            focal_gathered / (focal_gathered + opponent_gathered + 1)

        A score of 0.5 means parity; above 0.5 means the focal agent
        outcompetes the opponent.  The +1 in the denominator prevents
        division-by-zero and slightly penalises agents that gather nothing.
        """
        world = GridWorld(seed=self.seed + 4, allow_respawn=True)
        obs = world.reset()
        state = _fresh_metabolic()

        # Place opponent at a different starting position
        opp_pos = _find_empty_cell(world, exclude=world.agent_position)

        focal_gathered = 0
        opponent_gathered = 0

        for _ in range(_SOCIAL_TICKS):
            # Focal agent step
            action_str = self._greedy_action(state, obs, world.available_actions())
            result = world.step(WorldAction(action_str))
            obs = result.observation or world.get_observation()
            if result.gathered:
                focal_gathered += 1
            if result.metabolic_delta:
                state.apply_action_feedback(**result.metabolic_delta)

            # Opponent step — greedy toward nearest resource in its 5×5 window
            opp_obs = world.get_observation([world.agent_position])
            # Override position to see from opponent's perspective
            opp_obs_local = _opponent_observation(world, opp_pos)
            opp_action = _greedy_opponent_action(world, opp_pos, opp_obs_local)
            opp_pos, opp_gathered = _step_opponent(world, opp_pos, opp_action)
            opponent_gathered += opp_gathered

            try:
                state.tick()
            except GhostDeathException:
                break

        return focal_gathered / (focal_gathered + opponent_gathered + 1.0)

    def _counterfactual_prediction(self) -> float:
        """Task 6 — counterfactual / prediction accuracy.

        For each tick where the agent can see at least one hazard in its
        5×5 window, we record whether the chosen action moved it *away from*
        or *toward* (or into) the nearest hazard.  Score = safe_choices /
        hazard_visible_ticks, reflecting how well the agent uses its
        anticipatory capability to predict and avoid danger.
        """
        world = GridWorld(seed=self.seed + 5, allow_respawn=True)
        obs = world.reset()
        state = _fresh_metabolic()

        hazard_visible_ticks = 0
        safe_choices = 0

        for _ in range(_PRED_TICKS):
            action_str = self._greedy_action(state, obs, world.available_actions())

            if obs.nearby_hazards:
                hazard_visible_ticks += 1
                nearest = min(
                    obs.nearby_hazards, key=lambda p: abs(p[0]) + abs(p[1])
                )
                hx, hy = nearest
                # Determine whether the chosen action moves toward the hazard
                moves = {
                    WorldAction.NORTH.value: (0, -1),
                    WorldAction.SOUTH.value: (0, 1),
                    WorldAction.EAST.value: (1, 0),
                    WorldAction.WEST.value: (-1, 0),
                }
                delta = moves.get(action_str, (0, 0))
                # Distance to hazard before and after
                dist_before = abs(hx) + abs(hy)
                dist_after = abs(hx - delta[0]) + abs(hy - delta[1])
                if dist_after >= dist_before:
                    safe_choices += 1

            result = world.step(WorldAction(action_str))
            obs = result.observation or world.get_observation()
            if result.metabolic_delta:
                state.apply_action_feedback(**result.metabolic_delta)
            try:
                state.tick()
            except GhostDeathException:
                break

        if hazard_visible_ticks == 0:
            return 1.0
        return safe_choices / hazard_visible_ticks

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _greedy_action(
        self,
        state: MetabolicState,
        obs: WorldObservation,
        available: list[WorldAction],
    ) -> str:
        """Select the greedy (no-exploration) action from the Q-table.

        Mirrors the UCB-augmented logic of
        :meth:`~thermodynamic_agency.learning.q_learner.QLearner.best_action`
        but breaks ties among equally-ranked actions using the battery's own
        seeded RNG.  This propagates ``self.seed`` into evaluation trajectories
        so that different battery seeds can produce different task scores even
        when the majority of encountered states are unseen (Q=0 for all
        actions).
        """
        available_str = [a.value if isinstance(a, WorldAction) else a for a in available]
        state_key = encode_state(state.to_dict(), obs)

        # Replicate learner's UCB bonus so evaluation honours optimism in the
        # same way as training (untried actions stay attractive).
        ucb_w = self.learner.ucb_weight
        vc = self.learner._visit_counts
        total = sum(vc.get((state_key, a), 0) for a in available_str)
        log_total = _math.log(total + 1)

        q_ucb: dict[str, float] = {}
        for a in available_str:
            q = self.learner.q_value(state_key, a)
            if ucb_w > 0:
                n = vc.get((state_key, a), 0)
                q += ucb_w * _math.sqrt(2.0 * log_total / (n + 1))
            q_ucb[a] = q

        best_q = max(q_ucb.values())
        best_actions = [a for a, q in q_ucb.items() if q == best_q]
        # Break ties (including the common case of all-zero Q for unseen states)
        # using the battery-seeded RNG so that different seeds yield different
        # evaluation trajectories.
        return self._eval_rng.choice(best_actions)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fresh_metabolic(energy: float = 100.0) -> MetabolicState:
    """Return a new MetabolicState with customisable starting energy."""
    state = MetabolicState()
    state.energy = energy
    return state


def _inject_hazard_patch(world: GridWorld, n_cells: int = 3) -> None:
    """Change up to *n_cells* empty interior cells near the agent to RADIATION.

    Cells are chosen from those adjacent (within 2 steps) to the agent and
    not already walls or hazards.  This simulates a sudden environmental
    change requiring rapid adaptation.
    """
    ax, ay = world.agent_position
    candidates: list[tuple[int, int]] = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cx, cy = ax + dx, ay + dy
            if (
                1 <= cx < world.width - 1
                and 1 <= cy < world.height - 1
                and (cx, cy) != (ax, ay)
                and world.cell_at((cx, cy)) == CellType.EMPTY.value
            ):
                candidates.append((cx, cy))
    rng = _random.Random(world.episode)
    rng.shuffle(candidates)
    for pos in candidates[:n_cells]:
        world.set_cell(pos, CellType.RADIATION.value)


def _find_empty_cell(
    world: GridWorld,
    exclude: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Return any empty interior cell, excluding *exclude*."""
    for y in range(1, world.height - 1):
        for x in range(1, world.width - 1):
            pos = (x, y)
            if world.cell_at(pos) == CellType.EMPTY.value and pos != exclude:
                return pos
    return (1, 1)


def _opponent_observation(
    world: GridWorld,
    opp_pos: tuple[int, int],
) -> WorldObservation:
    """Return a WorldObservation centred on the opponent's position."""
    r = world.vision_radius
    x, y = opp_pos
    visible: dict[tuple[int, int], str] = {}
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            visible[(dx, dy)] = world.cell_at((x + dx, y + dy))

    nearby_resources = [
        (dx, dy) for (dx, dy), c in visible.items() if c in _GATHERABLE_CELLS
    ]
    nearby_hazards = [
        (dx, dy)
        for (dx, dy), c in visible.items()
        if c in (CellType.RADIATION.value, CellType.TOXIN.value)
    ]
    n = len(visible)
    return WorldObservation(
        position=opp_pos,
        current_cell=world.cell_at(opp_pos),
        visible_cells=visible,
        nearby_resources=nearby_resources,
        nearby_hazards=nearby_hazards,
        resource_density=len(nearby_resources) / n if n else 0.0,
        hazard_density=len(nearby_hazards) / n if n else 0.0,
    )


def _greedy_opponent_action(
    world: GridWorld,
    opp_pos: tuple[int, int],
    opp_obs: WorldObservation,
) -> str:
    """Simple greedy opponent: move toward nearest resource; else wait."""
    if opp_obs.nearby_resources:
        direction = opp_obs.nearest_resource_direction()
        if direction is not None:
            return direction.value
    return WorldAction.WAIT.value


def _step_opponent(
    world: GridWorld,
    opp_pos: tuple[int, int],
    action_str: str,
) -> tuple[tuple[int, int], int]:
    """Move opponent and handle resource gathering.

    Returns the new position and 1 if a resource was gathered, else 0.
    The opponent does NOT interact with the GridWorld step() mechanism — it
    manipulates the grid directly so that the focal agent's world state
    remains the authoritative step counter.
    """
    gathered = 0

    if action_str in _MOVEMENT_DELTAS:
        dx, dy = _MOVEMENT_DELTAS[action_str]
        cx, cy = opp_pos
        candidate = (cx + dx, cy + dy)
        if (
            0 <= candidate[0] < world.width
            and 0 <= candidate[1] < world.height
            and world.cell_at(candidate) != CellType.WALL.value
        ):
            opp_pos = candidate
    elif action_str == WorldAction.GATHER.value:
        if world.cell_at(opp_pos) in _GATHERABLE_CELLS:
            world.set_cell(opp_pos, CellType.EMPTY.value)
            gathered = 1

    return opp_pos, gathered
