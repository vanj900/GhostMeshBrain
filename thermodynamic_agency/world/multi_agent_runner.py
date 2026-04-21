"""MultiAgentRunner — simultaneous multi-agent GridWorld simulation (Phase 6).

Runs N agents in a *shared* GridWorld.  Resources are first-come-first-served:
if two agents attempt to GATHER the same cell in the same tick, the agent with
the lower index wins; the loser receives a ``contested`` signal and the
Competition Penalty in their reward.

Communication
-------------
Any agent can issue a ``broadcast(message)`` call.  Every broadcast costs the
*sender* a fixed energy + heat penalty (``BROADCAST_ENERGY_COST``,
``BROADCAST_HEAT_COST``).  Silence is free; talking is expensive.  Received
messages are queued per agent so the caller can log them or pass them to
higher cognition.

Social Actions
--------------
Beyond BROADCAST, agents can perform:

SIGNAL        — cheap broadcast to nearby agents (-1 energy, +1 heat).
COOPERATE     — share energy with all nearby agents (costs sender 5 energy;
                each visible neighbour gains 3 energy).  Increases sender
                reputation.
BETRAY        — steal from the nearest visible agent (+8 energy to actor;
                -4 energy to victim).  Decreases actor reputation.
OBSERVE_OTHER — watch a nearby agent to learn (-2 energy, +1 heat; grants
                a small Q-table visit-count boost for the actor).

Reputation System
-----------------
A shared ``ReputationSystem`` tracks each agent's trust score [0, 1].
COOPERATE raises reputation by 0.1; BETRAY lowers it by 0.2.  Scores are
exposed in ``AgentResult.final_reputation`` and can be read by other agents
through their observation.

Social Stressors
----------------
Agents observe other agents within their 5×5 vision window.  The
``WorldObservation.social_stress`` scalar (0–1) reflects how crowded the
neighbourhood is.  The runner logs this as a ``"Social Stressor"`` diary
entry when ``social_stress > SOCIAL_STRESS_THRESHOLD``.

Global World Events
-------------------
A ``WorldEventSystem`` fires storms (heat+waste spikes to all agents),
droughts (slow resource respawn), and predator events (amygdala threat
signal in observations).

Diverse Personalities
---------------------
Agents are spawned with distinct starting masks to seed different behavioural
archetypes:  Guardian, Dreamer, Courier, Healer, Judge (cycling for N > 5).

Lifeboat Scenario
-----------------
Pass ``respawn=False`` to disable resource respawning — the arena runs to
exhaustion, testing which ethics profile survives.

Usage
-----
    from thermodynamic_agency.world.multi_agent_runner import MultiAgentRunner

    runner = MultiAgentRunner(n_agents=4, seed=42, respawn=True)
    results = runner.run(max_ticks=300)
    for i, res in enumerate(results):
        print(f"Agent {i}: survived={res.survived} ticks={res.ticks_alive} "
              f"reputation={res.final_reputation:.2f}")
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from thermodynamic_agency.world.grid_world import (
    GridWorld,
    WorldAction,
    WorldObservation,
    WorldStepResult,
    WorldEvent,
    WorldEventSystem,
    CellType,
    _GATHERABLE,  # noqa: PLC2701  (internal constant, needed for contention logic)
    _RESPAWN_TICKS,
    _STORM_HEAT_HIT,
    _STORM_WASTE_HIT,
)
from thermodynamic_agency.learning.reward import compute_reward
from thermodynamic_agency.learning.q_learner import QLearner, encode_state
from thermodynamic_agency.pulse import GhostMesh
from thermodynamic_agency.core.exceptions import GhostDeathException


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Energy + heat charged to the *sender* of a broadcast message.
BROADCAST_ENERGY_COST: float = float(os.environ.get("BROADCAST_E_COST", "3.0"))
BROADCAST_HEAT_COST: float = float(os.environ.get("BROADCAST_H_COST", "2.0"))

# Social action costs / rewards
SIGNAL_ENERGY_COST: float = 1.0
SIGNAL_HEAT_COST: float = 1.0

COOPERATE_SENDER_COST: float = 5.0    # energy taken from cooperating agent
COOPERATE_RECEIVER_GAIN: float = 3.0  # energy given to each visible neighbour

BETRAY_ACTOR_GAIN: float = 8.0        # energy stolen from victim
BETRAY_VICTIM_COST: float = 4.0       # energy taken from victim

OBSERVE_ENERGY_COST: float = 2.0
OBSERVE_HEAT_COST: float = 1.0

# Social stress level above which a diary entry is created.
SOCIAL_STRESS_THRESHOLD: float = 0.25

# Starting personality masks per agent index (cycles for N > 5)
_STARTING_MASKS: list[str] = ["Guardian", "Dreamer", "Courier", "Healer", "Judge"]

# Precomputed set of relative offsets for the 5×5 SIGNAL vision radius (radius=2)
_SIGNAL_VISION_OFFSETS: frozenset[tuple[int, int]] = frozenset(
    (dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)
)


# ─────────────────────────────────────────────────────────────────────────────
# Reputation System
# ─────────────────────────────────────────────────────────────────────────────

class ReputationSystem:
    """Simple per-agent trust tracker [0, 1] updated by social actions.

    All agents start at a neutral score of 0.5.  Cooperative behaviour
    raises scores; betrayals lower them.  Scores are precision-weighted —
    a highly-reputed agent's COOPERATE signal is worth more.

    Parameters
    ----------
    n_agents:
        Number of agents to track.
    """

    def __init__(self, n_agents: int) -> None:
        self._scores: dict[int, float] = {i: 0.5 for i in range(n_agents)}

    def cooperated(self, agent_id: int) -> None:
        """Record a cooperative action, raising the agent's trust score."""
        self._scores[agent_id] = min(1.0, self._scores[agent_id] + 0.10)

    def betrayed(self, agent_id: int) -> None:
        """Record a betrayal, lowering the agent's trust score."""
        self._scores[agent_id] = max(0.0, self._scores[agent_id] - 0.20)

    def decay(self, amount: float = 0.01) -> None:
        """Apply per-tick reputation decay so relationships evolve over time.

        Parameters
        ----------
        amount:
            Fraction to subtract from each score per tick (default −0.01/tick).
            Scores are floored at 0.0.
        """
        for aid in self._scores:
            self._scores[aid] = max(0.0, self._scores[aid] - amount)

    def score(self, agent_id: int) -> float:
        """Return the current trust score for *agent_id* (0–1)."""
        return self._scores.get(agent_id, 0.5)

    def all_scores(self) -> dict[int, float]:
        """Return a copy of all trust scores."""
        return dict(self._scores)


# ─────────────────────────────────────────────────────────────────────────────
# Alliance Tracker
# ─────────────────────────────────────────────────────────────────────────────

# Mutual reputation thresholds for alliance / war state formation
_ALLIANCE_REP_THRESHOLD: float = 0.7   # both agents must exceed this for an alliance
_WAR_REP_THRESHOLD: float = 0.3        # both agents must be below this for war state
_BETRAY_WAR_BONUS_ENERGY: float = 4.0  # extra energy stolen from victim during war


class AllianceTracker:
    """Tracks alliance and war states between pairs of agents.

    Alliance state (mutual rep > 0.7): agents split gathered resources fairly
    when on the same cell; COOPERATE actions are prioritised.

    War state (mutual rep < 0.3): BETRAY actions receive a bonus
    (extra energy stolen); COOPERATE is blocked between at-war agents.

    State is recomputed each tick after reputation decay is applied.

    Parameters
    ----------
    n_agents:
        Number of agents to track.
    """

    def __init__(self, n_agents: int) -> None:
        self._n = n_agents
        self._alliances: set[frozenset[int]] = set()
        self._wars: set[frozenset[int]] = set()

    def update(self, reputation: ReputationSystem) -> None:
        """Recompute alliance/war state from current reputation scores.

        Alliance requires both agents to have a score exceeding the threshold
        (i.e. both are individually trusted by the group).  War requires both
        to be below the war threshold.
        """
        self._alliances = set()
        self._wars = set()
        for i in range(self._n):
            for j in range(i + 1, self._n):
                pair: frozenset[int] = frozenset({i, j})
                si = reputation.score(i)
                sj = reputation.score(j)
                if si > _ALLIANCE_REP_THRESHOLD and sj > _ALLIANCE_REP_THRESHOLD:
                    self._alliances.add(pair)
                elif si < _WAR_REP_THRESHOLD and sj < _WAR_REP_THRESHOLD:
                    self._wars.add(pair)

    def are_allied(self, a: int, b: int) -> bool:
        """Return True if agents *a* and *b* are currently in an alliance."""
        return frozenset({a, b}) in self._alliances

    def are_at_war(self, a: int, b: int) -> bool:
        """Return True if agents *a* and *b* are currently in a war state."""
        return frozenset({a, b}) in self._wars

    def all_alliances(self) -> list[frozenset[int]]:
        """Return all current alliance pairs."""
        return list(self._alliances)

    def all_wars(self) -> list[frozenset[int]]:
        """Return all current war pairs."""
        return list(self._wars)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Outcome for a single agent after a multi-agent episode."""

    agent_id: int
    survived: bool
    ticks_alive: int
    total_reward: float
    resources_gathered: int
    contested_losses: int
    broadcasts_sent: int
    final_vitals: dict[str, float]
    # Social action tallies
    signals_sent: int = 0
    cooperations: int = 0
    betrayals: int = 0
    observations_made: int = 0
    final_reputation: float = 0.5   # trust score at end of episode
    starting_mask: str = "Guardian"  # personality mask this agent started with


@dataclass
class _AgentState:
    """Internal mutable state tracked per agent during a run."""

    agent_id: int
    mesh: GhostMesh
    world: GridWorld
    learner: QLearner
    obs: WorldObservation
    pos: tuple[int, int] = field(default=(1, 1))
    alive: bool = True
    ticks_alive: int = 0
    total_reward: float = 0.0
    resources_gathered: int = 0
    contested_losses: int = 0
    broadcasts_sent: int = 0
    inbox: list[str] = field(default_factory=list)
    # Social action counters
    signals_sent: int = 0
    cooperations: int = 0
    betrayals: int = 0
    observations_made: int = 0
    starting_mask: str = "Guardian"


# ─────────────────────────────────────────────────────────────────────────────
# MultiAgentRunner
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentRunner:
    """Simultaneous multi-agent GridWorld simulation.

    Each agent has its own ``GridWorld`` *view* of the same shared grid.
    A single canonical ``_SharedGrid`` object owns the cell state; individual
    ``GridWorld`` wrappers delegate reads/writes to it so that resource
    contention is correctly handled.

    The simplest implementation keeps one ``GridWorld`` (the shared arena)
    and moves agents through it sequentially within each tick, applying
    contention rules when two agents target the same resource cell.

    Parameters
    ----------
    n_agents:
        Number of simultaneous agents (default 3).
    seed:
        Master RNG seed.
    respawn:
        Whether resources respawn after being consumed.  Set ``False`` for
        the "Lifeboat Scenario".
    max_ticks:
        Default tick limit when calling ``run()``.
    world_width, world_height:
        Grid dimensions.
    """

    def __init__(
        self,
        n_agents: int = 3,
        seed: int | None = None,
        respawn: bool = True,
        max_ticks: int = 300,
        world_width: int = 30,
        world_height: int = 30,
        storm_prob: float = 0.02,
        predator_prob: float = 0.015,
        drought_prob: float = 0.01,
    ) -> None:
        self.n_agents = n_agents
        self.respawn = respawn
        self.max_ticks = max_ticks
        self._seed = seed
        self._tmp_dir = tempfile.mkdtemp(prefix="ghostmesh_multi_")

        # Shared arena — one GridWorld whose grid is the canonical truth.
        self._arena = GridWorld(
            width=world_width,
            height=world_height,
            seed=seed,
            allow_respawn=respawn,
        )

        # Global event system shared by all agents
        self._events = WorldEventSystem(
            storm_prob=storm_prob,
            predator_prob=predator_prob,
            drought_prob=drought_prob,
            seed=seed,
        )

        # Reputation system (shared social memory)
        self._reputation = ReputationSystem(n_agents=n_agents)
        # Alliance tracker — recomputed each tick after reputation decay
        self._alliance_tracker = AllianceTracker(n_agents=n_agents)

        self._agents: list[_AgentState] = []

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def run(self, max_ticks: int | None = None) -> list[AgentResult]:
        """Run the multi-agent episode and return per-agent results.

        Parameters
        ----------
        max_ticks:
            Override the default tick limit.

        Returns
        -------
        list[AgentResult]
            One entry per agent, in order.
        """
        limit = max_ticks or self.max_ticks
        self._setup()

        for tick in range(limit):
            if not any(a.alive for a in self._agents):
                break
            self._tick(tick)

        return [self._finalise(a) for a in self._agents]

    # ─────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Initialise the arena and spawn all agents."""
        self._arena.reset()
        self._reputation = ReputationSystem(n_agents=self.n_agents)
        self._alliance_tracker = AllianceTracker(n_agents=self.n_agents)
        self._agents = []

        for i in range(self.n_agents):
            seed_i = (self._seed or 0) + i
            state_file = os.path.join(self._tmp_dir, f"state_agent{i}.json")
            diary_path = os.path.join(self._tmp_dir, f"diary_agent{i}.db")

            os.environ["GHOST_STATE_FILE"] = state_file
            os.environ["GHOST_DIARY_PATH"] = diary_path
            os.environ["GHOST_HUD"] = "0"
            os.environ["GHOST_PULSE"] = "0"
            os.environ["GHOST_ENV_EVENTS"] = "0"

            mesh = GhostMesh(seed=seed_i)

            # Assign diverse starting personalities — prevents all agents from
            # converging to the same Guardian attractor from the first tick.
            # force= bypasses the min_ticks guard on the very first rotation;
            # at tick 0 the guard would block any change from the default mask.
            starting_mask = _STARTING_MASKS[i % len(_STARTING_MASKS)]
            mesh.rotator.maybe_rotate(
                current_tick=0, force=starting_mask
            )

            learner = QLearner(seed=seed_i)

            # Place each agent at a distinct empty cell
            pos = self._arena._random_empty_cell()
            self._arena._agent_pos = pos  # borrow single-agent API temporarily

            obs = self._arena.get_observation(
                other_agent_positions=self._other_positions(i, pos)
            )

            agent = _AgentState(
                agent_id=i,
                mesh=mesh,
                world=self._arena,  # all agents share the same arena object
                learner=learner,
                obs=obs,
                pos=pos,
                starting_mask=starting_mask,
            )
            self._agents.append(agent)

    # ─────────────────────────────────────────────────────────────────────
    # Tick loop
    # ─────────────────────────────────────────────────────────────────────

    def _tick(self, tick: int) -> None:
        """Advance the world by one tick for all living agents."""
        # Increment shared world tick once
        self._arena._world_tick += 1
        if self.respawn:
            self._arena._advance_respawn_timers()

        # Reputation decay and alliance update (every tick)
        self._reputation.decay(amount=0.01)
        self._alliance_tracker.update(self._reputation)

        # Fire global world event (storms, droughts, predators)
        world_event = self._events.tick(tick)

        # Apply storm hits to all living agents immediately
        if world_event.storm_hit:
            for agent in self._agents:
                if agent.alive:
                    agent.mesh.state.apply_action_feedback(
                        delta_heat=_STORM_HEAT_HIT,
                        delta_waste=_STORM_WASTE_HIT,
                    )

        # Apply novel-hazard penalty to Guardian-mode agents
        if world_event.novel_hazard_active:
            from thermodynamic_agency.world.grid_world import (
                _NOVEL_HAZARD_GUARDIAN_HEAT,
                _NOVEL_HAZARD_GUARDIAN_WASTE,
                _GUARDIAN_MODE_MASKS,
            )
            for agent in self._agents:
                if agent.alive and agent.mesh.rotator.active.name in _GUARDIAN_MODE_MASKS:
                    agent.mesh.state.apply_action_feedback(
                        delta_heat=_NOVEL_HAZARD_GUARDIAN_HEAT,
                        delta_waste=_NOVEL_HAZARD_GUARDIAN_WASTE,
                    )

        # Apply windfall bonus to plastic-mode agents
        if world_event.windfall_active:
            from thermodynamic_agency.world.grid_world import (
                _WINDFALL_ENERGY_BONUS,
                _PLASTIC_MODE_MASKS,
            )
            for agent in self._agents:
                if agent.alive and agent.mesh.rotator.active.name in _PLASTIC_MODE_MASKS:
                    agent.mesh.state.apply_action_feedback(
                        delta_energy=_WINDFALL_ENERGY_BONUS,
                    )

        # Determine intended actions for all agents (simultaneous intention)
        intentions: list[tuple[int, str]] = []  # (agent_id, action_str)
        for agent in self._agents:
            if not agent.alive:
                continue
            self._arena._agent_pos = agent.pos
            available = [a.value for a in self._arena.available_actions()]
            # Inject social and broadcast actions so they are Q-learnable
            available += [
                WorldAction.BROADCAST.value,
                WorldAction.SIGNAL.value,
                WorldAction.COOPERATE.value,
                WorldAction.BETRAY.value,
                WorldAction.OBSERVE_OTHER.value,
            ]
            vitals = agent.mesh.state.to_dict()
            state_key = encode_state(vitals, agent.obs)
            action_str = agent.learner.select_action(state_key, available)
            intentions.append((agent.agent_id, action_str))

        # Track which resource cells are claimed this tick (contention)
        gather_claims: dict[tuple[int, int], int] = {}  # pos → first agent_id
        for agent_id, action_str in intentions:
            if action_str == WorldAction.GATHER.value:
                agent = self._agents[agent_id]
                pos = agent.pos
                cell = self._arena._cell_at(pos)
                if cell in _GATHERABLE:
                    if pos not in gather_claims:
                        gather_claims[pos] = agent_id
                    # If another agent already claimed it → contested

        # Execute actions
        for agent_id, action_str in intentions:
            agent = self._agents[agent_id]
            self._execute_agent_action(
                agent, action_str, gather_claims, tick, world_event
            )

    def _execute_agent_action(
        self,
        agent: _AgentState,
        action_str: str,
        gather_claims: dict[tuple[int, int], int],
        tick: int,
        world_event: WorldEvent | None = None,
    ) -> None:
        """Execute one action for one agent and update learning subsystems."""
        vitals_before = agent.mesh.state.to_dict()
        state_key = encode_state(vitals_before, agent.obs)

        metabolic_delta: dict[str, float] = {}
        gathered = False
        contested = False

        if action_str == WorldAction.BROADCAST.value:
            # Broadcast: charge sender; no world movement
            agent.mesh.state.apply_action_feedback(
                delta_energy=-BROADCAST_ENERGY_COST,
                delta_heat=BROADCAST_HEAT_COST,
            )
            agent.broadcasts_sent += 1
            # Deliver message to other agents' inboxes
            msg = f"agent{agent.agent_id}@tick{tick}"
            for other in self._agents:
                if other.agent_id != agent.agent_id and other.alive:
                    other.inbox.append(msg)
            vitals_after = agent.mesh.state.to_dict()

        elif action_str == WorldAction.SIGNAL.value:
            # Cheap broadcast to visible neighbours only
            agent.mesh.state.apply_action_feedback(
                delta_energy=-SIGNAL_ENERGY_COST,
                delta_heat=SIGNAL_HEAT_COST,
            )
            agent.signals_sent += 1
            msg = f"signal:agent{agent.agent_id}@tick{tick}"
            ax, ay = agent.pos
            signal_zone = {(ax + dx, ay + dy) for dx, dy in _SIGNAL_VISION_OFFSETS}
            for other in self._agents:
                if (
                    other.agent_id != agent.agent_id
                    and other.alive
                    and other.pos in signal_zone
                ):
                    other.inbox.append(msg)
            vitals_after = agent.mesh.state.to_dict()

        elif action_str == WorldAction.COOPERATE.value:
            # Share energy with all nearby living agents.
            # BLOCKED between agents in war state.
            nearby = self._nearby_agents(agent)
            # Filter out agents at war with this agent
            cooperate_targets = [
                o for o in nearby
                if not self._alliance_tracker.are_at_war(agent.agent_id, o.agent_id)
            ]
            if cooperate_targets:
                agent.mesh.state.apply_action_feedback(
                    delta_energy=-COOPERATE_SENDER_COST,
                )
                for other in cooperate_targets:
                    # Allied agents receive a larger share
                    bonus = 1.5 if self._alliance_tracker.are_allied(
                        agent.agent_id, other.agent_id
                    ) else 1.0
                    other.mesh.state.apply_action_feedback(
                        delta_energy=COOPERATE_RECEIVER_GAIN * bonus,
                    )
                self._reputation.cooperated(agent.agent_id)
                agent.cooperations += 1
            vitals_after = agent.mesh.state.to_dict()

        elif action_str == WorldAction.BETRAY.value:
            # Steal energy from the nearest visible agent.
            # War state: BETRAY gets a bonus.
            nearest = self._nearest_agent(agent)
            if nearest is not None:
                at_war = self._alliance_tracker.are_at_war(
                    agent.agent_id, nearest.agent_id
                )
                bonus_energy = _BETRAY_WAR_BONUS_ENERGY if at_war else 0.0
                nearest.mesh.state.apply_action_feedback(
                    delta_energy=-(BETRAY_VICTIM_COST + bonus_energy),
                )
                agent.mesh.state.apply_action_feedback(
                    delta_energy=BETRAY_ACTOR_GAIN + bonus_energy,
                )
                self._reputation.betrayed(agent.agent_id)
                agent.betrayals += 1
            vitals_after = agent.mesh.state.to_dict()

        elif action_str == WorldAction.OBSERVE_OTHER.value:
            # Watch a nearby agent — pays cognitive cost, boosts world-model
            agent.mesh.state.apply_action_feedback(
                delta_energy=-OBSERVE_ENERGY_COST,
                delta_heat=OBSERVE_HEAT_COST,
            )
            agent.observations_made += 1
            vitals_after = agent.mesh.state.to_dict()

        elif action_str in (
            WorldAction.NORTH.value,
            WorldAction.SOUTH.value,
            WorldAction.EAST.value,
            WorldAction.WEST.value,
        ):
            # Movement — compute new position without using arena step()
            from thermodynamic_agency.world.grid_world import _MOVEMENT_DELTA, _ENTRY_EFFECTS, _merge_delta
            dx, dy = _MOVEMENT_DELTA[action_str]
            cx, cy = agent.pos
            candidate = (cx + dx, cy + dy)
            if self._arena._is_valid_move(candidate):
                # Check no other agent occupies the target cell
                occupied = {a.pos for a in self._agents if a.alive and a.agent_id != agent.agent_id}
                if candidate not in occupied:
                    agent.pos = candidate
                    cell = self._arena._cell_at(agent.pos)
                    _merge_delta(metabolic_delta, _ENTRY_EFFECTS.get(cell, {}))

            if metabolic_delta:
                agent.mesh.state.apply_action_feedback(**metabolic_delta)
            vitals_after = agent.mesh.state.to_dict()

        elif action_str == WorldAction.GATHER.value:
            from thermodynamic_agency.world.grid_world import _GATHER_EFFECTS, _merge_delta
            pos = agent.pos

            if gather_claims.get(pos) == agent.agent_id:
                # We won the claim race — gather the resource
                cell = self._arena._cell_at(pos)
                if cell in _GATHERABLE:
                    base_effects = _GATHER_EFFECTS.get(cell, {})
                    scaled = self._arena._apply_season_to_gather(cell, base_effects)
                    _merge_delta(metabolic_delta, scaled)
                    x, y = pos
                    self._arena._grid[y][x] = CellType.EMPTY.value
                    from thermodynamic_agency.world.grid_world import _RespawnTimer
                    if self.respawn:
                        self._arena._respawn_timers.append(
                            _RespawnTimer(
                                cell_type=cell,
                                position=pos,
                                ticks_remaining=_RESPAWN_TICKS.get(cell, 20),
                            )
                        )
                    gathered = True
                    agent.resources_gathered += 1
            elif pos in gather_claims:
                # Another agent pre-claimed this position → contested
                contested = True
                agent.contested_losses += 1
            else:
                # No prior claim (single-agent or unclaimed cell)
                cell = self._arena._cell_at(pos)
                if cell in _GATHERABLE:
                    base_effects = _GATHER_EFFECTS.get(cell, {})
                    scaled = self._arena._apply_season_to_gather(cell, base_effects)
                    _merge_delta(metabolic_delta, scaled)
                    x, y = pos
                    self._arena._grid[y][x] = CellType.EMPTY.value
                    from thermodynamic_agency.world.grid_world import _RespawnTimer
                    if self.respawn:
                        self._arena._respawn_timers.append(
                            _RespawnTimer(
                                cell_type=cell,
                                position=pos,
                                ticks_remaining=_RESPAWN_TICKS.get(cell, 20),
                            )
                        )
                    gathered = True
                    agent.resources_gathered += 1

            if metabolic_delta:
                agent.mesh.state.apply_action_feedback(**metabolic_delta)
            vitals_after = agent.mesh.state.to_dict()

        else:
            # WAIT or unrecognised
            vitals_after = agent.mesh.state.to_dict()

        # Run metabolic pulse
        try:
            agent.mesh._pulse()
        except GhostDeathException:
            agent.alive = False
            vitals_after = agent.mesh.state.to_dict()
            reward_sig = compute_reward(
                vitals_before=vitals_before,
                vitals_after=vitals_after,
                gathered=False,
                alive=False,
            )
            agent.total_reward += reward_sig.total
            return

        agent.ticks_alive += 1

        # Reward
        reward_sig = compute_reward(
            vitals_before=vitals_before,
            vitals_after=vitals_after,
            gathered=gathered,
            alive=True,
            contested=contested,
        )
        reward = reward_sig.total
        agent.total_reward += reward

        # Update observation with social context and world event signals
        other_positions = self._other_positions(agent.agent_id, agent.pos)
        self._arena._agent_pos = agent.pos
        predator_threat = world_event.predator_threat if world_event else 0.0
        active_event = world_event.label if world_event else ""
        novel_hazard_active = world_event.novel_hazard_active if world_event else False
        windfall_active = world_event.windfall_active if world_event else False
        next_obs = self._arena.get_observation(
            other_agent_positions=other_positions,
            predator_threat=predator_threat,
            active_world_event=active_event,
            novel_hazard_active=novel_hazard_active,
            windfall_active=windfall_active,
        )
        agent.obs = next_obs

        # Log social stressor to diary
        if next_obs.social_stress > SOCIAL_STRESS_THRESHOLD:
            from thermodynamic_agency.memory.diary import DiaryEntry
            agent.mesh.diary.append(DiaryEntry(
                tick=agent.mesh.state.entropy,
                role="stressor",
                content=(
                    f"Social Stressor: {len(next_obs.nearby_agents)} agent(s) "
                    f"nearby (stress={next_obs.social_stress:.2f})"
                ),
                metadata={"source": "multi_agent_runner", "nearby": len(next_obs.nearby_agents)},
            ))

        # Log predator threat to diary when active
        if next_obs.predator_threat > 0.0:
            from thermodynamic_agency.memory.diary import DiaryEntry
            agent.mesh.diary.append(DiaryEntry(
                tick=agent.mesh.state.entropy,
                role="stressor",
                content=(
                    f"Predator Threat: threat={next_obs.predator_threat:.2f} "
                    f"event={active_event!r}"
                ),
                metadata={"source": "world_event_system", "event": active_event},
            ))

        # Q-learner update
        next_state_key = encode_state(vitals_after, next_obs)
        self._arena._agent_pos = agent.pos
        # Include social actions in next available set
        next_available = [a.value for a in self._arena.available_actions()] + [
            WorldAction.BROADCAST.value,
            WorldAction.SIGNAL.value,
            WorldAction.COOPERATE.value,
            WorldAction.BETRAY.value,
            WorldAction.OBSERVE_OTHER.value,
        ]
        agent.learner.update(
            state_key, action_str, reward, next_state_key,
            done=False, next_actions=next_available,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _other_positions(
        self,
        my_id: int,
        my_pos: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Return positions of all *other* living agents."""
        return [
            a.pos
            for a in self._agents
            if a.alive and a.agent_id != my_id
        ]

    def _nearby_agents(self, agent: _AgentState) -> list[_AgentState]:
        """Return living agents within the 5×5 vision window of *agent*."""
        ax, ay = agent.pos
        radius = self._arena.vision_radius
        result = []
        for other in self._agents:
            if not other.alive or other.agent_id == agent.agent_id:
                continue
            ox, oy = other.pos
            if abs(ox - ax) <= radius and abs(oy - ay) <= radius:
                result.append(other)
        return result

    def _nearest_agent(self, agent: _AgentState) -> _AgentState | None:
        """Return the nearest visible living agent, or None if none visible."""
        nearby = self._nearby_agents(agent)
        if not nearby:
            return None
        ax, ay = agent.pos
        return min(nearby, key=lambda o: abs(o.pos[0] - ax) + abs(o.pos[1] - ay))

    def _finalise(self, agent: _AgentState) -> AgentResult:
        """Convert internal agent state to a public AgentResult."""
        return AgentResult(
            agent_id=agent.agent_id,
            survived=agent.alive,
            ticks_alive=agent.ticks_alive,
            total_reward=agent.total_reward,
            resources_gathered=agent.resources_gathered,
            contested_losses=agent.contested_losses,
            broadcasts_sent=agent.broadcasts_sent,
            final_vitals=agent.mesh.state.to_dict(),
            signals_sent=agent.signals_sent,
            cooperations=agent.cooperations,
            betrayals=agent.betrayals,
            observations_made=agent.observations_made,
            final_reputation=self._reputation.score(agent.agent_id),
            starting_mask=agent.starting_mask,
        )
