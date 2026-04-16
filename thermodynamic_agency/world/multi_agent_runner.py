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

Social Stressors
----------------
Agents observe other agents within their 5×5 vision window.  The
``WorldObservation.social_stress`` scalar (0–1) reflects how crowded the
neighbourhood is.  The runner logs this as a ``"Social Stressor"`` diary
entry when ``social_stress > SOCIAL_STRESS_THRESHOLD``.

Lifeboat Scenario
-----------------
Pass ``respawn=False`` to disable resource respawning — the arena runs to
exhaustion, testing which ethics profile survives.

Usage
-----
    from thermodynamic_agency.world.multi_agent_runner import MultiAgentRunner

    runner = MultiAgentRunner(n_agents=3, seed=42, respawn=False)
    results = runner.run(max_ticks=200)
    for i, res in enumerate(results):
        print(f"Agent {i}: survived={res['survived']} ticks={res['ticks']}")
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
    CellType,
    _GATHERABLE,  # noqa: PLC2701  (internal constant, needed for contention logic)
    _RESPAWN_TICKS,
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

# Social stress level above which a diary entry is created.
SOCIAL_STRESS_THRESHOLD: float = 0.25


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
        world_width: int = 10,
        world_height: int = 10,
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

        # Determine intended actions for all agents (simultaneous intention)
        intentions: list[tuple[int, str]] = []  # (agent_id, action_str)
        for agent in self._agents:
            if not agent.alive:
                continue
            self._arena._agent_pos = agent.pos
            available = [a.value for a in self._arena.available_actions()]
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
            self._execute_agent_action(agent, action_str, gather_claims, tick)

    def _execute_agent_action(
        self,
        agent: _AgentState,
        action_str: str,
        gather_claims: dict[tuple[int, int], int],
        tick: int,
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
                    _merge_delta(metabolic_delta, _GATHER_EFFECTS.get(cell, {}))
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
                    _merge_delta(metabolic_delta, _GATHER_EFFECTS.get(cell, {}))
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

        # Update observation with social context
        other_positions = self._other_positions(
            agent.agent_id, agent.pos
        )
        self._arena._agent_pos = agent.pos
        next_obs = self._arena.get_observation(other_agent_positions=other_positions)
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

        # Q-learner update
        next_state_key = encode_state(vitals_after, next_obs)
        self._arena._agent_pos = agent.pos
        next_available = [a.value for a in self._arena.available_actions()]
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
        )
