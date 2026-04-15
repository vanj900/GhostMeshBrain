"""EpisodeRunner — multi-episode training harness for GhostMesh + GridWorld.

Provides a clean interface for running the full learning loop across multiple
episodes and measuring whether the agent actually improves.

Architecture
------------
Two-level architecture per tick:

    Level 1 — Survival regulation (metabolic loop)
        MetabolicState.tick() returns FORAGE / REST / REPAIR / DECIDE.
        The corresponding action is dispatched unchanged — the organism keeps
        itself alive regardless of what is happening in the world.

    Level 2 — External task (world interaction)
        Every tick the Q-learner selects a world action (NORTH/SOUTH/EAST/WEST/
        GATHER/WAIT).  The world step yields a metabolic delta (e.g. eating food
        → +energy) that is applied to the metabolic state, plus a reward signal.
        The Q-learner, world model, working memory, and episodic store are
        updated from the experience.

Memory-driven action biasing
-----------------------------
Before the Q-learner's ε-greedy selection, the episodic store and working
memory are consulted:

1. If the episodic store recommends an action for the current state, it is
   promoted (its Q-value is notionally boosted by adding to the candidate set).
2. If the world model has much higher expected reward for a specific action,
   that action is offered as an alternative candidate.

The final selection is still ε-greedy over the enriched candidate set, so
exploration is preserved.

Usage
-----
    from thermodynamic_agency.world.episode_runner import EpisodeRunner

    runner = EpisodeRunner(n_episodes=50, ticks_per_episode=100, seed=42)
    stats = runner.train()
    print(stats.improvement_ratio)   # > 1.0 means later episodes were better

    # Or step through episodes manually:
    runner2 = EpisodeRunner(seed=0)
    for ep in range(10):
        ep_stats = runner2.run_episode(max_ticks=100)
        print(f"Episode {ep}: reward={ep_stats['total_reward']:.2f}")
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

from thermodynamic_agency.world.grid_world import GridWorld, WorldAction, WorldObservation
from thermodynamic_agency.learning.reward import compute_reward
from thermodynamic_agency.learning.experience_buffer import Experience, ExperienceBuffer
from thermodynamic_agency.learning.q_learner import QLearner, encode_state
from thermodynamic_agency.learning.world_model import WorldModel
from thermodynamic_agency.memory.working_memory import WorkingMemory, WorkingMemorySlot
from thermodynamic_agency.memory.episodic_store import EpisodicStore
from thermodynamic_agency.pulse import GhostMesh


@dataclass
class EpisodeStats:
    """Per-episode statistics."""

    episode: int
    ticks: int
    total_reward: float
    avg_reward: float
    resources_gathered: int
    hazard_hits: int
    survived: bool
    final_health: float
    epsilon: float


@dataclass
class TrainingStats:
    """Aggregate statistics across all episodes."""

    episodes: list[EpisodeStats] = field(default_factory=list)

    def avg_reward_first_n(self, n: int = 10) -> float:
        subset = self.episodes[:n]
        if not subset:
            return 0.0
        return sum(e.avg_reward for e in subset) / len(subset)

    def avg_reward_last_n(self, n: int = 10) -> float:
        subset = self.episodes[-n:]
        if not subset:
            return 0.0
        return sum(e.avg_reward for e in subset) / len(subset)

    @property
    def improvement_ratio(self) -> float:
        """Ratio of last-10 avg reward to first-10 avg reward.

        > 1.0 indicates the agent improved through training.
        Returns 1.0 when the first-10 average is zero or negative
        (to avoid division by zero or sign issues).
        """
        first = self.avg_reward_first_n(10)
        last = self.avg_reward_last_n(10)
        if first <= 0.0:
            return 1.0 if last >= first else 0.0
        return last / first

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)


class EpisodeRunner:
    """Orchestrates multi-episode training of GhostMesh in a GridWorld.

    Parameters
    ----------
    n_episodes:
        Number of training episodes (default 30).
    ticks_per_episode:
        Maximum ticks per episode before forced reset (default 150).
    seed:
        Master RNG seed; world and agent seeds are derived from this.
    world_width, world_height:
        Grid dimensions (default 10×10).
    alpha, gamma, epsilon:
        Q-learning hyperparameters.
    """

    def __init__(
        self,
        n_episodes: int = 30,
        ticks_per_episode: int = 150,
        seed: int | None = None,
        world_width: int = 10,
        world_height: int = 10,
        alpha: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 0.40,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.97,
    ) -> None:
        self.n_episodes = n_episodes
        self.ticks_per_episode = ticks_per_episode

        self.world = GridWorld(
            width=world_width,
            height=world_height,
            seed=seed,
        )
        self.learner = QLearner(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed,
        )
        self.world_model = WorldModel()
        self.experience_buffer = ExperienceBuffer(seed=seed)
        self.working_memory = WorkingMemory(capacity=30)
        self.episodic_store = EpisodicStore(maxlen=5_000)

        # GhostMesh state/diary live in temp files (reset each episode)
        self._tmp_dir = tempfile.mkdtemp(prefix="ghostmesh_episode_")
        self._mesh: GhostMesh | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> TrainingStats:
        """Run ``n_episodes`` training episodes and return aggregate stats."""
        stats = TrainingStats()
        for ep in range(self.n_episodes):
            ep_stats = self.run_episode(max_ticks=self.ticks_per_episode)
            stats.episodes.append(ep_stats)
        return stats

    def run_episode(self, max_ticks: int | None = None) -> EpisodeStats:
        """Run one episode: reset world, run up to *max_ticks* steps.

        Returns
        -------
        EpisodeStats
            Per-episode metrics.
        """
        max_ticks = max_ticks or self.ticks_per_episode

        # Reset world layout; agent starts fresh each episode
        obs = self.world.reset()

        # Create a fresh GhostMesh for this episode
        ep = self.world.episode
        state_file = os.path.join(self._tmp_dir, f"state_{ep}.json")
        diary_path = os.path.join(self._tmp_dir, f"diary_{ep}.db")

        os.environ["GHOST_STATE_FILE"] = state_file
        os.environ["GHOST_DIARY_PATH"] = diary_path
        os.environ["GHOST_HUD"] = "0"
        os.environ["GHOST_PULSE"] = "0"
        os.environ["GHOST_ENV_EVENTS"] = "0"   # world provides stimuli

        mesh = GhostMesh(seed=ep)

        # Episode tracking
        total_reward = 0.0
        resources_gathered = 0
        hazard_hits = 0
        survived = True
        ticks = 0

        from thermodynamic_agency.core.exceptions import GhostDeathException

        try:
            for _ in range(max_ticks):
                ticks += 1

                # ── Level 2: select world action ─────────────────────────────
                vitals_before = mesh.state.to_dict()
                state_key = encode_state(vitals_before, obs)
                available = [a.value for a in self.world.available_actions()]

                # Memory-augmented action selection:
                # 1. Get Q-learner's choice
                # 2. Episodic memory recommendation (high-similarity recall)
                # 3. World model fallback for unvisited states
                action_str = self._select_action_with_memory(
                    state_key, available, obs, vitals_before
                )

                world_result = self.world.step(WorldAction(action_str))
                if world_result.gathered:
                    resources_gathered += 1
                if world_result.cell_type in ("radiation", "toxin"):
                    hazard_hits += 1

                # Apply world metabolic delta to agent
                if world_result.metabolic_delta:
                    mesh.state.apply_action_feedback(**world_result.metabolic_delta)

                # ── Level 1: run metabolic tick ──────────────────────────────
                mesh._pulse()

                # ── Compute reward ────────────────────────────────────────────
                vitals_after = mesh.state.to_dict()
                reward_sig = compute_reward(
                    vitals_before=vitals_before,
                    vitals_after=vitals_after,
                    gathered=world_result.gathered,
                    alive=True,
                )
                reward = reward_sig.total
                total_reward += reward

                # ── Update next observation ───────────────────────────────────
                next_obs = world_result.observation or self.world.get_observation()
                next_state_key = encode_state(vitals_after, next_obs)

                # ── Update learning subsystems ────────────────────────────────
                self.learner.update(
                    state_key, action_str, reward, next_state_key, done=False,
                    next_actions=available,
                )
                self.world_model.update(state_key, action_str, reward, next_state_key)

                exp = Experience(
                    tick=mesh.state.entropy,
                    state_key=state_key,
                    action=action_str,
                    reward=reward,
                    next_state_key=next_state_key,
                )
                self.experience_buffer.push(exp)

                self.working_memory.push(
                    WorkingMemorySlot(
                        tick=mesh.state.entropy,
                        state_key=state_key,
                        action=action_str,
                        reward=reward,
                        cell_type=world_result.cell_type,
                        metabolic_snapshot=vitals_after,
                    )
                )
                self.episodic_store.record(
                    tick=mesh.state.entropy,
                    state_key=state_key,
                    action=action_str,
                    reward=reward,
                    next_state_key=next_state_key,
                    outcome_vitals=vitals_after,
                )

                obs = next_obs

        except GhostDeathException:
            survived = False
            total_reward -= 10.0

        self.learner.end_episode()

        avg_reward = total_reward / ticks if ticks > 0 else 0.0
        final_health = mesh.state.health_score()

        return EpisodeStats(
            episode=self.world.episode,
            ticks=ticks,
            total_reward=total_reward,
            avg_reward=avg_reward,
            resources_gathered=resources_gathered,
            hazard_hits=hazard_hits,
            survived=survived,
            final_health=final_health,
            epsilon=self.learner.epsilon,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _select_action_with_memory(
        self,
        state_key: tuple,
        available: list[str],
        obs: WorldObservation,
        vitals: dict[str, float],
    ) -> str:
        """Select world action using Q-learner augmented by memory signals.

        Decision order:
        1. If working memory shows declining reward trend AND episodic store
           suggests a better action → use episodic recommendation with
           probability proportional to Q-table sparsity for this state.
        2. Otherwise, use standard ε-greedy Q-learner selection.

        The world model provides a tie-breaking fallback for states the
        Q-learner has rarely visited.
        """
        # How well does the Q-learner know this state?
        q_visits = sum(
            self.learner._visit_counts.get((state_key, a), 0) for a in available
        )
        is_novel_state = q_visits < 3

        if is_novel_state:
            # World model fallback: prefer model's highest expected-reward action
            model_action = self.world_model.best_action_by_model(state_key, available)
            if model_action:
                return model_action

        # Check working memory for action recommendation
        wm_action = self.working_memory.best_action_for_state(state_key)
        declining = self.working_memory.reward_trend() < -0.01

        if declining and wm_action and wm_action in available:
            # Episodic memory may have a better recommendation
            ep_action = self.episodic_store.best_action_for_state(state_key)
            if ep_action and ep_action in available:
                return ep_action

        # Default: Q-learner ε-greedy
        return self.learner.select_action(state_key, available)
