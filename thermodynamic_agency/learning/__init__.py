"""Learning module — reward, experience, Q-learning, and world model."""

from thermodynamic_agency.learning.reward import compute_reward, RewardSignal
from thermodynamic_agency.learning.experience_buffer import Experience, ExperienceBuffer
from thermodynamic_agency.learning.q_learner import QLearner, encode_state
from thermodynamic_agency.learning.world_model import WorldModel

__all__ = [
    "compute_reward",
    "RewardSignal",
    "Experience",
    "ExperienceBuffer",
    "QLearner",
    "encode_state",
    "WorldModel",
]
