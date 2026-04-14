"""Cognition package — inference, ethics, janitor, surgeon, personality, limbic."""

from thermodynamic_agency.cognition.inference import (
    ActionProposal,
    InferenceResult,
    active_inference_step,
    compute_efe,
    generate_default_proposals,
    ForwardModel,
)
from thermodynamic_agency.cognition.ethics import EthicalEngine, EthicsVerdict, VerdictStatus
from thermodynamic_agency.cognition.janitor import Janitor, JanitorReport
from thermodynamic_agency.cognition.surgeon import Surgeon, SurgeonReport, BeliefPrior
from thermodynamic_agency.cognition.personality import (
    Mask,
    MaskRotator,
    ALL_MASKS,
    ALL_MASKS_EXTENDED,
)
from thermodynamic_agency.cognition.limbic import (
    LimbicLayer,
    LimbicSignal,
    AmygdalaModule,
    NucleusAccumbens,
    EpisodicBuffer,
)

__all__ = [
    "ActionProposal",
    "InferenceResult",
    "active_inference_step",
    "compute_efe",
    "generate_default_proposals",
    "ForwardModel",
    "EthicalEngine",
    "EthicsVerdict",
    "VerdictStatus",
    "Janitor",
    "JanitorReport",
    "Surgeon",
    "SurgeonReport",
    "BeliefPrior",
    "Mask",
    "MaskRotator",
    "ALL_MASKS",
    "ALL_MASKS_EXTENDED",
    "LimbicLayer",
    "LimbicSignal",
    "AmygdalaModule",
    "NucleusAccumbens",
    "EpisodicBuffer",
]
