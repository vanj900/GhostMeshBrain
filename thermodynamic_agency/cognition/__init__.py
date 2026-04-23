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
from thermodynamic_agency.cognition.predictive_hierarchy import (
    PredictiveHierarchy,
    HierarchySignal,
    LayerBelief,
)
from thermodynamic_agency.cognition.thalamus import (
    ThalamusGate,
    GateReport,
)
from thermodynamic_agency.cognition.basal_ganglia import (
    BasalGanglia,
    HabitRecord,
    HabitSignal,
)
from thermodynamic_agency.cognition.goal_engine import GoalEngine, Goal
from thermodynamic_agency.cognition.self_mod_engine import (
    SelfModEngine,
    SelfModProposal,
    SelfModVerdict,
    SelfModResult,
    SelfModTarget,
    PHASE4_MIN_TICKS,
    PHASE4_MIN_HEALTH,
    BELIEF_PRECISION_MIN,
    DO_NO_HARM_FLOOR,
    VALUE_WEIGHT_MIN,
    VALUE_WEIGHT_MAX,
    WATCHDOG_THRESHOLD,
    WATCHDOG_WINDOW,
    CHILL_TICKS,
)
from thermodynamic_agency.cognition.counterfactual import (
    CounterfactualEngine,
    CounterfactualTrace,
    CF_RISK_WEIGHT,
)
from thermodynamic_agency.cognition.language_cognition import (
    LanguageCognition,
    LanguageCognitionReport,
)
from thermodynamic_agency.cognition.homeostasis import (
    HomeostasisAdapter,
    HomeostasisStatus,
)
from thermodynamic_agency.cognition.soul_tension import (
    SoulTension,
    SoulScar,
    SoulTensionReport,
    WAR_CRY_TENSION_THRESHOLD,
    WAR_CRY_AFFECT_THRESHOLD,
    SCAR_HEAT_THRESHOLD,
    SCAR_ENERGY_THRESHOLD,
    SCAR_INTEGRITY_THRESHOLD,
    SCAR_STABILITY_THRESHOLD,
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
    "GoalEngine",
    "Goal",
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
    "PredictiveHierarchy",
    "HierarchySignal",
    "LayerBelief",
    "ThalamusGate",
    "GateReport",
    "BasalGanglia",
    "HabitRecord",
    "HabitSignal",
    "SelfModEngine",
    "SelfModProposal",
    "SelfModVerdict",
    "SelfModResult",
    "SelfModTarget",
    "PHASE4_MIN_TICKS",
    "PHASE4_MIN_HEALTH",
    "BELIEF_PRECISION_MIN",
    "DO_NO_HARM_FLOOR",
    "VALUE_WEIGHT_MIN",
    "VALUE_WEIGHT_MAX",
    "WATCHDOG_THRESHOLD",
    "WATCHDOG_WINDOW",
    "CHILL_TICKS",
    # Feature 1: CounterfactualEngine
    "CounterfactualEngine",
    "CounterfactualTrace",
    "CF_RISK_WEIGHT",
    # Feature 2: LanguageCognition
    "LanguageCognition",
    "LanguageCognitionReport",
    # Feature 3: HomeostasisAdapter
    "HomeostasisAdapter",
    "HomeostasisStatus",
    # Feature 4: SoulTension — patterned tension of coherence inside chaos
    "SoulTension",
    "SoulScar",
    "SoulTensionReport",
    "WAR_CRY_TENSION_THRESHOLD",
    "WAR_CRY_AFFECT_THRESHOLD",
    "SCAR_HEAT_THRESHOLD",
    "SCAR_ENERGY_THRESHOLD",
    "SCAR_INTEGRITY_THRESHOLD",
    "SCAR_STABILITY_THRESHOLD",
]
