"""evaluation — cognitive battery and g-factor measurement for GhostMesh.

Submodules
----------
cognitive_battery
    Six-task evaluation battery that measures specialised and general
    cognitive performance for one agent episode.

g_factor
    PCA-based g-factor computation across a batch of score vectors.
"""

from thermodynamic_agency.evaluation.cognitive_battery import (
    CognitiveBattery,
    TaskScores,
    TASK_NAMES,
)
from thermodynamic_agency.evaluation.g_factor import (
    GFactorResult,
    measure_g,
)

__all__ = [
    "CognitiveBattery",
    "TaskScores",
    "TASK_NAMES",
    "GFactorResult",
    "measure_g",
]
