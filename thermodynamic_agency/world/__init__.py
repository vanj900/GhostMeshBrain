"""World module — external environment for embodied GhostMesh."""

from thermodynamic_agency.world.grid_world import (
    GridWorld,
    WorldAction,
    WorldObservation,
    WorldStepResult,
    CellType,
)

__all__ = [
    "GridWorld",
    "WorldAction",
    "WorldObservation",
    "WorldStepResult",
    "CellType",
]
