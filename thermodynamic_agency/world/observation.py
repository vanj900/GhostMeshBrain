"""Observation encoding for GhostMesh GridWorld.

Provides a canonical "raw sensor" representation of the agent's perception
(raycasts + proprioception) and two numeric encoders:

* ``encode_id``     → compact vector: one integer ID per ray hit type
* ``encode_onehot`` → one-hot vector: N_TYPES bits per ray hit type

Both encoders return an :class:`ObservationVector` that exposes the data as a
Python list (JSON / JSONL friendly) **and** a NumPy array (model friendly).

Example usage
-------------
::

    from thermodynamic_agency.world.observation import (
        HitType, RayHit, RawObservation, encode_id, encode_onehot
    )

    raw = RawObservation(
        rays=[RayHit(HitType.RESOURCE, dist=3, dh=-1)],
        proprio=[0.5, 0.2, 0.0],
        meta={"ray_count": 1, "ray_range": 6},
    )

    vec_id = encode_id(raw)
    vec_oh = encode_onehot(raw)

    print(vec_id.as_list())           # [1, 3, -1, ...]
    print(vec_oh.as_numpy().shape)    # (N_TYPES + 2 + len(proprio),)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# HitType — canonical ordered enum
# The ordering is frozen here; do not reorder without updating any stored data.
# ─────────────────────────────────────────────────────────────────────────────

class HitType(Enum):
    """What a ray hit (or failed to hit)."""
    NONE    = 0  # ray completed its range with no hit
    WALL    = 1
    RESOURCE = 2
    HAZARD  = 3
    AGENT   = 4
    SHELTER = 5


# Derived constants from HitType — keep in sync automatically.
_N_HIT_TYPES: int = len(HitType)
_HIT_TYPE_IDS: dict[HitType, int] = {ht: ht.value for ht in HitType}


# ─────────────────────────────────────────────────────────────────────────────
# Raw sensor dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RayHit:
    """The result of a single raycast.

    Attributes
    ----------
    hit_type:
        Category of the first object the ray encountered.
    dist:
        Manhattan cell distance to the hit (0 = hit on the agent's cell;
        ``ray_range`` = ray completed without hitting anything solid).
    dh:
        Height difference: ``height_at_hit - height_here`` (signed integer).
        Positive = hit is uphill; negative = hit is downhill.
    """
    hit_type: HitType
    dist: int
    dh: int


@dataclass
class RawObservation:
    """Symbolic sensor vector before numeric encoding.

    Attributes
    ----------
    rays:
        One :class:`RayHit` per cast ray, in the same angular order they
        were cast (deterministic).
    proprio:
        Proprioceptive scalars (e.g. height_here, local_temp, local_waste).
        Always ``float``.
    meta:
        Encoding metadata: ``ray_count``, ``ray_range``, ``n_hit_types``,
        and any extra diagnostics.
    """
    rays: list[RayHit]
    proprio: list[float]
    meta: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# ObservationVector — numeric output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObservationVector:
    """Numeric observation vector.

    Attributes
    ----------
    vector:
        Flat list of floats.  JSON-serialisable, JSONL-friendly.
    meta:
        Metadata carried from the raw observation plus encoding info:
        ``encoding`` (``"id"`` or ``"onehot"``), ``vector_length``, and
        everything in :attr:`RawObservation.meta`.
    """
    vector: list[float]
    meta: dict = field(default_factory=dict)

    def as_list(self) -> list[float]:
        """Return the observation as a plain Python list of floats."""
        return list(self.vector)

    def as_numpy(self, dtype=None):
        """Return the observation as a NumPy ``float32`` array.

        Parameters
        ----------
        dtype:
            NumPy dtype to cast to (default ``np.float32``).

        Raises
        ------
        ImportError
            If NumPy is not installed.
        """
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "NumPy is required for ObservationVector.as_numpy(). "
                "Install it with: pip install numpy"
            ) from exc
        if dtype is None:
            dtype = np.float32
        return np.asarray(self.vector, dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────

def encode_id(raw: RawObservation) -> ObservationVector:
    """Encode a :class:`RawObservation` using integer IDs for hit types.

    Layout per ray (3 values):
        ``[hit_type_id, dist, dh]``

    Followed by the proprioceptive scalars.

    Vector length: ``ray_count * 3 + len(proprio)``
    """
    values: list[float] = []
    for ray in raw.rays:
        values.append(float(_HIT_TYPE_IDS[ray.hit_type]))
        values.append(float(ray.dist))
        values.append(float(ray.dh))
    values.extend(float(p) for p in raw.proprio)

    meta = dict(raw.meta)
    meta.update({
        "encoding": "id",
        "vector_length": len(values),
        "n_hit_types": _N_HIT_TYPES,
    })
    return ObservationVector(vector=values, meta=meta)


def encode_onehot(raw: RawObservation) -> ObservationVector:
    """Encode a :class:`RawObservation` using one-hot hit type vectors.

    Layout per ray (``N_HIT_TYPES + 2`` values):
        ``[*onehot(hit_type), dist, dh]``

    Followed by the proprioceptive scalars.

    Vector length: ``ray_count * (N_HIT_TYPES + 2) + len(proprio)``
    """
    values: list[float] = []
    for ray in raw.rays:
        # One-hot over all HitType values
        onehot = [0.0] * _N_HIT_TYPES
        onehot[_HIT_TYPE_IDS[ray.hit_type]] = 1.0
        values.extend(onehot)
        values.append(float(ray.dist))
        values.append(float(ray.dh))
    values.extend(float(p) for p in raw.proprio)

    meta = dict(raw.meta)
    meta.update({
        "encoding": "onehot",
        "vector_length": len(values),
        "n_hit_types": _N_HIT_TYPES,
    })
    return ObservationVector(vector=values, meta=meta)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: expected vector lengths
# ─────────────────────────────────────────────────────────────────────────────

def expected_vector_length(
    encoding: str,
    ray_count: int,
    n_proprio: int,
) -> int:
    """Compute expected vector length without constructing an observation.

    Parameters
    ----------
    encoding:
        ``"id"`` or ``"onehot"``.
    ray_count:
        Number of rays cast per observation.
    n_proprio:
        Number of proprioceptive scalars.

    Returns
    -------
    int
        Total vector length.

    Raises
    ------
    ValueError
        For unknown encoding strings.
    """
    if encoding == "id":
        return ray_count * 3 + n_proprio
    if encoding == "onehot":
        return ray_count * (_N_HIT_TYPES + 2) + n_proprio
    raise ValueError(f"Unknown encoding: {encoding!r}. Choose 'id' or 'onehot'.")


# ─────────────────────────────────────────────────────────────────────────────
# Angle helpers (used by GridWorld raycasting)
# ─────────────────────────────────────────────────────────────────────────────

def ray_directions(ray_count: int) -> list[tuple[float, float]]:
    """Return ``ray_count`` evenly-spaced unit direction vectors.

    Angles start at 0 (east) and increase counter-clockwise.  Returns
    ``(dx, dy)`` as floats; callers step by rounding to the nearest
    integer cell offset.
    """
    directions = []
    for i in range(ray_count):
        angle = 2.0 * math.pi * i / ray_count
        directions.append((math.cos(angle), math.sin(angle)))
    return directions
