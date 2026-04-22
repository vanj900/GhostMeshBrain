"""Observation encoding demo — shows 2.5D raw sensor vectors.

Runs a very short episode (5 ticks) with both ``id`` and ``onehot`` encodings,
printing the vector length and a sample of values for each.

Usage::

    python examples/observe_encoding_demo.py

The script also shows how to obtain NumPy arrays for model consumption.
"""

from __future__ import annotations

from thermodynamic_agency.world.grid_world import GridWorld, WorldAction
from thermodynamic_agency.world.observation import expected_vector_length, HitType

SEED = 42
RAY_COUNT = 16
RAY_RANGE = 6
N_PROPRIO = 3  # height_here, local_ambient_temp, local_waste_field


def demo_encoding(encoding: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  Encoding: {encoding!r}")
    print(f"{'─' * 50}")

    world = GridWorld(
        seed=SEED,
        obs_encoding=encoding,
        ray_count=RAY_COUNT,
        ray_range=RAY_RANGE,
    )
    world.reset()

    expected_len = expected_vector_length(encoding, RAY_COUNT, N_PROPRIO)
    print(f"  Expected vector length : {expected_len}")

    obs_vec = world.observe()
    actual_list = obs_vec.as_list()
    print(f"  Actual vector length   : {len(actual_list)}")
    assert len(actual_list) == expected_len, (
        f"Length mismatch: got {len(actual_list)}, expected {expected_len}"
    )

    # First 10 values as a preview
    preview = [round(v, 3) for v in actual_list[:10]]
    print(f"  First 10 values        : {preview}")

    # NumPy array
    arr = obs_vec.as_numpy()
    print(f"  NumPy shape            : {arr.shape}  dtype={arr.dtype}")

    # Step a few ticks and show observations remain stable
    for i in range(5):
        world.step(WorldAction.WAIT)
        vec = world.observe()
        assert len(vec.as_list()) == expected_len, (
            f"Tick {i}: length changed to {len(vec.as_list())}"
        )
    print(f"  Observation length is stable over 5 ticks ✓")

    # Show hit-type distribution for first reset
    raw = world.sense_raw()
    type_counts: dict[str, int] = {}
    for ray in raw.rays:
        name = ray.hit_type.name
        type_counts[name] = type_counts.get(name, 0) + 1
    print(f"  Ray hit-type distribution: {type_counts}")

    # Proprioception
    print(f"  Proprio (h, temp, waste): {[round(p, 3) for p in raw.proprio]}")


def main() -> None:
    print("GhostMesh 2.5D Observation Encoding Demo")
    print(f"  seed={SEED}  ray_count={RAY_COUNT}  ray_range={RAY_RANGE}")
    print(f"  HitTypes: {[ht.name for ht in HitType]}")

    demo_encoding("id")
    demo_encoding("onehot")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
