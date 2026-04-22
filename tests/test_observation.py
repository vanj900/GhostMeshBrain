"""Tests for the 2.5D observation module and GridWorld raycasting."""

from __future__ import annotations

import pytest

from thermodynamic_agency.world.observation import (
    HitType,
    RayHit,
    RawObservation,
    ObservationVector,
    encode_id,
    encode_onehot,
    expected_vector_length,
    ray_directions,
    _N_HIT_TYPES,
)
from thermodynamic_agency.world.grid_world import GridWorld, WorldAction


# ─────────────────────────────────────────────────────────────────────────────
# observation.py unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHitType:
    def test_all_values_unique(self):
        values = [ht.value for ht in HitType]
        assert len(values) == len(set(values))

    def test_required_types_present(self):
        names = {ht.name for ht in HitType}
        for required in ("NONE", "WALL", "RESOURCE", "HAZARD", "AGENT", "SHELTER"):
            assert required in names, f"HitType.{required} is missing"

    def test_n_hit_types_matches_enum(self):
        assert _N_HIT_TYPES == len(HitType)


class TestRayHit:
    def test_basic_construction(self):
        rh = RayHit(hit_type=HitType.WALL, dist=3, dh=-1)
        assert rh.hit_type is HitType.WALL
        assert rh.dist == 3
        assert rh.dh == -1


class TestEncoders:
    """Tests for encode_id and encode_onehot."""

    def _make_raw(self, ray_count: int = 4) -> RawObservation:
        rays = [
            RayHit(HitType.NONE,     dist=6, dh=0),
            RayHit(HitType.WALL,     dist=2, dh=3),
            RayHit(HitType.RESOURCE, dist=4, dh=-1),
            RayHit(HitType.HAZARD,   dist=1, dh=2),
        ][:ray_count]
        proprio = [0.5, 0.2, 0.1]
        return RawObservation(
            rays=rays,
            proprio=proprio,
            meta={"ray_count": ray_count, "ray_range": 6},
        )

    # ── encode_id ────────────────────────────────────────────────────────────

    def test_id_vector_length(self):
        ray_count = 4
        raw = self._make_raw(ray_count)
        vec = encode_id(raw)
        assert len(vec.vector) == expected_vector_length("id", ray_count, len(raw.proprio))

    def test_id_hit_type_values_are_ints(self):
        raw = self._make_raw()
        vec = encode_id(raw)
        # Every 3rd value starting at index 0 is a hit-type ID
        for i in range(0, len(raw.rays) * 3, 3):
            assert vec.vector[i] == int(vec.vector[i])

    def test_id_encoding_correct_values(self):
        raw = self._make_raw(1)
        raw.rays = [RayHit(HitType.RESOURCE, dist=3, dh=-2)]
        vec = encode_id(raw)
        assert vec.vector[0] == float(HitType.RESOURCE.value)
        assert vec.vector[1] == 3.0
        assert vec.vector[2] == -2.0

    def test_id_proprio_appended(self):
        raw = self._make_raw(2)
        vec = encode_id(raw)
        tail = vec.vector[2 * 3:]
        assert tail == [float(p) for p in raw.proprio]

    def test_id_meta_encoding_field(self):
        vec = encode_id(self._make_raw())
        assert vec.meta["encoding"] == "id"

    # ── encode_onehot ────────────────────────────────────────────────────────

    def test_onehot_vector_length(self):
        ray_count = 4
        raw = self._make_raw(ray_count)
        vec = encode_onehot(raw)
        expected = expected_vector_length("onehot", ray_count, len(raw.proprio))
        assert len(vec.vector) == expected

    def test_onehot_exactly_one_hot(self):
        raw = self._make_raw()
        vec = encode_onehot(raw)
        stride = _N_HIT_TYPES + 2  # onehot bits + dist + dh
        for i in range(len(raw.rays)):
            onehot_slice = vec.vector[i * stride : i * stride + _N_HIT_TYPES]
            assert sum(onehot_slice) == 1.0, f"Ray {i}: one-hot sum is not 1"
            assert all(v in (0.0, 1.0) for v in onehot_slice)

    def test_onehot_correct_bit_position(self):
        raw = self._make_raw(1)
        raw.rays = [RayHit(HitType.AGENT, dist=2, dh=1)]
        vec = encode_onehot(raw)
        hot_idx = HitType.AGENT.value
        assert vec.vector[hot_idx] == 1.0
        # All other bits must be 0
        for j in range(_N_HIT_TYPES):
            if j != hot_idx:
                assert vec.vector[j] == 0.0

    def test_onehot_proprio_appended(self):
        raw = self._make_raw(2)
        vec = encode_onehot(raw)
        stride = _N_HIT_TYPES + 2
        tail = vec.vector[2 * stride:]
        assert tail == [float(p) for p in raw.proprio]

    def test_onehot_meta_encoding_field(self):
        vec = encode_onehot(self._make_raw())
        assert vec.meta["encoding"] == "onehot"

    # ── consistency ──────────────────────────────────────────────────────────

    def test_both_encodings_include_same_proprio(self):
        raw = self._make_raw()
        id_vec = encode_id(raw)
        oh_vec = encode_onehot(raw)
        assert id_vec.vector[-3:] == oh_vec.vector[-3:]

    def test_expected_vector_length_id(self):
        assert expected_vector_length("id", 8, 3) == 8 * 3 + 3

    def test_expected_vector_length_onehot(self):
        assert expected_vector_length("onehot", 8, 3) == 8 * (_N_HIT_TYPES + 2) + 3

    def test_expected_vector_length_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown encoding"):
            expected_vector_length("unknown", 8, 3)


class TestObservationVector:
    def _make_vec(self) -> ObservationVector:
        return ObservationVector(vector=[1.0, 2.0, 3.0], meta={"encoding": "id"})

    def test_as_list_returns_list(self):
        vec = self._make_vec()
        result = vec.as_list()
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_as_list_is_copy(self):
        vec = self._make_vec()
        lst = vec.as_list()
        lst.append(99.0)
        assert len(vec.vector) == 3  # original not mutated

    def test_as_numpy_dtype_float32(self):
        import numpy as np
        vec = self._make_vec()
        arr = vec.as_numpy()
        assert arr.dtype == np.float32
        assert arr.shape == (3,)

    def test_as_numpy_custom_dtype(self):
        import numpy as np
        vec = self._make_vec()
        arr = vec.as_numpy(dtype=np.float64)
        assert arr.dtype == np.float64

    def test_as_numpy_values_match(self):
        import numpy as np
        vec = self._make_vec()
        arr = vec.as_numpy()
        assert list(arr) == pytest.approx([1.0, 2.0, 3.0])


class TestRayDirections:
    def test_count_matches(self):
        dirs = ray_directions(16)
        assert len(dirs) == 16

    def test_unit_vectors(self):
        import math
        dirs = ray_directions(8)
        for dx, dy in dirs:
            mag = math.sqrt(dx ** 2 + dy ** 2)
            assert abs(mag - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# GridWorld integration: sense_raw + observe
# ─────────────────────────────────────────────────────────────────────────────

class TestGridWorldHeightmap:
    def test_heightmap_generated_on_reset(self):
        world = GridWorld(seed=1)
        assert len(world._heightmap) == world.height
        assert len(world._heightmap[0]) == world.width

    def test_heightmap_values_in_range(self):
        world = GridWorld(seed=7)
        for row in world._heightmap:
            for h in row:
                assert 0 <= h <= 7

    def test_wall_cells_have_max_height(self):
        world = GridWorld(seed=2)
        from thermodynamic_agency.world.grid_world import CellType
        for y in range(world.height):
            for x in range(world.width):
                if world._grid[y][x] == CellType.WALL.value:
                    assert world._heightmap[y][x] == 7

    def test_heightmap_reproducible_with_seed(self):
        w1 = GridWorld(seed=99)
        w2 = GridWorld(seed=99)
        assert w1._heightmap == w2._heightmap

    def test_heightmap_stable_across_episodes(self):
        world = GridWorld(seed=5)
        hm1 = [row[:] for row in world._heightmap]
        world.reset()
        hm2 = [row[:] for row in world._heightmap]
        # Layout is persistent — heightmap must be identical across resets.
        assert hm1 == hm2


class TestSenseRaw:
    def test_ray_count_matches(self):
        world = GridWorld(seed=1, ray_count=8, ray_range=4)
        raw = world.sense_raw()
        assert len(raw.rays) == 8

    def test_ray_count_16(self):
        world = GridWorld(seed=2, ray_count=16, ray_range=6)
        raw = world.sense_raw()
        assert len(raw.rays) == 16

    def test_all_rays_are_rayhit(self):
        from thermodynamic_agency.world.observation import RayHit
        world = GridWorld(seed=3, ray_count=8, ray_range=5)
        raw = world.sense_raw()
        for r in raw.rays:
            assert isinstance(r, RayHit)

    def test_hit_types_valid(self):
        world = GridWorld(seed=4, ray_count=16, ray_range=6)
        raw = world.sense_raw()
        valid = set(HitType)
        for r in raw.rays:
            assert r.hit_type in valid

    def test_dist_within_range(self):
        ray_range = 5
        world = GridWorld(seed=4, ray_count=16, ray_range=ray_range)
        raw = world.sense_raw()
        for r in raw.rays:
            assert 1 <= r.dist <= ray_range

    def test_proprio_length(self):
        world = GridWorld(seed=5, ray_count=8, ray_range=4)
        raw = world.sense_raw()
        assert len(raw.proprio) == 3  # height, temp, waste

    def test_proprio_in_unit_range(self):
        world = GridWorld(seed=6, ray_count=8, ray_range=4)
        raw = world.sense_raw()
        for p in raw.proprio:
            assert 0.0 <= p <= 1.0

    def test_meta_contains_keys(self):
        world = GridWorld(seed=7, ray_count=8, ray_range=4)
        raw = world.sense_raw()
        for key in ("ray_count", "ray_range", "n_hit_types", "n_proprio"):
            assert key in raw.meta

    def test_determinism_with_seed(self):
        """Fixed seed → first observation at reset is stable."""
        w1 = GridWorld(seed=42, ray_count=16, ray_range=6)
        w2 = GridWorld(seed=42, ray_count=16, ray_range=6)
        raw1 = w1.sense_raw()
        raw2 = w2.sense_raw()
        for r1, r2 in zip(raw1.rays, raw2.rays):
            assert r1.hit_type == r2.hit_type
            assert r1.dist == r2.dist
            assert r1.dh == r2.dh
        assert raw1.proprio == raw2.proprio


class TestObserve:
    def test_observe_id_length(self):
        ray_count = 8
        world = GridWorld(seed=1, ray_count=ray_count, ray_range=5, obs_encoding="id")
        vec = world.observe()
        expected = expected_vector_length("id", ray_count, 3)
        assert len(vec.as_list()) == expected

    def test_observe_onehot_length(self):
        ray_count = 8
        world = GridWorld(seed=1, ray_count=ray_count, ray_range=5, obs_encoding="onehot")
        vec = world.observe()
        expected = expected_vector_length("onehot", ray_count, 3)
        assert len(vec.as_list()) == expected

    def test_observe_returns_observation_vector(self):
        world = GridWorld(seed=2, ray_count=4, ray_range=3)
        vec = world.observe()
        assert isinstance(vec, ObservationVector)

    def test_observe_encoding_set_in_meta(self):
        for enc in ("id", "onehot"):
            world = GridWorld(seed=3, obs_encoding=enc)
            vec = world.observe()
            assert vec.meta["encoding"] == enc

    def test_observe_numpy_shape_and_dtype(self):
        import numpy as np
        ray_count = 4
        world = GridWorld(seed=4, ray_count=ray_count, ray_range=3, obs_encoding="id")
        arr = world.observe().as_numpy()
        expected_len = expected_vector_length("id", ray_count, 3)
        assert arr.shape == (expected_len,)
        assert arr.dtype == np.float32

    def test_observe_stable_over_steps(self):
        ray_count = 8
        world = GridWorld(seed=5, ray_count=ray_count, ray_range=5, obs_encoding="id")
        expected = expected_vector_length("id", ray_count, 3)
        for _ in range(5):
            world.step(WorldAction.WAIT)
            vec = world.observe()
            assert len(vec.as_list()) == expected

    def test_invalid_obs_encoding_raises(self):
        with pytest.raises(ValueError, match="obs_encoding"):
            GridWorld(obs_encoding="invalid")

    def test_16_rays_id_full_length(self):
        """16-ray ID encoding: vector = 16*3 + 3 = 51."""
        world = GridWorld(seed=10, ray_count=16, ray_range=6, obs_encoding="id")
        vec = world.observe()
        assert len(vec.as_list()) == 51

    def test_16_rays_onehot_full_length(self):
        """16-ray one-hot: vector = 16*(N+2) + 3 where N = len(HitType)."""
        world = GridWorld(seed=10, ray_count=16, ray_range=6, obs_encoding="onehot")
        vec = world.observe()
        n = len(HitType)
        assert len(vec.as_list()) == 16 * (n + 2) + 3


# ─────────────────────────────────────────────────────────────────────────────
# Constructor arg pass-through: runners
# ─────────────────────────────────────────────────────────────────────────────

class TestRunnerConstructorArgs:
    def test_multi_agent_runner_accepts_obs_args(self):
        from thermodynamic_agency.world.multi_agent_runner import MultiAgentRunner
        runner = MultiAgentRunner(
            n_agents=2, seed=1, max_ticks=2,
            obs_encoding="onehot", ray_count=4, ray_range=3,
        )
        assert runner._arena.obs_encoding == "onehot"
        assert runner._arena.ray_count == 4
        assert runner._arena.ray_range == 3

    def test_episode_runner_accepts_obs_args(self):
        from thermodynamic_agency.world.episode_runner import EpisodeRunner
        runner = EpisodeRunner(
            seed=1,
            obs_encoding="onehot",
            ray_count=4,
            ray_range=3,
        )
        assert runner.world.obs_encoding == "onehot"
        assert runner.world.ray_count == 4
        assert runner.world.ray_range == 3
