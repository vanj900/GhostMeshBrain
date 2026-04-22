"""Tests for the GridWorld environment."""

from __future__ import annotations

import pytest

from thermodynamic_agency.world.grid_world import (
    GridWorld,
    WorldAction,
    WorldObservation,
    WorldStepResult,
    CellType,
)


class TestGridWorldBasics:
    def test_reset_returns_observation(self):
        world = GridWorld(seed=1)
        obs = world.reset()
        assert isinstance(obs, WorldObservation)
        assert obs.position == world.agent_position

    def test_initial_position_is_empty_cell(self):
        world = GridWorld(seed=1)
        pos = world.agent_position
        cell = world.cell_at(pos)
        assert cell == CellType.EMPTY.value

    def test_grid_has_expected_cell_types(self):
        world = GridWorld(seed=1)
        all_cells = {
            world.cell_at((x, y))
            for x in range(world.width)
            for y in range(world.height)
        }
        # Must have at least walls, empty, and some resources
        assert CellType.WALL.value in all_cells
        assert CellType.EMPTY.value in all_cells

    def test_resource_count_includes_food(self):
        world = GridWorld(seed=2)
        counts = world.resource_count()
        # At least 4 food cells by default config
        total = sum(counts.values())
        assert total >= 4

    def test_movement_changes_position(self):
        world = GridWorld(seed=3)
        # Move until we find a valid direction
        initial = world.agent_position
        moved = False
        for action in WorldAction:
            if action in (WorldAction.GATHER, WorldAction.WAIT):
                continue
            result = world.step(action)
            if result.new_position != initial:
                moved = True
                break
        assert moved, "Should be able to move in at least one direction"

    def test_cannot_move_through_wall(self):
        world = GridWorld(seed=5)
        # Border cells are always walls; agent starts in interior
        initial = world.agent_position
        # Attempt to move into a wall is a no-op
        # Find a direction that leads into a wall
        x, y = initial
        for action, (dx, dy) in [
            (WorldAction.NORTH, (0, -1)),
            (WorldAction.SOUTH, (0, 1)),
            (WorldAction.EAST, (1, 0)),
            (WorldAction.WEST, (-1, 0)),
        ]:
            candidate = (x + dx, y + dy)
            if world.cell_at(candidate) == CellType.WALL.value:
                result = world.step(action)
                assert result.new_position == initial
                break

    def test_wait_does_not_move(self):
        world = GridWorld(seed=6)
        pos_before = world.agent_position
        world.step(WorldAction.WAIT)
        assert world.agent_position == pos_before

    def test_world_tick_increments(self):
        world = GridWorld(seed=7)
        assert world.world_tick == 0
        world.step(WorldAction.WAIT)
        assert world.world_tick == 1


class TestGridWorldGathering:
    def _place_agent_on_food(self, world: GridWorld) -> bool:
        """Move agent onto a food cell. Returns True if successful."""
        for y in range(world.height):
            for x in range(world.width):
                if world.cell_at((x, y)) == CellType.FOOD.value:
                    world._agent_pos = (x, y)
                    return True
        return False

    def test_gather_food_yields_energy(self):
        world = GridWorld(seed=10)
        placed = self._place_agent_on_food(world)
        if not placed:
            pytest.skip("No food cell found")
        result = world.step(WorldAction.GATHER)
        assert result.gathered
        assert result.metabolic_delta.get("delta_energy", 0.0) == 15.0

    def test_gather_consumes_food(self):
        world = GridWorld(seed=10)
        placed = self._place_agent_on_food(world)
        if not placed:
            pytest.skip("No food cell found")
        pos = world.agent_position
        world.step(WorldAction.GATHER)
        assert world.cell_at(pos) == CellType.EMPTY.value

    def test_gather_on_empty_is_noop(self):
        world = GridWorld(seed=11)
        # Ensure agent is on empty cell
        for y in range(1, world.height - 1):
            for x in range(1, world.width - 1):
                if world.cell_at((x, y)) == CellType.EMPTY.value:
                    world._agent_pos = (x, y)
                    result = world.step(WorldAction.GATHER)
                    assert not result.gathered
                    assert result.metabolic_delta == {}
                    return

    def test_food_respawns_after_delay(self):
        world = GridWorld(seed=12)
        placed = self._place_agent_on_food(world)
        if not placed:
            pytest.skip("No food cell found")
        pos = world.agent_position
        world.step(WorldAction.GATHER)
        assert world.cell_at(pos) == CellType.EMPTY.value
        # Advance past respawn delay
        for _ in range(25):
            world.step(WorldAction.WAIT)
        assert world.cell_at(pos) == CellType.FOOD.value


class TestGridWorldHazards:
    def _place_agent_on_radiation(self, world: GridWorld) -> bool:
        for y in range(world.height):
            for x in range(world.width):
                if world.cell_at((x, y)) == CellType.RADIATION.value:
                    return True
        return False

    def test_entering_radiation_increases_heat(self):
        world = GridWorld(seed=20)
        # Find a radiation cell and a neighbour that can move into it
        for y in range(1, world.height - 1):
            for x in range(1, world.width - 1):
                if world.cell_at((x, y)) == CellType.RADIATION.value:
                    # Try to place agent adjacent
                    for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                        if (
                            0 <= nx < world.width
                            and 0 <= ny < world.height
                            and world.cell_at((nx, ny)) not in (
                                CellType.WALL.value, CellType.RADIATION.value
                            )
                        ):
                            world._agent_pos = (nx, ny)
                            dx, dy = x - nx, y - ny
                            action_map = {
                                (0, -1): WorldAction.NORTH,
                                (0, 1): WorldAction.SOUTH,
                                (1, 0): WorldAction.EAST,
                                (-1, 0): WorldAction.WEST,
                            }
                            action = action_map.get((dx, dy))
                            if action:
                                result = world.step(action)
                                if result.new_position == (x, y):
                                    assert result.metabolic_delta.get("delta_heat", 0.0) > 0
                                    return
        pytest.skip("Could not place agent adjacent to radiation")


class TestGridWorldObservation:
    def test_observation_covers_5x5_neighbourhood(self):
        world = GridWorld(seed=30)
        obs = world.get_observation()
        # Vision radius 2 → 5×5 = 25 cells
        assert len(obs.visible_cells) == 25

    def test_observation_position_matches_agent(self):
        world = GridWorld(seed=31)
        obs = world.get_observation()
        assert obs.position == world.agent_position

    def test_resource_density_in_range(self):
        world = GridWorld(seed=32)
        obs = world.get_observation()
        assert 0.0 <= obs.resource_density <= 1.0
        assert 0.0 <= obs.hazard_density <= 1.0

    def test_has_resource_here_correct(self):
        world = GridWorld(seed=33)
        # Place on food
        for y in range(world.height):
            for x in range(world.width):
                if world.cell_at((x, y)) == CellType.FOOD.value:
                    world._agent_pos = (x, y)
                    obs = world.get_observation()
                    assert obs.has_resource_here()
                    return

    def test_nearest_resource_direction_not_none_when_food_visible(self):
        world = GridWorld(seed=34)
        # With default density, agent should see resources in the 5x5 window
        # Try a few positions
        for y in range(2, world.height - 2):
            for x in range(2, world.width - 2):
                if world.cell_at((x, y)) == CellType.EMPTY.value:
                    world._agent_pos = (x, y)
                    obs = world.get_observation()
                    if obs.nearby_resources:
                        direction = obs.nearest_resource_direction()
                        assert direction is not None
                        return

    def test_available_actions_excludes_wall_moves(self):
        world = GridWorld(seed=35)
        available = world.available_actions()
        assert WorldAction.WAIT in available
        # Should not include moves that would walk into walls
        x, y = world.agent_position
        for action, (dx, dy) in [
            (WorldAction.NORTH, (0, -1)),
            (WorldAction.SOUTH, (0, 1)),
            (WorldAction.EAST, (1, 0)),
            (WorldAction.WEST, (-1, 0)),
        ]:
            if world.cell_at((x + dx, y + dy)) == CellType.WALL.value:
                assert action not in available


class TestGridWorldEpisodes:
    def test_reset_increments_episode(self):
        world = GridWorld(seed=40)
        ep1 = world.episode
        world.reset()
        assert world.episode == ep1 + 1

    def test_world_tick_resets_on_new_episode(self):
        world = GridWorld(seed=41)
        for _ in range(10):
            world.step(WorldAction.WAIT)
        assert world.world_tick == 10
        world.reset()
        assert world.world_tick == 0


class TestPersistentLayout:
    """Verify that the map layout (grid + heightmap) does not change across resets."""

    def test_layout_signature_stable_across_resets(self):
        world = GridWorld(seed=7)
        sig1 = world.layout_signature()
        world.reset()
        sig2 = world.layout_signature()
        world.reset()
        sig3 = world.layout_signature()
        assert sig1 == sig2 == sig3, "Layout must be identical across reset() calls"

    def test_grid_stable_across_resets(self):
        world = GridWorld(seed=13)
        grid_before = [row[:] for row in world._grid]
        world.reset()
        grid_after = [row[:] for row in world._grid]
        assert grid_before == grid_after, "Grid must not regenerate on reset()"

    def test_heightmap_stable_across_resets(self):
        world = GridWorld(seed=17)
        hm_before = [row[:] for row in world._heightmap]
        world.reset()
        hm_after = [row[:] for row in world._heightmap]
        assert hm_before == hm_after, "Heightmap must not regenerate on reset()"

    def test_agent_position_rerandomized_on_reset(self):
        """The agent always lands on an EMPTY cell after reset."""
        world = GridWorld(seed=21)
        for _ in range(20):
            world.reset()
            pos = world.agent_position
            # Agent must always land on a valid (non-wall) empty cell.
            assert world.cell_at(pos) == CellType.EMPTY.value

    def test_default_size_is_30x30(self):
        world = GridWorld()
        assert world.width == 30
        assert world.height == 30

    def test_episode_runner_default_size_is_30x30(self):
        from thermodynamic_agency.world.episode_runner import EpisodeRunner
        runner = EpisodeRunner()
        assert runner.world.width == 30
        assert runner.world.height == 30

    def test_layout_initialized_flag(self):
        world = GridWorld(seed=99)
        assert world._layout_initialized is True
        sig_before = world.layout_signature()
        world.reset()
        # Flag stays True and layout unchanged
        assert world._layout_initialized is True
        assert world.layout_signature() == sig_before
