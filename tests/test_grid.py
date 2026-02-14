"""Tests for Grid: placement, adjacency, bounds, rendering."""
import pytest

from markov.agent import Agent
from markov.config import load_game_config
from markov.family import Family
from markov.grid import CORNER_SPAWNS, Grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(name: str, tier: int = 1, pos: tuple[int, int] = (0, 0)) -> Agent:
    return Agent(
        id=name.lower(),
        name=name,
        family="TestFamily",
        provider="test",
        model="test-model",
        tier=tier,
        temperature=0.7,
        position=pos,
    )


def _make_grid_with_agents(*specs: tuple[str, tuple[int, int]]) -> tuple[Grid, dict[str, Agent]]:
    """Create a grid and place agents at specified positions."""
    grid = Grid(6)
    agents: dict[str, Agent] = {}
    for name, pos in specs:
        agent = _make_agent(name, pos=pos)
        agent.position = pos
        agents[agent.id] = agent
        grid.place_agent(agent.id, pos)
    return grid, agents


# ---------------------------------------------------------------------------
# Placement tests
# ---------------------------------------------------------------------------

class TestPlacement:
    def test_place_agent_basic(self):
        grid = Grid(6)
        grid.place_agent("a", (0, 0))
        assert grid.get_occupant((0, 0)) == "a"
        assert grid.get_agent_position("a") == (0, 0)

    def test_place_agent_occupied_raises(self):
        grid = Grid(6)
        grid.place_agent("a", (0, 0))
        with pytest.raises(ValueError, match="already occupied"):
            grid.place_agent("b", (0, 0))

    def test_place_agent_out_of_bounds_raises(self):
        grid = Grid(6)
        with pytest.raises(ValueError, match="out of bounds"):
            grid.place_agent("a", (6, 0))
        with pytest.raises(ValueError, match="out of bounds"):
            grid.place_agent("a", (-1, 0))

    def test_starting_positions_full_game(self):
        """Load real config, place 12 agents, verify corner positions."""
        config = load_game_config()
        grid = Grid(config.grid_size)
        agents: dict[str, Agent] = {}
        families: list[Family] = []

        for family_cfg in config.families:
            family = Family.from_config(family_cfg)
            families.append(family)
            for agent_cfg in family_cfg.agents:
                agent = Agent.from_config(agent_cfg)
                agents[agent.id] = agent

        grid.place_starting_positions(families, agents)

        # Verify 12 agents placed
        assert len(grid._positions) == 12

        # Verify corner positions per family
        for i, family in enumerate(families):
            members = sorted(
                [agents[aid] for aid in family.agent_ids],
                key=lambda a: a.tier,
            )
            for agent, expected_pos in zip(members, CORNER_SPAWNS[i]):
                assert agent.position == expected_pos, (
                    f"{agent.name} expected at {expected_pos}, got {agent.position}"
                )

    def test_no_overlapping_positions(self):
        """All 12 starting positions are unique."""
        config = load_game_config()
        grid = Grid(config.grid_size)
        agents: dict[str, Agent] = {}
        families: list[Family] = []

        for family_cfg in config.families:
            family = Family.from_config(family_cfg)
            families.append(family)
            for agent_cfg in family_cfg.agents:
                agent = Agent.from_config(agent_cfg)
                agents[agent.id] = agent

        grid.place_starting_positions(families, agents)
        positions = [a.position for a in agents.values()]
        assert len(set(positions)) == 12


# ---------------------------------------------------------------------------
# Adjacency tests
# ---------------------------------------------------------------------------

class TestAdjacency:
    def test_center_cell_has_8_neighbors(self):
        grid = Grid(6)
        neighbors = grid.get_adjacent(3, 3)
        assert len(neighbors) == 8

    def test_corner_cell_has_3_neighbors(self):
        grid = Grid(6)
        assert len(grid.get_adjacent(0, 0)) == 3
        assert len(grid.get_adjacent(0, 5)) == 3
        assert len(grid.get_adjacent(5, 0)) == 3
        assert len(grid.get_adjacent(5, 5)) == 3

    def test_edge_cell_has_5_neighbors(self):
        grid = Grid(6)
        # Top edge, not corner
        assert len(grid.get_adjacent(0, 3)) == 5
        # Left edge, not corner
        assert len(grid.get_adjacent(3, 0)) == 5

    def test_are_adjacent_orthogonal(self):
        grid = Grid(6)
        assert grid.are_adjacent((2, 2), (2, 3))
        assert grid.are_adjacent((2, 2), (1, 2))

    def test_are_adjacent_diagonal(self):
        grid = Grid(6)
        assert grid.are_adjacent((2, 2), (3, 3))
        assert grid.are_adjacent((2, 2), (1, 1))

    def test_not_adjacent_distance_2(self):
        grid = Grid(6)
        assert not grid.are_adjacent((0, 0), (2, 0))
        assert not grid.are_adjacent((0, 0), (0, 2))
        assert not grid.are_adjacent((0, 0), (2, 2))

    def test_not_adjacent_same_cell(self):
        grid = Grid(6)
        assert not grid.are_adjacent((2, 2), (2, 2))

    def test_get_adjacent_agents(self):
        grid, agents = _make_grid_with_agents(
            ("Alpha", (2, 2)),
            ("Bravo", (2, 3)),
            ("Charlie", (4, 4)),
        )
        adj = grid.get_adjacent_agents("alpha")
        assert "bravo" in adj
        assert "charlie" not in adj


# ---------------------------------------------------------------------------
# Bounds checking
# ---------------------------------------------------------------------------

class TestBounds:
    def test_in_bounds(self):
        grid = Grid(6)
        assert grid.in_bounds((0, 0))
        assert grid.in_bounds((5, 5))
        assert grid.in_bounds((3, 3))

    def test_out_of_bounds(self):
        grid = Grid(6)
        assert not grid.in_bounds((-1, 0))
        assert not grid.in_bounds((0, -1))
        assert not grid.in_bounds((6, 0))
        assert not grid.in_bounds((0, 6))
        assert not grid.in_bounds((6, 6))


# ---------------------------------------------------------------------------
# Movement target
# ---------------------------------------------------------------------------

class TestComputeTarget:
    def test_all_directions(self):
        grid = Grid(6)
        pos = (3, 3)
        assert grid.compute_target(pos, "north") == (2, 3)
        assert grid.compute_target(pos, "south") == (4, 3)
        assert grid.compute_target(pos, "east") == (3, 4)
        assert grid.compute_target(pos, "west") == (3, 2)
        assert grid.compute_target(pos, "ne") == (2, 4)
        assert grid.compute_target(pos, "nw") == (2, 2)
        assert grid.compute_target(pos, "se") == (4, 4)
        assert grid.compute_target(pos, "sw") == (4, 2)

    def test_invalid_direction_raises(self):
        grid = Grid(6)
        with pytest.raises(ValueError, match="unknown direction"):
            grid.compute_target((3, 3), "up")


# ---------------------------------------------------------------------------
# Movement and removal
# ---------------------------------------------------------------------------

class TestMoveAndRemove:
    def test_move_agent(self):
        grid = Grid(6)
        grid.place_agent("a", (0, 0))
        grid.move_agent("a", (0, 1))
        assert grid.get_occupant((0, 0)) is None
        assert grid.get_occupant((0, 1)) == "a"
        assert grid.get_agent_position("a") == (0, 1)

    def test_remove_agent(self):
        grid = Grid(6)
        grid.place_agent("a", (2, 2))
        grid.remove_agent("a")
        assert grid.get_occupant((2, 2)) is None
        assert grid.is_empty((2, 2))

    def test_get_position_missing_raises(self):
        grid = Grid(6)
        with pytest.raises(KeyError):
            grid.get_agent_position("nonexistent")


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------

class TestRender:
    def test_render_has_correct_dimensions(self):
        grid, agents = _make_grid_with_agents(
            ("Atlas", (0, 0)),
            ("Nova", (0, 5)),
        )
        output = grid.render_ascii(agents)
        lines = output.split("\n")
        # header + (separator + row) * 6 + final separator = 1 + 12 + 1 = 14
        assert len(lines) == 14

    def test_render_shows_agent_abbreviations(self):
        grid, agents = _make_grid_with_agents(
            ("Atlas", (0, 0)),
            ("Nova", (0, 5)),
        )
        output = grid.render_ascii(agents)
        assert "At" in output
        assert "No" in output


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_round_trip(self):
        grid = Grid(6)
        grid.place_agent("a", (1, 2))
        grid.place_agent("b", (3, 4))
        d = grid.to_dict()
        assert d["size"] == 6
        assert d["occupants"]["1,2"] == "a"
        assert d["occupants"]["3,4"] == "b"
