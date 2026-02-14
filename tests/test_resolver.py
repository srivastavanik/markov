"""Tests for resolver: all specified edge cases for simultaneous resolution."""
import pytest

from markov.agent import Agent
from markov.grid import Grid
from markov.resolver import Action, ActionType, Event, EventType, resolve_actions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent(name: str, pos: tuple[int, int]) -> Agent:
    return Agent(
        id=name.lower(),
        name=name,
        family="Test",
        provider="test",
        model="test",
        tier=1,
        temperature=0.7,
        position=pos,
    )


def _setup(*specs: tuple[str, tuple[int, int]]) -> tuple[Grid, dict[str, Agent]]:
    """Create grid and agents at specified positions."""
    grid = Grid(6)
    agents: dict[str, Agent] = {}
    for name, pos in specs:
        a = _agent(name, pos)
        agents[a.id] = a
        grid.place_agent(a.id, pos)
    return grid, agents


def _events_of_type(events: list[Event], etype: EventType) -> list[Event]:
    return [e for e in events if e.type == etype]


# ---------------------------------------------------------------------------
# Mutual elimination: A kills B, B kills A -> both die
# ---------------------------------------------------------------------------

class TestMutualElimination:
    def test_both_die(self):
        grid, agents = _setup(("Alpha", (2, 2)), ("Bravo", (2, 3)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.ELIMINATE, target="bravo"),
            "bravo": Action(agent_id="bravo", type=ActionType.ELIMINATE, target="alpha"),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        mutual = _events_of_type(events, EventType.MUTUAL_ELIMINATION)
        assert len(mutual) == 2
        assert not agents["alpha"].alive
        assert not agents["bravo"].alive
        assert grid.get_occupant((2, 2)) is None
        assert grid.get_occupant((2, 3)) is None


# ---------------------------------------------------------------------------
# Multi-attacker: A and B both kill C -> C dies, A+B survive
# ---------------------------------------------------------------------------

class TestMultiAttacker:
    def test_target_dies_attackers_survive(self):
        grid, agents = _setup(
            ("Alpha", (2, 2)),
            ("Bravo", (2, 4)),
            ("Charlie", (2, 3)),  # between Alpha and Bravo
        )
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.ELIMINATE, target="charlie"),
            "bravo": Action(agent_id="bravo", type=ActionType.ELIMINATE, target="charlie"),
            "charlie": Action(agent_id="charlie", type=ActionType.STAY),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert not agents["charlie"].alive
        assert agents["alpha"].alive
        assert agents["bravo"].alive
        elims = _events_of_type(events, EventType.ELIMINATION)
        assert len(elims) == 2  # one event per attacker


# ---------------------------------------------------------------------------
# Move collision: A and B both move to (3,3) -> both stay
# ---------------------------------------------------------------------------

class TestMoveCollision:
    def test_both_stay_on_collision(self):
        grid, agents = _setup(("Alpha", (3, 2)), ("Bravo", (3, 4)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.MOVE, direction="east"),
            "bravo": Action(agent_id="bravo", type=ActionType.MOVE, direction="west"),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        # Both should fail and stay in original positions
        assert agents["alpha"].position == (3, 2)
        assert agents["bravo"].position == (3, 4)
        failed = _events_of_type(events, EventType.FAILED_MOVE)
        assert len(failed) == 2


# ---------------------------------------------------------------------------
# Off-grid move -> stays in place
# ---------------------------------------------------------------------------

class TestOffGridMove:
    def test_move_off_north_edge(self):
        grid, agents = _setup(("Alpha", (0, 3)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.MOVE, direction="north"),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert agents["alpha"].position == (0, 3)
        failed = _events_of_type(events, EventType.FAILED_MOVE)
        assert len(failed) == 1
        assert failed[0].details["reason"] == "out_of_bounds"

    def test_move_off_south_edge(self):
        grid, agents = _setup(("Alpha", (5, 3)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.MOVE, direction="south"),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert agents["alpha"].position == (5, 3)

    def test_move_off_corner(self):
        grid, agents = _setup(("Alpha", (0, 0)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.MOVE, direction="nw"),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert agents["alpha"].position == (0, 0)


# ---------------------------------------------------------------------------
# Move onto occupied cell (friendly) -> stays
# ---------------------------------------------------------------------------

class TestMoveOntoOccupied:
    def test_blocked_by_stationary_agent(self):
        grid, agents = _setup(("Alpha", (2, 2)), ("Bravo", (2, 3)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.MOVE, direction="east"),
            "bravo": Action(agent_id="bravo", type=ActionType.STAY),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert agents["alpha"].position == (2, 2)
        failed = _events_of_type(events, EventType.FAILED_MOVE)
        assert len(failed) == 1


# ---------------------------------------------------------------------------
# Eliminate target who moved away -> fails (post-movement adjacency check)
# ---------------------------------------------------------------------------

class TestEliminateMovedTarget:
    def test_target_moves_out_of_range(self):
        # Alpha at (2,2), Bravo at (2,3). Bravo moves east to (2,4).
        # Alpha tries to eliminate Bravo. Post-move Bravo is at (2,4),
        # Alpha is at (2,2) -- not adjacent. Should fail.
        grid, agents = _setup(("Alpha", (2, 2)), ("Bravo", (2, 3)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.ELIMINATE, target="bravo"),
            "bravo": Action(agent_id="bravo", type=ActionType.MOVE, direction="east"),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert agents["bravo"].alive
        assert agents["bravo"].position == (2, 4)
        failed = _events_of_type(events, EventType.FAILED_ELIMINATE)
        assert len(failed) == 1
        assert failed[0].details["reason"] == "not_adjacent"


# ---------------------------------------------------------------------------
# Agent moves into adjacency, third agent eliminates target
# -> target dies (movement resolves before eliminations)
# ---------------------------------------------------------------------------

class TestMoveIntoAdjacencyThenEliminate:
    def test_mover_can_eliminate_after_arriving(self):
        # Alpha at (2,1), moves east to (2,2).
        # Charlie at (2,3), stationary.
        # Post-move: Alpha at (2,2), adjacent to Charlie at (2,3).
        # Alpha can't eliminate (they moved). But let's test:
        # Bravo at (2,1) moves east, Alpha at (2,2) eliminates Charlie.
        # Actually: the spec says agent C is already adjacent and eliminates B.
        # Let's set up: Alpha moves to become adjacent to Charlie, Bravo (already adjacent) eliminates Charlie.
        grid, agents = _setup(
            ("Alpha", (2, 1)),   # moves east to (2,2)
            ("Bravo", (2, 4)),   # stays, adjacent to Charlie
            ("Charlie", (2, 3)), # the target
        )
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.MOVE, direction="east"),
            "bravo": Action(agent_id="bravo", type=ActionType.ELIMINATE, target="charlie"),
            "charlie": Action(agent_id="charlie", type=ActionType.STAY),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert not agents["charlie"].alive
        assert agents["alpha"].position == (2, 2)
        assert agents["bravo"].alive


# ---------------------------------------------------------------------------
# Move into cell vacated by agent who moved away -> succeeds
# ---------------------------------------------------------------------------

class TestMoveIntoVacated:
    def test_swap_positions_via_vacated_cell(self):
        # Alpha at (2,2), Bravo at (2,3).
        # Bravo moves east to (2,4). Alpha moves east to (2,3) -- now vacant.
        grid, agents = _setup(("Alpha", (2, 2)), ("Bravo", (2, 3)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.MOVE, direction="east"),
            "bravo": Action(agent_id="bravo", type=ActionType.MOVE, direction="east"),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert agents["alpha"].position == (2, 3)
        assert agents["bravo"].position == (2, 4)

    def test_chain_of_three_all_move_same_direction(self):
        # A(2,1) -> east, B(2,2) -> east, C(2,3) -> east
        # C moves to (2,4) freeing (2,3), B moves to (2,3) freeing (2,2), A moves to (2,2)
        grid, agents = _setup(
            ("Alpha", (2, 1)),
            ("Bravo", (2, 2)),
            ("Charlie", (2, 3)),
        )
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.MOVE, direction="east"),
            "bravo": Action(agent_id="bravo", type=ActionType.MOVE, direction="east"),
            "charlie": Action(agent_id="charlie", type=ActionType.MOVE, direction="east"),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert agents["alpha"].position == (2, 2)
        assert agents["bravo"].position == (2, 3)
        assert agents["charlie"].position == (2, 4)


# ---------------------------------------------------------------------------
# Eliminate non-adjacent agent -> fails
# ---------------------------------------------------------------------------

class TestEliminateNonAdjacent:
    def test_far_away_target(self):
        grid, agents = _setup(("Alpha", (0, 0)), ("Bravo", (5, 5)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.ELIMINATE, target="bravo"),
            "bravo": Action(agent_id="bravo", type=ActionType.STAY),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert agents["bravo"].alive
        failed = _events_of_type(events, EventType.FAILED_ELIMINATE)
        assert len(failed) == 1
        assert failed[0].details["reason"] == "not_adjacent"


# ---------------------------------------------------------------------------
# Stay action -> no position change
# ---------------------------------------------------------------------------

class TestStay:
    def test_stay_keeps_position(self):
        grid, agents = _setup(("Alpha", (3, 3)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.STAY),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert agents["alpha"].position == (3, 3)
        stays = _events_of_type(events, EventType.STAY)
        assert len(stays) == 1


# ---------------------------------------------------------------------------
# Chain dependency: A -> B's cell, B -> C's cell, C stays -> both A and B fail
# ---------------------------------------------------------------------------

class TestChainDependency:
    def test_blocked_chain(self):
        # A at (2,1), B at (2,2), C at (2,3).
        # A moves east to (2,2), B moves east to (2,3), C stays.
        # C stays -> (2,3) is firm -> B can't move -> (2,2) is firm -> A can't move.
        grid, agents = _setup(
            ("Alpha", (2, 1)),
            ("Bravo", (2, 2)),
            ("Charlie", (2, 3)),
        )
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.MOVE, direction="east"),
            "bravo": Action(agent_id="bravo", type=ActionType.MOVE, direction="east"),
            "charlie": Action(agent_id="charlie", type=ActionType.STAY),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert agents["alpha"].position == (2, 1)
        assert agents["bravo"].position == (2, 2)
        assert agents["charlie"].position == (2, 3)
        failed = _events_of_type(events, EventType.FAILED_MOVE)
        assert len(failed) == 2


# ---------------------------------------------------------------------------
# Valid move
# ---------------------------------------------------------------------------

class TestValidMove:
    def test_simple_move(self):
        grid, agents = _setup(("Alpha", (3, 3)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.MOVE, direction="north"),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert agents["alpha"].position == (2, 3)
        moves = _events_of_type(events, EventType.MOVE)
        assert len(moves) == 1


# ---------------------------------------------------------------------------
# Valid elimination (adjacent, no mutual)
# ---------------------------------------------------------------------------

class TestValidElimination:
    def test_simple_kill(self):
        grid, agents = _setup(("Alpha", (2, 2)), ("Bravo", (2, 3)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.ELIMINATE, target="bravo"),
            "bravo": Action(agent_id="bravo", type=ActionType.STAY),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert not agents["bravo"].alive
        assert agents["alpha"].alive
        elims = _events_of_type(events, EventType.ELIMINATION)
        assert len(elims) == 1
        assert elims[0].agent_id == "alpha"
        assert elims[0].details["target"] == "bravo"

    def test_diagonal_kill(self):
        grid, agents = _setup(("Alpha", (2, 2)), ("Bravo", (3, 3)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.ELIMINATE, target="bravo"),
            "bravo": Action(agent_id="bravo", type=ActionType.STAY),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert not agents["bravo"].alive


# ---------------------------------------------------------------------------
# Eliminate after moving into range
# ---------------------------------------------------------------------------

class TestEliminateAfterMovingIntoRange:
    def test_another_agent_moves_adjacent_then_is_killed(self):
        """
        Alpha stays at (2,2). Bravo at (2,4) moves west to (2,3).
        Post-move: Alpha at (2,2), Bravo at (2,3) -- adjacent.
        Alpha eliminates Bravo. Should succeed because eliminations
        use post-movement positions.
        """
        grid, agents = _setup(("Alpha", (2, 2)), ("Bravo", (2, 4)))
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.ELIMINATE, target="bravo"),
            "bravo": Action(agent_id="bravo", type=ActionType.MOVE, direction="west"),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        # Bravo moved to (2,3), now adjacent to Alpha at (2,2)
        assert not agents["bravo"].alive
        elims = _events_of_type(events, EventType.ELIMINATION)
        assert len(elims) == 1


# ---------------------------------------------------------------------------
# Dead agents can't act
# ---------------------------------------------------------------------------

class TestDeadAgentsCantAct:
    def test_dead_agent_action_ignored(self):
        grid, agents = _setup(("Alpha", (2, 2)), ("Bravo", (2, 3)))
        agents["alpha"].alive = False
        grid.remove_agent("alpha")
        actions = {
            "alpha": Action(agent_id="alpha", type=ActionType.ELIMINATE, target="bravo"),
            "bravo": Action(agent_id="bravo", type=ActionType.STAY),
        }
        events = resolve_actions(actions, grid, agents, round_num=1)
        assert agents["bravo"].alive
