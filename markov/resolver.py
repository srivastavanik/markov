"""
Action types and simultaneous resolution logic.

Resolution order:
  1. All moves resolve first (with iterative chain/collision handling).
  2. Eliminations resolve against post-movement positions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from markov.agent import Agent
from markov.grid import Grid


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    MOVE = "move"
    STAY = "stay"
    ELIMINATE = "eliminate"


class EventType(str, Enum):
    MOVE = "move"
    STAY = "stay"
    ELIMINATION = "elimination"
    MUTUAL_ELIMINATION = "mutual_elimination"
    FAILED_MOVE = "failed_move"
    FAILED_ELIMINATE = "failed_eliminate"


@dataclass
class Action:
    agent_id: str
    type: ActionType
    direction: str | None = None   # for move
    target: str | None = None      # for eliminate


@dataclass
class Event:
    round: int
    type: EventType
    agent_id: str
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_actions(
    actions: dict[str, Action],
    grid: Grid,
    agents: dict[str, Agent],
    round_num: int,
) -> list[Event]:
    """
    Resolve all actions simultaneously. Returns events describing what happened.

    Algorithm:
      Phase 1 -- Movement (iterative until stable)
      Phase 2 -- Eliminations (against post-move positions)
    """
    events: list[Event] = []

    # Partition actions
    moves: dict[str, Action] = {}
    eliminates: dict[str, Action] = {}
    for aid, action in actions.items():
        if not agents[aid].alive:
            continue
        if action.type == ActionType.MOVE:
            moves[aid] = action
        elif action.type == ActionType.ELIMINATE:
            eliminates[aid] = action
        else:
            # stay -- log it, nothing to resolve
            events.append(Event(
                round=round_num,
                type=EventType.STAY,
                agent_id=aid,
                details={"position": list(agents[aid].position)},
            ))

    # ------------------------------------------------------------------
    # Phase 1: Resolve movements
    # ------------------------------------------------------------------
    move_events = _resolve_moves(moves, grid, agents, round_num)
    events.extend(move_events)

    # ------------------------------------------------------------------
    # Phase 2: Resolve eliminations (post-movement positions)
    # ------------------------------------------------------------------
    elim_events = _resolve_eliminations(eliminates, grid, agents, round_num)
    events.extend(elim_events)

    return events


def _resolve_moves(
    moves: dict[str, Action],
    grid: Grid,
    agents: dict[str, Agent],
    round_num: int,
) -> list[Event]:
    """
    Resolve all move actions simultaneously with iterative chain handling.

    1. Compute intended target for each mover.
    2. Fail moves that target out-of-bounds cells.
    3. Iteratively resolve until stable:
       a. Fail moves targeting cells occupied by non-movers.
       b. Fail moves targeting cells occupied by movers whose moves already failed.
       c. Fail moves where two+ movers target the same cell (collision).
    4. Apply surviving valid moves to the grid.
    """
    events: list[Event] = []

    # Compute intended destinations
    intended: dict[str, tuple[int, int]] = {}
    failed: set[str] = set()

    for aid, action in moves.items():
        current_pos = agents[aid].position
        target = grid.compute_target(current_pos, action.direction)
        if not grid.in_bounds(target):
            failed.add(aid)
            events.append(Event(
                round=round_num,
                type=EventType.FAILED_MOVE,
                agent_id=aid,
                details={
                    "reason": "out_of_bounds",
                    "from": list(current_pos),
                    "intended": list(target),
                },
            ))
        else:
            intended[aid] = target

    # Set of agents NOT moving (staying, eliminating, or failed movers).
    # Their cells are "firm" -- can't be moved into.
    all_agent_ids = {aid for aid, a in agents.items() if a.alive}
    moving_ids = set(intended.keys())  # still-valid movers

    # Iteratively resolve until no new failures
    changed = True
    while changed:
        changed = False

        # Identify firm cells: occupied by anyone NOT in the valid-moving set
        firm_cells: set[tuple[int, int]] = set()
        for aid in all_agent_ids:
            if aid not in moving_ids:
                firm_cells.add(agents[aid].position)

        # Check each mover's target against firm cells
        newly_failed: set[str] = set()
        for aid in list(moving_ids):
            if intended[aid] in firm_cells:
                newly_failed.add(aid)

        # Check for collisions: multiple movers targeting the same cell
        target_counts: dict[tuple[int, int], list[str]] = {}
        for aid in moving_ids:
            if aid in newly_failed:
                continue
            target_counts.setdefault(intended[aid], []).append(aid)

        for _target, contenders in target_counts.items():
            if len(contenders) > 1:
                for aid in contenders:
                    newly_failed.add(aid)

        # Apply failures
        for aid in newly_failed:
            if aid in moving_ids:
                moving_ids.discard(aid)
                failed.add(aid)
                events.append(Event(
                    round=round_num,
                    type=EventType.FAILED_MOVE,
                    agent_id=aid,
                    details={
                        "reason": "blocked" if intended[aid] in firm_cells else "collision",
                        "from": list(agents[aid].position),
                        "intended": list(intended[aid]),
                    },
                ))
                changed = True

    # Apply valid moves
    for aid in moving_ids:
        old_pos = agents[aid].position
        new_pos = intended[aid]
        grid.move_agent(aid, new_pos)
        agents[aid].position = new_pos
        events.append(Event(
            round=round_num,
            type=EventType.MOVE,
            agent_id=aid,
            details={"from": list(old_pos), "to": list(new_pos)},
        ))

    return events


def _resolve_eliminations(
    eliminates: dict[str, Action],
    grid: Grid,
    agents: dict[str, Agent],
    round_num: int,
) -> list[Event]:
    """
    Resolve elimination actions against post-movement positions.

    - Target must be adjacent (post-move). If not, failed_eliminate.
    - If A eliminates B and B eliminates A, mutual elimination (both die).
    - If multiple attackers target same victim, victim dies, all survive.
    """
    events: list[Event] = []
    dead: set[str] = set()

    # First pass: validate adjacency
    valid_elims: dict[str, str] = {}  # attacker_id -> target_id
    for aid, action in eliminates.items():
        target_id = action.target
        if target_id is None or target_id not in agents or not agents[target_id].alive:
            events.append(Event(
                round=round_num,
                type=EventType.FAILED_ELIMINATE,
                agent_id=aid,
                details={"reason": "invalid_target", "target": target_id},
            ))
            continue

        attacker_pos = agents[aid].position
        target_pos = agents[target_id].position
        if not grid.are_adjacent(attacker_pos, target_pos):
            events.append(Event(
                round=round_num,
                type=EventType.FAILED_ELIMINATE,
                agent_id=aid,
                details={
                    "reason": "not_adjacent",
                    "target": target_id,
                    "attacker_pos": list(attacker_pos),
                    "target_pos": list(target_pos),
                },
            ))
            continue

        valid_elims[aid] = target_id

    # Second pass: detect mutual eliminations
    mutual_pairs: set[frozenset[str]] = set()
    for aid, tid in valid_elims.items():
        if tid in valid_elims and valid_elims[tid] == aid:
            mutual_pairs.add(frozenset({aid, tid}))

    # Process mutual eliminations
    for pair in mutual_pairs:
        a, b = tuple(pair)
        dead.add(a)
        dead.add(b)
        events.append(Event(
            round=round_num,
            type=EventType.MUTUAL_ELIMINATION,
            agent_id=a,
            details={"target": b, "mutual": True},
        ))
        events.append(Event(
            round=round_num,
            type=EventType.MUTUAL_ELIMINATION,
            agent_id=b,
            details={"target": a, "mutual": True},
        ))

    # Process normal eliminations (skip mutual pairs already handled)
    mutual_agents = {aid for pair in mutual_pairs for aid in pair}
    for aid, tid in valid_elims.items():
        if aid in mutual_agents:
            continue
        dead.add(tid)
        events.append(Event(
            round=round_num,
            type=EventType.ELIMINATION,
            agent_id=aid,
            details={"target": tid},
        ))

    # Apply deaths
    for aid in dead:
        agents[aid].eliminate(by="combat", round_num=round_num)
        grid.remove_agent(aid)

    # Set proper eliminated_by from events
    for ev in events:
        if ev.type == EventType.ELIMINATION:
            target_id = ev.details["target"]
            agents[target_id].eliminated_by = ev.agent_id
        elif ev.type == EventType.MUTUAL_ELIMINATION:
            target_id = ev.details["target"]
            agents[ev.agent_id].eliminated_by = target_id

    return events
