"""
6x6 grid. Tracks agent positions, handles adjacency, renders ASCII.
"""
from __future__ import annotations

import random

from darwin.agent import Agent
from darwin.family import Family

# Direction -> (row_delta, col_delta)
DIRECTION_DELTAS: dict[str, tuple[int, int]] = {
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
    "ne": (-1, 1),
    "nw": (-1, -1),
    "se": (1, 1),
    "sw": (1, -1),
}


class Grid:
    """
    Flat dict-based grid. cells[(row,col)] -> agent_id | None.
    Only occupied cells are stored; absence means empty.
    """

    def __init__(self, size: int = 6) -> None:
        self.size = size
        self._occupants: dict[tuple[int, int], str] = {}
        self._positions: dict[str, tuple[int, int]] = {}  # agent_id -> pos

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def place_agent(self, agent_id: str, pos: tuple[int, int]) -> None:
        """Place an agent on the grid. Raises if cell is occupied."""
        if not self.in_bounds(pos):
            raise ValueError(f"position {pos} out of bounds for grid size {self.size}")
        if pos in self._occupants:
            raise ValueError(f"cell {pos} already occupied by {self._occupants[pos]}")
        self._occupants[pos] = agent_id
        self._positions[agent_id] = pos

    def place_starting_positions(
        self, families: list[Family], agents: dict[str, Agent]
    ) -> None:
        """
        Place all agents in fully randomized positions across the grid.
        No family clustering — avoids positional bias.
        """
        all_cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        random.shuffle(all_cells)

        all_agents = []
        for family in families:
            all_agents.extend(agents[aid] for aid in family.agent_ids)
        random.shuffle(all_agents)

        for agent, pos in zip(all_agents, all_cells):
            agent.position = pos
            self.place_agent(agent.id, pos)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size

    def get_occupant(self, pos: tuple[int, int]) -> str | None:
        return self._occupants.get(pos)

    def is_empty(self, pos: tuple[int, int]) -> bool:
        return pos not in self._occupants

    def get_agent_position(self, agent_id: str) -> tuple[int, int]:
        if agent_id not in self._positions:
            raise KeyError(f"agent {agent_id!r} not on grid")
        return self._positions[agent_id]

    def get_adjacent(self, row: int, col: int) -> list[tuple[int, int]]:
        """All valid 8-directional neighbors within bounds."""
        neighbors: list[tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    neighbors.append((nr, nc))
        return neighbors

    def are_adjacent(self, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> bool:
        """True if Chebyshev distance <= 1 (8-directional adjacency)."""
        if pos_a == pos_b:
            return False
        return max(abs(pos_a[0] - pos_b[0]), abs(pos_a[1] - pos_b[1])) <= 1

    def compute_target(
        self, pos: tuple[int, int], direction: str
    ) -> tuple[int, int]:
        """Given a position and direction string, return target cell."""
        if direction not in DIRECTION_DELTAS:
            raise ValueError(f"unknown direction {direction!r}")
        dr, dc = DIRECTION_DELTAS[direction]
        return (pos[0] + dr, pos[1] + dc)

    def get_adjacent_agents(self, agent_id: str) -> list[str]:
        """Return IDs of all agents adjacent to the given agent."""
        pos = self.get_agent_position(agent_id)
        result: list[str] = []
        for neighbor in self.get_adjacent(*pos):
            occ = self.get_occupant(neighbor)
            if occ is not None:
                result.append(occ)
        return result

    # ------------------------------------------------------------------
    # Mutation (used by resolver)
    # ------------------------------------------------------------------

    def move_agent(self, agent_id: str, new_pos: tuple[int, int]) -> None:
        """Move an agent to a new cell. Caller must verify validity."""
        old_pos = self._positions[agent_id]
        del self._occupants[old_pos]
        self._occupants[new_pos] = agent_id
        self._positions[agent_id] = new_pos

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the grid (elimination)."""
        if agent_id in self._positions:
            pos = self._positions.pop(agent_id)
            del self._occupants[pos]

    def shrink(self, new_size: int) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
        """Shrink grid to new_size, pushing border agents inward.

        Agents on the removed edge (row >= new_size or col >= new_size) are
        clamped to the nearest valid position. If that cell is occupied,
        BFS finds the closest empty cell within new bounds.

        Returns dict mapping displaced agent_id -> (old_pos, new_pos).
        """
        if new_size >= self.size:
            return {}

        self.size = new_size

        # Identify displaced agents (now out of bounds)
        displaced: list[tuple[str, tuple[int, int]]] = []
        for agent_id in list(self._positions):
            pos = self._positions[agent_id]
            r, c = pos
            if r >= new_size or c >= new_size:
                displaced.append((agent_id, pos))
                del self._occupants[pos]
                del self._positions[agent_id]

        # Process closest-to-edge first so they land near their original spot
        displaced.sort(key=lambda x: max(x[1][0], x[1][1]))

        moves: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
        for agent_id, old_pos in displaced:
            # Clamp to new max coordinate (move inward by 1)
            clamped = (min(old_pos[0], new_size - 1), min(old_pos[1], new_size - 1))
            target = self._nearest_empty(clamped)
            if target is not None:
                self._occupants[target] = agent_id
                self._positions[agent_id] = target
                moves[agent_id] = (old_pos, target)

        return moves

    def _nearest_empty(self, start: tuple[int, int]) -> tuple[int, int] | None:
        """Find nearest empty in-bounds cell via BFS from start."""
        from collections import deque

        if self.in_bounds(start) and start not in self._occupants:
            return start

        visited: set[tuple[int, int]] = {start}
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    pos = (r + dr, c + dc)
                    if pos in visited or not self.in_bounds(pos):
                        continue
                    visited.add(pos)
                    if pos not in self._occupants:
                        return pos
                    queue.append(pos)
        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_ascii(self, agents: dict[str, Agent]) -> str:
        """
        Render board as ASCII for agent prompts.

        Agent names truncated to 2 chars. Empty cells are spaces.
        Example:
              0   1   2   3   4   5
            +---+---+---+---+---+---+
          0 |At |Ci |   |   |   |No |
            +---+---+---+---+---+---+
          ...
        """
        # Build name lookup: agent_id -> 2-char abbreviation
        abbrev: dict[str, str] = {}
        for aid, agent in agents.items():
            abbrev[aid] = agent.name[:2]

        header = "    " + "   ".join(str(c) for c in range(self.size))
        separator = "  +" + "---+" * self.size
        lines = [header]
        for r in range(self.size):
            lines.append(separator)
            row_cells: list[str] = []
            for c in range(self.size):
                occ = self.get_occupant((r, c))
                if occ is not None:
                    row_cells.append(f"{abbrev.get(occ, '??'):2s} ")
                else:
                    row_cells.append("   ")
            lines.append(f"{r} |" + "|".join(row_cells) + "|")
        lines.append(separator)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "occupants": {f"{r},{c}": aid for (r, c), aid in self._occupants.items()},
        }
