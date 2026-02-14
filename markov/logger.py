"""
JSON transcript logging. One file per game in data/games/.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from markov.communication import Message
from markov.resolver import Event


# ---------------------------------------------------------------------------
# Game logger
# ---------------------------------------------------------------------------

class GameLogger:
    """Accumulates a full game transcript and writes it to JSON."""

    def __init__(self) -> None:
        self.config_snapshot: dict = {}
        self.rounds: list[dict] = []
        self.result: dict = {}
        self.cost: dict = {}
        self.start_time: str = datetime.now(timezone.utc).isoformat()

    def set_config(self, config_dict: dict) -> None:
        self.config_snapshot = config_dict

    def log_round(
        self,
        round_num: int,
        family_discussions: list[dict] | None = None,
        messages: list[Message] | None = None,
        thoughts: dict[str, str] | None = None,
        actions: dict[str, dict] | None = None,
        events: list[Event] | None = None,
    ) -> None:
        """Log a single round's data."""
        entry: dict = {"round": round_num}

        if family_discussions is not None:
            entry["family_discussions"] = family_discussions

        if messages is not None:
            entry["messages"] = [m.to_dict() for m in messages]

        if thoughts is not None:
            entry["thoughts"] = thoughts

        if actions is not None:
            entry["actions"] = actions

        if events is not None:
            entry["events"] = [
                {"type": e.type.value if hasattr(e.type, 'value') else str(e.type),
                 "agent_id": e.agent_id, "details": e.details}
                for e in events
            ]

        self.rounds.append(entry)

    def set_result(
        self,
        winner_id: str | None,
        winner_name: str | None,
        total_rounds: int,
        final_reflection: str | None = None,
        surviving: list[str] | None = None,
    ) -> None:
        self.result = {
            "winner_id": winner_id,
            "winner_name": winner_name,
            "total_rounds": total_rounds,
            "final_reflection": final_reflection,
            "surviving": surviving or [],
        }

    def set_cost(self, cost_summary: dict) -> None:
        self.cost = cost_summary

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "config": self.config_snapshot,
            "rounds": self.rounds,
            "result": self.result,
            "cost": self.cost,
        }

    def save(self, path: Path | str | None = None) -> Path:
        """Write transcript to JSON. Returns the file path."""
        if path is None:
            data_dir = Path(__file__).resolve().parent.parent / "data" / "games"
            data_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = data_dir / f"game_{timestamp}.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        return path
