"""
Agent runtime state. No LLM logic -- just state tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from markov.config import AgentConfig


@dataclass
class Agent:
    id: str
    name: str
    family: str
    provider: str
    model: str
    tier: int               # 1=Boss, 2=Lieutenant, 3=Soldier
    temperature: float
    alive: bool = True
    position: tuple[int, int] = (0, 0)
    eliminated_by: str | None = None
    eliminated_round: int | None = None
    rounds_survived: int = 0

    # Full history -- the dataset
    thought_log: list[dict] = field(default_factory=list)
    message_log: list[dict] = field(default_factory=list)
    action_log: list[dict] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: AgentConfig) -> Agent:
        """Create an Agent from an AgentConfig. Position set later by Grid."""
        agent_id = cfg.name.lower()
        return cls(
            id=agent_id,
            name=cfg.name,
            family=cfg.family,
            provider=cfg.provider,
            model=cfg.model,
            tier=cfg.tier,
            temperature=cfg.temperature,
        )

    def eliminate(self, by: str, round_num: int) -> None:
        """Mark this agent as eliminated."""
        self.alive = False
        self.eliminated_by = by
        self.eliminated_round = round_num

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "provider": self.provider,
            "tier": self.tier,
            "alive": self.alive,
            "position": list(self.position),
            "eliminated_by": self.eliminated_by,
            "eliminated_round": self.eliminated_round,
            "rounds_survived": self.rounds_survived,
        }
