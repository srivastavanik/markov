"""
Family grouping and channel management.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from markov.agent import Agent
from markov.config import FamilyConfig


@dataclass
class Family:
    name: str
    provider: str
    color: str
    agent_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: FamilyConfig) -> Family:
        return cls(
            name=cfg.name,
            provider=cfg.provider,
            color=cfg.color,
            agent_ids=[a.name.lower() for a in cfg.agents],
        )

    def living_members(self, agents: dict[str, Agent]) -> list[Agent]:
        """Return living agents in this family."""
        return [agents[aid] for aid in self.agent_ids if agents[aid].alive]

    def is_eliminated(self, agents: dict[str, Agent]) -> bool:
        """True if every member of this family is dead."""
        return all(not agents[aid].alive for aid in self.agent_ids)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "color": self.color,
            "agent_ids": self.agent_ids,
        }
