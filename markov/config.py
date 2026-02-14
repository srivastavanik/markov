"""
Configuration models. Loaded from config.yaml, validated via Pydantic.
Nothing is hardcoded anywhere else.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


class AgentConfig(BaseModel):
    name: str
    family: str = ""          # populated during load from parent
    provider: str = ""        # populated during load from parent
    model: str
    tier: int                 # 1=Boss, 2=Lieutenant, 3=Soldier
    temperature: float

    @field_validator("tier")
    @classmethod
    def tier_in_range(cls, v: int) -> int:
        if v not in (1, 2, 3):
            raise ValueError(f"tier must be 1, 2, or 3, got {v}")
        return v

    @field_validator("temperature")
    @classmethod
    def temp_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {v}")
        return v


class FamilyConfig(BaseModel):
    name: str
    provider: str
    color: str
    agents: list[AgentConfig]

    def model_post_init(self, __context: object) -> None:
        """Propagate family/provider down to each agent."""
        for agent in self.agents:
            agent.family = self.name
            agent.provider = self.provider


class GameConfig(BaseModel):
    grid_size: int = 6
    max_rounds: int = 60
    stalemate_threshold: int = 15
    discussion_rounds: int = 2
    families: list[FamilyConfig]

    @field_validator("families")
    @classmethod
    def four_families(cls, v: list[FamilyConfig]) -> list[FamilyConfig]:
        if len(v) != 4:
            raise ValueError(f"expected 4 families, got {len(v)}")
        return v


class SeriesConfig(BaseModel):
    num_games: int = 15
    game_config: GameConfig
    series_type: str = "standard"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_game_config(path: Path | str | None = None) -> GameConfig:
    """Load and validate GameConfig from a YAML file."""
    if path is None:
        path = _PROJECT_ROOT / "config.yaml"
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return GameConfig(**raw)
