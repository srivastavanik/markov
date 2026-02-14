from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


GameMode = Literal["full", "quick"]
GameStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class StartGameRequest(BaseModel):
    mode: GameMode = "full"
    verbose: bool = False


class StartGameResponse(BaseModel):
    game_id: str
    status: GameStatus
    ws_url: str
    ws_token_required: bool


class GameSummary(BaseModel):
    game_id: str
    status: GameStatus
    mode: GameMode
    started_at: datetime
    ended_at: datetime | None = None
    output_dir: str | None = None
    error: str | None = None
    winner: str | None = None
    total_rounds: int | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    websocket_host: str
    websocket_port: int


class CancelGameResponse(BaseModel):
    game_id: str
    status: GameStatus

