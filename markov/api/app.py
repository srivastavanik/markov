from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from markov.api.game_runner import GameRunner
from markov.api.schemas import (
    CancelGameResponse,
    GameSummary,
    HealthResponse,
    StartGameRequest,
    StartGameResponse,
)
from markov.server import GameBroadcaster

WS_HOST = os.getenv("MARKOV_WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("MARKOV_WS_PORT", "8765"))
WS_PUBLIC_URL = os.getenv("MARKOV_WS_PUBLIC_URL", f"ws://localhost:{WS_PORT}")

broadcaster: GameBroadcaster | None = None
runner: GameRunner | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global broadcaster, runner
    broadcaster = GameBroadcaster(host=WS_HOST, port=WS_PORT)
    await broadcaster.start()
    runner = GameRunner(broadcaster=broadcaster)
    try:
        yield
    finally:
        if broadcaster:
            await broadcaster.stop()


app = FastAPI(title="MARKOV API", version="1.0.0", lifespan=lifespan)

cors_origins = os.getenv("MARKOV_CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _runner() -> GameRunner:
    if not runner:
        raise RuntimeError("Game runner unavailable")
    return runner


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", websocket_host=WS_HOST, websocket_port=WS_PORT)


@app.get("/api/config")
async def api_config() -> dict:
    ws_token_required = bool(os.getenv("MARKOV_WS_TOKEN"))
    return {
        "ws_url": WS_PUBLIC_URL,
        "ws_token_required": ws_token_required,
    }


@app.post("/api/games", response_model=StartGameResponse)
async def start_game(request: StartGameRequest) -> StartGameResponse:
    job = _runner().start_game(mode=request.mode, verbose=request.verbose)
    return StartGameResponse(
        game_id=job.game_id,
        status=job.status,
        ws_url=WS_PUBLIC_URL,
        ws_token_required=bool(os.getenv("MARKOV_WS_TOKEN")),
    )


@app.get("/api/games", response_model=list[GameSummary])
async def list_games() -> list[GameSummary]:
    return [GameSummary(**job.to_dict()) for job in _runner().list_jobs()]


@app.get("/api/games/{game_id}", response_model=GameSummary)
async def get_game(game_id: str) -> GameSummary:
    job = _runner().get_job(game_id)
    if not job:
        raise HTTPException(status_code=404, detail="Game not found")
    return GameSummary(**job.to_dict())


@app.post("/api/games/{game_id}/cancel", response_model=CancelGameResponse)
async def cancel_game(game_id: str) -> CancelGameResponse:
    job = _runner().request_cancel(game_id)
    if not job:
        raise HTTPException(status_code=404, detail="Game not found")
    return CancelGameResponse(game_id=game_id, status=job.status)


@app.get("/api/games/{game_id}/replay")
async def get_replay(game_id: str) -> dict:
    payload = _runner().get_replay_payload(game_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Replay not found")
    return payload

