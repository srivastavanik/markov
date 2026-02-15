from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from darwin.api.game_runner import GameRunner
from darwin.api.schemas import (
    CancelGameResponse,
    GameSummary,
    HealthResponse,
    SeriesDetail,
    SeriesSummary,
    StartGameRequest,
    StartGameResponse,
)
from darwin.server import GameBroadcaster

WS_HOST = os.getenv("MARKOV_WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("MARKOV_WS_PORT", "8765"))
# When MARKOV_SINGLE_PORT is set, WS is served on the same port via /ws endpoint
_SINGLE_PORT = os.getenv("MARKOV_SINGLE_PORT", "").lower() in ("1", "true", "yes")
WS_PUBLIC_URL = os.getenv("MARKOV_WS_PUBLIC_URL", f"ws://localhost:{WS_PORT}")

broadcaster: GameBroadcaster | None = None
runner: GameRunner | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global broadcaster, runner
    broadcaster = GameBroadcaster(host=WS_HOST, port=WS_PORT)
    if not _SINGLE_PORT:
        # Standalone WS server on separate port (local dev)
        await broadcaster.start()
    # In single-port mode, WS is handled by the /ws FastAPI endpoint below
    runner = GameRunner(broadcaster=broadcaster)
    try:
        yield
    finally:
        if broadcaster and not _SINGLE_PORT:
            await broadcaster.stop()


app = FastAPI(title="DARWIN API", version="1.0.0", lifespan=lifespan)

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


@app.get("/api/games/{game_id}/state")
async def get_game_state(game_id: str) -> dict:
    """Get cached game state (init + rounds) for late-joining dashboard clients."""
    if not broadcaster:
        raise HTTPException(status_code=503, detail="Broadcaster not ready")
    init = broadcaster._last_init.get(game_id) or broadcaster._last_init.get(None)
    rounds = broadcaster._last_rounds.get(game_id) or broadcaster._last_rounds.get(None) or []
    if not init:
        raise HTTPException(status_code=404, detail="No cached state")
    return {"init": init, "rounds": rounds}


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


@app.get("/api/games/{game_id}/metrics")
async def get_game_metrics(game_id: str) -> dict:
    payload = _runner().get_metrics_payload(game_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return payload


@app.get("/api/games/{game_id}/analysis")
async def get_game_analysis(game_id: str) -> dict | list:
    payload = _runner().get_analysis_payload(game_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return payload


@app.get("/api/series", response_model=list[SeriesSummary])
async def list_series() -> list[SeriesSummary]:
    return [SeriesSummary(**row) for row in _runner().list_series()]


@app.get("/api/series/{series_id}", response_model=SeriesDetail)
async def get_series(series_id: str) -> SeriesDetail:
    row = _runner().get_series_detail(series_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Series not found")
    return SeriesDetail(**row)


# ---------------------------------------------------------------------------
# WebSocket endpoint (single-port mode for cloud deployment)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str | None = None) -> None:
    """Bridge FastAPI WebSocket to the GameBroadcaster.

    When MARKOV_SINGLE_PORT=1, the standalone websockets server is disabled
    and this endpoint serves WS on the same port as the REST API.
    Works in both modes so local dev can also use /ws if desired.
    """
    if not broadcaster:
        await websocket.close(code=1011, reason="Broadcaster not ready")
        return

    # Token check
    token = websocket.query_params.get("token")
    if broadcaster.token and token != broadcaster.token:
        await websocket.close(code=4401, reason="Unauthorized")
        return

    await websocket.accept()

    # Register as a pseudo-client on the broadcaster
    # We use a lightweight shim so the broadcaster can send to this client
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=256)

    class _FastAPIClient:
        """Duck-type shim for websockets.ServerConnection used by broadcaster."""
        def __hash__(self) -> int:
            return id(self)
        def __eq__(self, other: object) -> bool:
            return self is other

    shim = _FastAPIClient()
    broadcaster.clients.add(shim)  # type: ignore[arg-type]
    broadcaster.client_game_ids[shim] = game_id  # type: ignore[index]

    # Monkey-patch _safe_send to route messages to our queue
    _orig_safe_send = broadcaster._safe_send

    async def _patched_safe_send(client: object, payload: str) -> None:
        if client is shim:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # drop if client is too slow
        else:
            await _orig_safe_send(client, payload)  # type: ignore[arg-type]

    broadcaster._safe_send = _patched_safe_send  # type: ignore[assignment]

    try:
        # Replay cached state for late joiners
        init_payload = broadcaster._last_init.get(game_id) or broadcaster._last_init.get(None)
        if init_payload:
            await websocket.send_text(json.dumps(init_payload, default=str))
        for rp in broadcaster._last_rounds.get(game_id) or broadcaster._last_rounds.get(None) or []:
            await websocket.send_text(json.dumps(rp, default=str))

        # Two concurrent tasks: forward broadcasts to client, and read (ignore) from client
        async def _sender() -> None:
            while True:
                payload = await queue.get()
                await websocket.send_text(payload)

        async def _receiver() -> None:
            while True:
                await websocket.receive_text()  # dashboard is read-only; ignore

        await asyncio.gather(_sender(), _receiver())

    except (WebSocketDisconnect, Exception):
        pass
    finally:
        broadcaster._safe_send = _orig_safe_send  # type: ignore[assignment]
        broadcaster.clients.discard(shim)  # type: ignore[arg-type]
        broadcaster.client_game_ids.pop(shim, None)  # type: ignore[arg-type]

