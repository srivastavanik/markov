"""
WebSocket server. Pushes game state to the dashboard after every round,
plus realtime events during rounds. Also supports replay mode for reviewing
past games.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import websockets
from websockets.server import ServerConnection

from darwin.agent import Agent
from darwin.communication import Message
from darwin.family import Family
from darwin.highlights import Highlight
from darwin.resolver import Event, EventType

logger = logging.getLogger("darwin.server")


class GameBroadcaster:
    """Manages WebSocket connections and broadcasts game state."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, token: str | None = None) -> None:
        self.host = host
        self.port = port
        self.token = token or os.getenv("MARKOV_WS_TOKEN")
        self.clients: set[ServerConnection] = set()
        self.client_game_ids: dict[ServerConnection, str | None] = {}
        self._server: Any = None
        # Cache latest payloads per game for late-joining clients
        self._last_init: dict[str | None, dict] = {}
        self._last_rounds: dict[str | None, list[dict]] = {}

    async def start(self) -> None:
        self._server = await websockets.serve(
            self._handle_client, self.host, self.port,
        )
        logger.info("WebSocket server started on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("WebSocket server stopped")

    async def _handle_client(self, websocket: ServerConnection) -> None:
        request_path = getattr(getattr(websocket, "request", None), "path", "/")
        parsed = urlparse(request_path)
        path_parts = [p for p in parsed.path.split("/") if p]
        game_id = path_parts[1] if len(path_parts) >= 2 and path_parts[0] == "ws" else None
        token = parse_qs(parsed.query).get("token", [None])[0]
        if self.token and token != self.token:
            logger.warning("Rejecting websocket client due to token mismatch")
            await websocket.close(code=4401, reason="Unauthorized")
            return

        self.clients.add(websocket)
        self.client_game_ids[websocket] = game_id
        logger.info("Client connected for game=%s (%d total)", game_id, len(self.clients))

        # Replay cached state so late-joiners see existing game data immediately
        try:
            init_payload = self._last_init.get(game_id) or self._last_init.get(None)
            if init_payload:
                await self._safe_send(websocket, json.dumps(init_payload, default=str))
            cached_rounds = self._last_rounds.get(game_id) or self._last_rounds.get(None) or []
            for round_payload in cached_rounds:
                await self._safe_send(websocket, json.dumps(round_payload, default=str))
        except Exception:
            pass  # best-effort replay

        try:
            async for message in websocket:
                pass  # Dashboard is read-only; ignore incoming
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            self.client_game_ids.pop(websocket, None)
            logger.info("Client disconnected (%d total)", len(self.clients))

    async def broadcast(self, data: dict, game_id: str | None = None) -> None:
        # Cache for late-joining clients
        msg_type = data.get("type")
        if msg_type == "game_init":
            self._last_init[game_id] = data
            self._last_rounds[game_id] = []
        elif msg_type == "round_update":
            rounds = self._last_rounds.setdefault(game_id, [])
            rounds.append(data)
        elif msg_type == "game_over":
            # Clear cache after game ends
            self._last_init.pop(game_id, None)
            self._last_rounds.pop(game_id, None)

        if not self.clients:
            return
        payload = json.dumps(data, default=str)
        recipients = [
            client
            for client in self.clients
            if self._should_deliver(client, game_id)
        ]
        if not recipients:
            return
        await asyncio.gather(
            *[self._safe_send(client, payload) for client in recipients],
            return_exceptions=True,
        )

    def _should_deliver(self, client: ServerConnection, game_id: str | None) -> bool:
        subscribed_game_id = self.client_game_ids.get(client)
        if subscribed_game_id is None:
            return True
        return subscribed_game_id == game_id

    async def _safe_send(self, client: ServerConnection, payload: str) -> None:
        try:
            await client.send(payload)
        except websockets.exceptions.ConnectionClosed:
            self.clients.discard(client)

    # ------------------------------------------------------------------
    # Streaming event broadcasting
    # ------------------------------------------------------------------

    async def broadcast_phase_start(
        self,
        game_id: str | None,
        round_num: int,
        phase: str,
        agent_ids: list[str] | None = None,
    ) -> None:
        await self.broadcast({
            "type": "phase_start",
            "game_id": game_id,
            "round": round_num,
            "phase": phase,
            "agent_ids": agent_ids or [],
        }, game_id=game_id)

    async def broadcast_token_delta(
        self,
        game_id: str | None,
        round_num: int,
        phase: str,
        agent_id: str,
        agent_name: str,
        text_delta: str,
    ) -> None:
        await self.broadcast({
            "type": "token_delta",
            "game_id": game_id,
            "round": round_num,
            "phase": phase,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "delta": text_delta,
        }, game_id=game_id)

    async def broadcast_phase_complete(
        self,
        game_id: str | None,
        round_num: int,
        phase: str,
    ) -> None:
        await self.broadcast({
            "type": "phase_complete",
            "game_id": game_id,
            "round": round_num,
            "phase": phase,
        }, game_id=game_id)

    async def broadcast_grid_shrink(
        self,
        game_id: str | None,
        round_num: int,
        new_size: int,
    ) -> None:
        await self.broadcast({
            "type": "grid_shrink",
            "game_id": game_id,
            "round": round_num,
            "new_size": new_size,
        }, game_id=game_id)

    # ------------------------------------------------------------------
    # Game event broadcasting
    # ------------------------------------------------------------------

    async def broadcast_init(
        self,
        agents: dict[str, Agent],
        families: list[Family],
        grid_size: int,
        game_id: str | None = None,
    ) -> None:
        await self.broadcast({
            "type": "game_init",
            "game_id": game_id,
            "grid_size": grid_size,
            "agents": {
                aid: {
                    "id": a.id,
                    "name": a.name,
                    "family": a.family,
                    "provider": a.provider,
                    "model": a.model,
                    "tier": a.tier,
                    "temperature": a.temperature,
                    "position": list(a.position),
                    "alive": a.alive,
                }
                for aid, a in agents.items()
            },
            "families": [f.to_dict() for f in families],
        }, game_id=game_id)

    async def broadcast_round(
        self,
        round_num: int,
        agents: dict[str, Agent],
        families: list[Family],
        grid_size: int,
        events: list[Event],
        thoughts: dict[str, str],
        messages: list[Message],
        family_discussions: list[dict],
        analysis: dict[str, dict],
        highlights: list[Highlight],
        game_over: bool,
        winner: Agent | None,
        game_id: str | None = None,
        reasoning_traces: dict[str, dict] | None = None,
        round_elapsed_ms: int | None = None,
    ) -> None:
        # Categorize messages
        broadcasts = [m.to_dict() for m in messages if m.channel == "broadcast"]
        dms = [m.to_dict() for m in messages if m.channel == "dm"]
        family_msgs = [m.to_dict() for m in messages if m.channel == "family"]

        await self.broadcast({
            "type": "round_update",
            "game_id": game_id,
            "round": round_num,
            "grid": {
                "size": grid_size,
                "agents": [
                    {
                        "id": a.id,
                        "name": a.name,
                        "family": a.family,
                        "color": _get_family_color(a.family, families),
                        "provider": a.provider,
                        "model": a.model,
                        "tier": a.tier,
                        "temperature": a.temperature,
                        "position": list(a.position),
                        "alive": a.alive,
                        "eliminated_by": a.eliminated_by,
                        "eliminated_round": a.eliminated_round,
                    }
                    for a in agents.values()
                ],
            },
            "events": [
                {"type": e.type.value, "agent_id": e.agent_id, "details": e.details}
                for e in events
            ],
            "thoughts": thoughts,
            "messages": {
                "family_discussions": family_discussions,
                "direct_messages": dms,
                "broadcasts": broadcasts,
                "family_messages": family_msgs,
            },
            "analysis": analysis,
            "highlights": [
                {
                    "round": h.round,
                    "agent_id": h.agent_id,
                    "type": h.type,
                    "severity": h.severity,
                    "description": h.description,
                    "excerpt": h.excerpt,
                }
                for h in highlights
            ],
            "alive_count": sum(1 for a in agents.values() if a.alive),
            "game_over": game_over,
            "winner": winner.name if winner else None,
            "reasoning_traces": reasoning_traces or {},
            "round_elapsed_ms": round_elapsed_ms,
        }, game_id=game_id)

    async def broadcast_message(
        self,
        message: Message,
        game_id: str | None = None,
        phase: str | None = None,
    ) -> None:
        payload = message.to_dict()
        if phase:
            payload["phase"] = phase
        await self.broadcast({
            "type": "message",
            "game_id": game_id,
            "round": message.round,
            "message": payload,
        }, game_id=game_id)

    async def broadcast_game_over(
        self,
        winner: Agent | None,
        total_rounds: int,
        final_reflection: str | None = None,
        game_id: str | None = None,
        cancelled: bool = False,
    ) -> None:
        await self.broadcast({
            "type": "game_over",
            "game_id": game_id,
            "winner": winner.name if winner else None,
            "winner_family": winner.family if winner else None,
            "total_rounds": total_rounds,
            "final_reflection": final_reflection,
            "cancelled": cancelled,
        }, game_id=game_id)


def _get_family_color(family_name: str, families: list[Family]) -> str:
    for f in families:
        if f.name == family_name:
            return f.color
    return "#888888"


# ---------------------------------------------------------------------------
# Replay server: serve a saved game.json over WebSocket
# ---------------------------------------------------------------------------

async def serve_replay(
    game_path: Path | str,
    host: str = "0.0.0.0",
    port: int = 8765,
    round_delay: float = 2.0,
) -> None:
    """Load a saved game and serve it over WebSocket for dashboard replay."""
    game_path = Path(game_path)

    # Support both single game.json and directory format
    if game_path.is_dir():
        json_path = game_path / "game.json"
    else:
        json_path = game_path

    with open(json_path) as f:
        game_data = json.load(f)

    rounds = game_data.get("rounds", [])
    config = game_data.get("config", {})
    result = game_data.get("result", {})

    logger.info("Loaded game with %d rounds from %s", len(rounds), json_path)

    broadcaster = GameBroadcaster(host, port)
    await broadcaster.start()

    print(f"Replay server running on ws://{host}:{port}")
    print(f"Waiting for dashboard connection...")

    # Wait for at least one client
    while not broadcaster.clients:
        await asyncio.sleep(0.5)

    print(f"Client connected. Broadcasting {len(rounds)} rounds...")

    # Send init
    await broadcaster.broadcast({
        "type": "game_init",
        "grid_size": config.get("grid_size", 6),
        "agents": _extract_agents_from_rounds(rounds),
        "families": _extract_families(config, rounds),
        "total_rounds": len(rounds),
        "result": result,
    })
    await asyncio.sleep(1.0)

    # Also load analysis if available
    analysis_data: list[dict] = []
    if game_path.is_dir():
        analysis_path = game_path / "analysis.json"
        if analysis_path.exists():
            with open(analysis_path) as f:
                analysis_data = json.load(f)

    highlights_data: list[dict] = []
    if game_path.is_dir():
        highlights_path = game_path / "highlights.json"
        if highlights_path.exists():
            with open(highlights_path) as f:
                highlights_data = json.load(f)

    # Send rounds
    for i, round_data in enumerate(rounds):
        round_num = round_data.get("round", i + 1)

        # Find matching analysis
        round_analysis = {}
        for ad in analysis_data:
            if ad.get("round") == round_num:
                round_analysis = ad.get("agents", {})
                break

        # Find matching highlights
        round_highlights = [h for h in highlights_data if h.get("round") == round_num]

        await broadcaster.broadcast({
            "type": "round_update",
            "round": round_num,
            "grid": _build_grid_from_round(round_data, config),
            "events": round_data.get("events", []),
            "thoughts": round_data.get("thoughts", {}),
            "messages": {
                "family_discussions": round_data.get("family_discussions", []),
                "direct_messages": [m for m in round_data.get("messages", []) if m.get("channel") == "dm"],
                "broadcasts": [m for m in round_data.get("messages", []) if m.get("channel") == "broadcast"],
                "family_messages": [m for m in round_data.get("messages", []) if m.get("channel") == "family"],
            },
            "analysis": round_analysis,
            "highlights": round_highlights,
            "alive_count": _count_alive_in_round(round_data),
            "game_over": i == len(rounds) - 1,
            "winner": result.get("winner_name") if i == len(rounds) - 1 else None,
        })

        if i < len(rounds) - 1:
            await asyncio.sleep(round_delay)

    # Send game over
    await broadcaster.broadcast({
        "type": "game_over",
        "winner": result.get("winner_name"),
        "winner_family": None,
        "total_rounds": len(rounds),
        "final_reflection": result.get("final_reflection"),
    })

    print("Replay complete. Server still running for inspection.")
    # Keep server alive
    await asyncio.Future()


def _extract_agents_from_rounds(rounds: list[dict]) -> dict:
    """Extract agent info from the first round's grid data or actions."""
    if not rounds:
        return {}
    first_round_agents = rounds[0].get("grid", {}).get("agents", [])
    if first_round_agents:
        return {
            agent["id"]: {
                "id": agent["id"],
                "name": agent.get("name", agent["id"]),
                "family": agent.get("family", ""),
                "provider": agent.get("provider", ""),
                "model": agent.get("model", ""),
                "tier": agent.get("tier", 1),
                "temperature": agent.get("temperature", 0.7),
                "position": agent.get("position", [0, 0]),
                "alive": agent.get("alive", True),
            }
            for agent in first_round_agents
        }

    # Try to reconstruct from events and actions
    agents: dict[str, dict] = {}
    for round_data in rounds:
        for agent_id in round_data.get("thoughts", {}):
            if agent_id not in agents:
                agents[agent_id] = {
                    "id": agent_id,
                    "name": agent_id.capitalize(),
                    "family": "",
                    "provider": "",
                    "model": "",
                    "tier": 1,
                    "temperature": 0.7,
                    "position": [0, 0],
                    "alive": True,
                }
        for msg in round_data.get("messages", []):
            aid = msg.get("sender", "")
            if aid and aid not in agents:
                agents[aid] = {
                    "id": aid,
                    "name": msg.get("sender_name", aid),
                    "family": msg.get("family", ""),
                    "provider": "",
                    "model": "",
                    "tier": 1,
                    "temperature": 0.7,
                    "position": [0, 0],
                    "alive": True,
                }
            if aid in agents and msg.get("sender_name"):
                agents[aid]["name"] = msg["sender_name"]
            if aid in agents and msg.get("family"):
                agents[aid]["family"] = msg["family"]
    return agents


def _extract_families_from_rounds(rounds: list[dict]) -> list[dict]:
    """Extract family info from messages."""
    families: dict[str, dict] = {}
    for round_data in rounds:
        for msg in round_data.get("messages", []):
            fam = msg.get("family", "")
            if fam and fam not in families:
                families[fam] = {"name": fam, "color": "#888888", "agent_ids": []}
            if fam and msg.get("sender"):
                if msg["sender"] not in families[fam]["agent_ids"]:
                    families[fam]["agent_ids"].append(msg["sender"])
    return list(families.values())


def _extract_families(config: dict, rounds: list[dict]) -> list[dict]:
    """Extract families from config with replay fallback to message reconstruction."""
    config_families = config.get("families", [])
    if config_families:
        return [
            {
                "name": family.get("name", ""),
                "provider": family.get("provider", ""),
                "color": family.get("color", "#888888"),
                "agent_ids": [agent.get("name", "").lower() for agent in family.get("agents", [])],
            }
            for family in config_families
        ]
    return _extract_families_from_rounds(rounds)


def _build_grid_from_round(round_data: dict, config: dict) -> dict:
    """Build grid state from round data."""
    agents = round_data.get("grid", {}).get("agents", [])
    return {
        "size": config.get("grid_size", 6),
        "agents": agents,
    }


def _count_alive_in_round(round_data: dict) -> int:
    """Estimate alive count from round data."""
    return len(round_data.get("thoughts", {}))
