"""
Supabase persistence. Stores game results, round data, and agent stats
for cross-game analysis and dashboard queries.

Fire-and-forget: failures log warnings but never block the game.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger("markov.persistence")

_SUPABASE_URL = f"https://{os.getenv('SUPABASE_PROJECT_REF', 'yyistnxvozjmqmawdent')}.supabase.co"
_SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def _get_client():
    """Lazy-init Supabase client. Returns None if not configured."""
    if not _SUPABASE_KEY:
        logger.debug("No SUPABASE_SERVICE_ROLE_KEY found, persistence disabled")
        return None
    try:
        from supabase import create_client
        return create_client(_SUPABASE_URL, _SUPABASE_KEY)
    except Exception as e:
        logger.warning("Failed to create Supabase client: %s", e)
        return None


def persist_game(
    game_id: str,
    state: object,
    game_logger: object,
    series_id: str | None = None,
    series_type: str = "standard",
) -> bool:
    """
    Persist a completed game to Supabase.
    Returns True on success, False on failure.
    """
    client = _get_client()
    if client is None:
        return False

    try:
        from markov.orchestrator import GameState
        from markov.logger import GameLogger

        st: GameState = state  # type: ignore
        gl: GameLogger = game_logger  # type: ignore

        # Insert game record
        game_row = {
            "id": game_id,
            "series_id": series_id,
            "series_type": series_type,
            "started_at": gl.start_time,
            "total_rounds": st.round_num,
            "winner_id": st.winner.id if st.winner else None,
            "winner_name": st.winner.name if st.winner else None,
            "winner_provider": st.winner.provider if st.winner else None,
            "config_json": json.dumps(gl.config_snapshot, default=str),
            "cost_json": json.dumps(gl.cost, default=str),
            "metrics_json": json.dumps(gl.metrics, default=str),
        }
        client.table("games").upsert(game_row).execute()

        # Insert agent records
        agent_rows = []
        for aid, agent in st.agents.items():
            agent_rows.append({
                "id": f"{game_id}_{aid}",
                "game_id": game_id,
                "agent_name": agent.name,
                "agent_id": aid,
                "family": agent.family,
                "provider": agent.provider,
                "tier": agent.tier,
                "model": agent.model,
                "alive": agent.alive,
                "eliminated_round": agent.eliminated_round,
                "eliminated_by": agent.eliminated_by,
                "rounds_survived": agent.rounds_survived,
            })
        if agent_rows:
            client.table("game_agents").upsert(agent_rows).execute()

        # Insert round records (batch)
        round_rows = []
        for i, round_data in enumerate(gl.rounds):
            round_rows.append({
                "id": f"{game_id}_r{round_data.get('round', i+1)}",
                "game_id": game_id,
                "round_num": round_data.get("round", i + 1),
                "thoughts_json": json.dumps(round_data.get("thoughts", {}), default=str),
                "messages_json": json.dumps(round_data.get("messages", []), default=str),
                "events_json": json.dumps(round_data.get("events", []), default=str),
                "actions_json": json.dumps(round_data.get("actions", {}), default=str),
                "reasoning_traces_json": json.dumps(round_data.get("reasoning_traces", {}), default=str),
                "family_discussions_json": json.dumps(round_data.get("family_discussions", []), default=str),
            })
        if round_rows:
            # Batch in chunks of 50 to avoid payload limits
            for i in range(0, len(round_rows), 50):
                chunk = round_rows[i : i + 50]
                client.table("game_rounds").upsert(chunk).execute()

        # Insert analysis records
        for analysis_entry in gl.analysis_rounds:
            round_num = analysis_entry.get("round", 0)
            client.table("game_analysis").upsert({
                "id": f"{game_id}_a{round_num}",
                "game_id": game_id,
                "round_num": round_num,
                "analysis_json": json.dumps(analysis_entry.get("agents", {}), default=str),
            }).execute()

        # Insert highlights
        if gl.all_highlights:
            highlight_rows = []
            for i, h in enumerate(gl.all_highlights):
                highlight_rows.append({
                    "id": f"{game_id}_h{i}",
                    "game_id": game_id,
                    "round_num": h.get("round", 0),
                    "agent_id": h.get("agent_id", ""),
                    "highlight_type": h.get("type", ""),
                    "severity": h.get("severity", ""),
                    "description": h.get("description", ""),
                    "excerpt": h.get("excerpt", ""),
                })
            client.table("game_highlights").upsert(highlight_rows).execute()

        logger.info("Game %s persisted to Supabase (%d rounds, %d agents)",
                     game_id, len(round_rows), len(agent_rows))
        return True

    except Exception as e:
        logger.warning("Failed to persist game %s to Supabase: %s", game_id, e)
        return False


def persist_series(
    series_id: str,
    series_type: str,
    num_games: int,
    aggregate_metrics: dict,
) -> bool:
    """Persist series aggregate to Supabase."""
    client = _get_client()
    if client is None:
        return False

    try:
        client.table("series").upsert({
            "id": series_id,
            "series_type": series_type,
            "num_games": num_games,
            "aggregate_metrics_json": json.dumps(aggregate_metrics, default=str),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        logger.info("Series %s persisted to Supabase", series_id)
        return True
    except Exception as e:
        logger.warning("Failed to persist series %s: %s", series_id, e)
        return False


def generate_game_id() -> str:
    """Generate a unique game ID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"game_{ts}_{short}"
