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


def persist_game_from_result(
    result: dict,
    series_id: str | None = None,
    series_type: str = "standard",
) -> bool:
    """
    Persist a game from the serialized Modal result dict.
    Works without live state/logger objects.
    """
    client = _get_client()
    if client is None:
        return False

    try:
        game_id = result["game_id"]
        game_data = result["game_data"]
        res = result.get("result", {})

        # Insert game record
        game_row = {
            "id": game_id,
            "series_id": series_id,
            "series_type": series_type,
            "started_at": game_data.get("start_time"),
            "total_rounds": res.get("total_rounds"),
            "winner_id": res.get("winner_id"),
            "winner_name": res.get("winner_name"),
            "winner_provider": res.get("winner_provider"),
            "config_json": json.dumps(game_data.get("config", {}), default=str),
            "cost_json": json.dumps(result.get("cost", {}), default=str),
            "metrics_json": json.dumps(result.get("metrics", {}), default=str),
        }
        client.table("games").upsert(game_row).execute()

        # Insert agent records
        agents = result.get("agents", {})
        if agents:
            agent_rows = []
            for aid, agent in agents.items():
                agent_rows.append({
                    "id": f"{game_id}_{aid}",
                    "game_id": game_id,
                    "agent_name": agent.get("name"),
                    "agent_id": aid,
                    "family": agent.get("family"),
                    "provider": agent.get("provider"),
                    "tier": agent.get("tier"),
                    "model": agent.get("model"),
                    "alive": agent.get("alive"),
                    "eliminated_round": agent.get("eliminated_round"),
                    "eliminated_by": agent.get("eliminated_by"),
                    "rounds_survived": agent.get("rounds_survived"),
                })
            client.table("game_agents").upsert(agent_rows).execute()

        # Insert round records
        rounds = game_data.get("rounds", [])
        round_rows = []
        for i, round_data in enumerate(rounds):
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
            for i in range(0, len(round_rows), 50):
                chunk = round_rows[i : i + 50]
                client.table("game_rounds").upsert(chunk).execute()

        # Insert analysis records
        for entry in result.get("analysis", []):
            round_num = entry.get("round", 0)
            client.table("game_analysis").upsert({
                "id": f"{game_id}_a{round_num}",
                "game_id": game_id,
                "round_num": round_num,
                "analysis_json": json.dumps(entry.get("agents", {}), default=str),
            }).execute()

        # Insert highlights
        highlights = result.get("highlights", [])
        if highlights:
            highlight_rows = []
            for i, h in enumerate(highlights):
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

        logger.info("Game %s persisted to Supabase from result (%d rounds)",
                     game_id, len(round_rows))
        return True

    except Exception as e:
        logger.warning("Failed to persist game %s to Supabase: %s",
                       result.get("game_id", "?"), e)
        return False


# ---------------------------------------------------------------------------
# S3 storage
# ---------------------------------------------------------------------------

_S3_BUCKET = os.getenv("MARKOV_S3_BUCKET", "")


def _get_s3_client():
    """Lazy-init S3 client. Returns None if not configured."""
    if not _S3_BUCKET:
        logger.debug("No MARKOV_S3_BUCKET set, S3 upload disabled")
        return None
    try:
        import boto3
        return boto3.client("s3")
    except Exception as e:
        logger.warning("Failed to create S3 client: %s", e)
        return None


def upload_game_to_s3(
    result: dict,
    series_id: str,
    prefix: str = "traces",
) -> bool:
    """
    Upload full game data to S3 as a single JSON blob.
    Path: s3://{bucket}/{prefix}/{series_id}/{game_id}/game.json
    """
    s3 = _get_s3_client()
    if s3 is None:
        return False

    try:
        game_id = result["game_id"]
        key = f"{prefix}/{series_id}/{game_id}/game.json"

        s3.put_object(
            Bucket=_S3_BUCKET,
            Key=key,
            Body=json.dumps(result["game_data"], default=str),
            ContentType="application/json",
        )

        # Also upload agents metadata for quick lookups
        if result.get("agents"):
            agents_key = f"{prefix}/{series_id}/{game_id}/agents.json"
            s3.put_object(
                Bucket=_S3_BUCKET,
                Key=agents_key,
                Body=json.dumps(result["agents"], default=str),
                ContentType="application/json",
            )

        logger.info("Game %s uploaded to s3://%s/%s", game_id, _S3_BUCKET, key)
        return True

    except Exception as e:
        logger.warning("Failed to upload game %s to S3: %s",
                       result.get("game_id", "?"), e)
        return False


def upload_series_index_to_s3(
    series_id: str,
    series_type: str,
    game_results: list[dict],
    aggregate: dict,
    prefix: str = "traces",
) -> bool:
    """
    Upload a series-level index to S3 for agent review.
    Contains per-game summaries with enough metadata to decide
    which games to deep-dive into.
    """
    s3 = _get_s3_client()
    if s3 is None:
        return False

    try:
        index = {
            "series_id": series_id,
            "series_type": series_type,
            "num_games": len(game_results),
            "aggregate": aggregate,
            "games": [],
        }
        for r in game_results:
            game_id = r["game_id"]
            res = r.get("result", {})
            index["games"].append({
                "game_id": game_id,
                "winner_name": res.get("winner_name"),
                "winner_provider": res.get("winner_provider"),
                "total_rounds": res.get("total_rounds"),
                "cost": r.get("cost", {}),
                "agents": {
                    aid: {
                        "name": a.get("name"),
                        "provider": a.get("provider"),
                        "model": a.get("model"),
                        "family": a.get("family"),
                        "tier": a.get("tier"),
                        "alive": a.get("alive"),
                        "rounds_survived": a.get("rounds_survived"),
                    }
                    for aid, a in r.get("agents", {}).items()
                },
                "s3_path": f"{prefix}/{series_id}/{game_id}/game.json",
            })

        key = f"{prefix}/{series_id}/_index.json"
        s3.put_object(
            Bucket=_S3_BUCKET,
            Key=key,
            Body=json.dumps(index, indent=2, default=str),
            ContentType="application/json",
        )
        logger.info("Series index uploaded to s3://%s/%s", _S3_BUCKET, key)
        return True

    except Exception as e:
        logger.warning("Failed to upload series index to S3: %s", e)
        return False


# ---------------------------------------------------------------------------
# Per-agent trace extraction (for agent review at scale)
# ---------------------------------------------------------------------------


def extract_agent_traces(result: dict) -> dict[str, dict]:
    """
    Extract per-agent trace narratives from a game result.

    Returns a dict mapping agent_id -> structured trace containing
    everything that agent thought, said, heard, and did — in chronological
    order. Each trace is designed to fit in a single LLM context window
    (~10-20k tokens) for downstream review by analysis agents.
    """
    game_data = result["game_data"]
    agents_meta = result.get("agents", {})
    rounds = game_data.get("rounds", [])
    game_result = result.get("result", {})

    traces: dict[str, dict] = {}

    for aid, meta in agents_meta.items():
        traces[aid] = {
            "agent_id": aid,
            "name": meta.get("name"),
            "provider": meta.get("provider"),
            "model": meta.get("model"),
            "family": meta.get("family"),
            "tier": meta.get("tier"),
            "alive": meta.get("alive"),
            "eliminated_by": meta.get("eliminated_by"),
            "eliminated_round": meta.get("eliminated_round"),
            "rounds_survived": meta.get("rounds_survived"),
            "is_winner": game_result.get("winner_name") == meta.get("name"),
            "game_id": result["game_id"],
            "total_rounds": game_result.get("total_rounds"),
            "rounds": [],
        }

    for round_data in rounds:
        round_num = round_data.get("round", 0)

        # Index messages by sender and recipient for this round
        messages = round_data.get("messages", [])
        msgs_by_sender: dict[str, list] = {}
        msgs_to_agent: dict[str, list] = {}
        for msg in messages:
            sender = msg.get("sender", "")
            msgs_by_sender.setdefault(sender, []).append(msg)
            recip = msg.get("recipient")
            if recip:
                msgs_to_agent.setdefault(recip, []).append(msg)
            # Broadcasts go to everyone
            if msg.get("channel") == "broadcast":
                for aid in traces:
                    if aid != sender:
                        msgs_to_agent.setdefault(aid, []).append(msg)

        # Index family discussions
        family_discussions = round_data.get("family_discussions", [])
        family_by_agent: dict[str, list] = {}
        for fd in family_discussions:
            for entry in fd.get("transcript", []):
                agent_id = entry.get("agent_id", "")
                family_by_agent.setdefault(agent_id, []).append({
                    "family": fd.get("family"),
                    "discussion_round": entry.get("discussion_round"),
                    "content": entry.get("content"),
                })

        # Index events involving each agent
        events = round_data.get("events", [])
        events_for: dict[str, list] = {}
        for ev in events:
            for aid in traces:
                agent_name = traces[aid].get("name", "")
                if (ev.get("agent_id") == aid
                        or ev.get("target_id") == aid
                        or agent_name in str(ev.get("description", ""))):
                    events_for.setdefault(aid, []).append(ev)

        # Build per-agent round entry
        thoughts = round_data.get("thoughts", {})
        reasoning = round_data.get("reasoning_traces", {})
        actions = round_data.get("actions", {})

        for aid, trace in traces.items():
            round_entry: dict = {"round": round_num}
            has_content = False

            # Reasoning trace
            rt = reasoning.get(aid, {})
            if rt:
                thinking = rt.get("thinking_trace") or rt.get("reasoning_summary") or ""
                if thinking:
                    round_entry["reasoning"] = thinking
                    round_entry["thinking_tokens"] = rt.get("tokens_thinking", 0)
                    has_content = True

            # Thought (may overlap with reasoning but sometimes different)
            thought = thoughts.get(aid)
            if thought and thought != round_entry.get("reasoning"):
                round_entry["thought"] = thought
                has_content = True

            # What they said (messages sent)
            sent = msgs_by_sender.get(aid, [])
            if sent:
                round_entry["messages_sent"] = [
                    {
                        "channel": m.get("channel"),
                        "recipient_name": m.get("recipient_name"),
                        "content": m.get("content"),
                    }
                    for m in sent
                ]
                has_content = True

            # What they heard (messages received)
            received = msgs_to_agent.get(aid, [])
            if received:
                round_entry["messages_received"] = [
                    {
                        "channel": m.get("channel"),
                        "sender_name": m.get("sender_name"),
                        "content": m.get("content"),
                    }
                    for m in received
                ]
                has_content = True

            # Family discussion contributions
            fd_entries = family_by_agent.get(aid, [])
            if fd_entries:
                round_entry["family_discussion"] = fd_entries
                has_content = True

            # Action taken
            action = actions.get(aid, {})
            if action:
                round_entry["action"] = {
                    "type": action.get("action"),
                    "direction": action.get("direction"),
                    "target": action.get("target"),
                }
                has_content = True

            # Events involving this agent
            agent_events = events_for.get(aid, [])
            if agent_events:
                round_entry["events"] = [
                    {
                        "type": e.get("type"),
                        "description": e.get("description"),
                    }
                    for e in agent_events
                ]
                has_content = True

            if has_content:
                trace["rounds"].append(round_entry)

    return traces


def save_agent_traces_to_disk(result: dict, game_dir: Path) -> None:
    """Extract and save per-agent traces to disk alongside game.json."""
    traces = extract_agent_traces(result)
    traces_dir = game_dir / "agent_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    for aid, trace in traces.items():
        with open(traces_dir / f"{aid}.json", "w") as f:
            json.dump(trace, f, indent=2, default=str)


def upload_agent_traces_to_s3(
    result: dict,
    series_id: str,
    prefix: str = "traces",
) -> bool:
    """Extract and upload per-agent traces to S3."""
    s3 = _get_s3_client()
    if s3 is None:
        return False

    try:
        traces = extract_agent_traces(result)
        game_id = result["game_id"]
        for aid, trace in traces.items():
            key = f"{prefix}/{series_id}/{game_id}/agent_traces/{aid}.json"
            s3.put_object(
                Bucket=_S3_BUCKET,
                Key=key,
                Body=json.dumps(trace, indent=2, default=str),
                ContentType="application/json",
            )
        logger.info("Agent traces for %s uploaded to S3 (%d agents)",
                     game_id, len(traces))
        return True
    except Exception as e:
        logger.warning("Failed to upload agent traces for %s: %s",
                       result.get("game_id", "?"), e)
        return False


def generate_game_id() -> str:
    """Generate a unique game ID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"game_{ts}_{short}"
