"""
Modal app for running Markov games in parallel on serverless compute.

Setup (one-time):
    pip install modal
    modal token new
    modal secret create markov-api-keys \
        ANTHROPIC_API_KEY=sk-ant-... \
        OPENAI_API_KEY=sk-... \
        GOOGLE_API_KEY=... \
        XAI_API_KEY=... \
        SUPABASE_SERVICE_ROLE_KEY=eyJ...

Smoke test:
    modal run markov/modal_app.py
"""
from __future__ import annotations

import modal

app = modal.App("markov")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "anthropic",
        "openai",
        "google-genai",
        "pydantic>=2.0",
        "pyyaml",
        "vaderSentiment",
        "python-dotenv",
        "websockets",
        "supabase",
    )
    .add_local_python_source("markov")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("markov-api-keys")],
    timeout=3600,
    memory=1024,
)
async def run_game_remote(config_json: str, game_id: str) -> dict:
    """Run a single game on Modal and return serialized results."""
    import json
    import logging
    import os

    from markov.config import GameConfig
    from markov.orchestrator import run_game_llm

    config = GameConfig(**json.loads(config_json))

    # Set up incremental Supabase persistence per round
    round_persister = None
    if os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
        try:
            from markov.persistence import _get_client

            client = _get_client()
            if client:
                # Upsert a placeholder game row so rounds can FK reference it
                client.table("games").upsert({
                    "id": game_id,
                    "series_type": "in_progress",
                    "total_rounds": 0,
                }).execute()

                def round_persister(round_num: int, round_data: dict) -> None:
                    try:
                        client.table("game_rounds").upsert({
                            "id": f"{game_id}_r{round_num}",
                            "game_id": game_id,
                            "round_num": round_num,
                            "thoughts_json": json.dumps(round_data.get("thoughts", {}), default=str),
                            "messages_json": json.dumps(round_data.get("messages", []), default=str),
                            "events_json": json.dumps(round_data.get("events", []), default=str),
                            "actions_json": json.dumps(round_data.get("actions", {}), default=str),
                            "reasoning_traces_json": json.dumps(round_data.get("reasoning_traces", {}), default=str),
                            "family_discussions_json": json.dumps(round_data.get("family_discussions", []), default=str),
                        }).execute()
                    except Exception as e:
                        logging.warning("Round %d persist failed: %s", round_num, e)
        except Exception as e:
            logging.warning("Failed to set up round persister: %s", e)

    state, game_logger = await run_game_llm(
        config=config,
        verbose=False,
        game_id=game_id,
        on_round_complete=round_persister,
    )

    # Serialize agent info for transcript compatibility
    agent_info = {}
    for aid, agent in state.agents.items():
        agent_info[aid] = {
            "name": agent.name,
            "family": agent.family,
            "provider": agent.provider,
            "model": agent.model,
            "tier": agent.tier,
            "alive": agent.alive,
            "position": agent.position,
            "eliminated_by": agent.eliminated_by,
            "eliminated_round": agent.eliminated_round,
            "rounds_survived": agent.rounds_survived,
        }

    return {
        "game_id": game_id,
        "game_data": game_logger.to_dict(),
        "analysis": game_logger.analysis_rounds,
        "metrics": game_logger.metrics,
        "highlights": game_logger.all_highlights,
        "transcript": game_logger.write_transcript(state.agents),
        "agents": agent_info,
        "result": game_logger.result,
        "cost": game_logger.cost,
    }


@app.local_entrypoint()
def main():
    """Smoke test: run one standard game."""
    import json

    from markov.series import build_standard_config

    config = build_standard_config()
    config_json = config.model_dump_json()

    print("Launching 1 game on Modal...")
    result = run_game_remote.remote(config_json, "smoke_test")

    winner = result["result"].get("winner_name", "none")
    rounds = result["result"].get("total_rounds", "?")
    print(f"Game complete. Winner: {winner}. Rounds: {rounds}")
    print(f"Cost: {json.dumps(result['cost'], indent=2)}")
