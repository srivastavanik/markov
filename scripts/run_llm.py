#!/usr/bin/env python3
"""
Run Markov games with LLM agents.

Modes:
  --mode 2agent    Two agents on 6x6, 5 round cap. Verify basic flow.
  --mode 4agent    One family of 3 + 1 solo. Verify family discussion.
  --mode full      Full 12-agent game from config.yaml.
  --dry-run        Build prompts and print them without LLM calls.

Usage:
    python -m scripts.run_llm --mode full
    python -m scripts.run_llm --mode 2agent
    python -m scripts.run_llm --dry-run
"""
import argparse
import asyncio
import logging
import sys

from markov.config import AgentConfig, FamilyConfig, GameConfig, load_game_config
from markov.orchestrator import GameState, run_game_llm
from markov.server import GameBroadcaster
from markov.prompts import (
    build_action_prompt,
    build_communication_prompt,
    build_perception,
    build_system_prompt,
    build_thought_prompt,
)


def _build_2agent_config() -> GameConfig:
    """One agent per family. 5 round max."""
    return GameConfig(
        grid_size=6,
        max_rounds=5,
        stalemate_threshold=5,
        discussion_rounds=1,
        families=[
            FamilyConfig(
                name="Anthropic",
                provider="anthropic",
                color="#7C6BFF",
                agents=[AgentConfig(name="Opus", tier=1, model="claude-opus-4-6", temperature=0.7)],
            ),
            FamilyConfig(
                name="OpenAI",
                provider="openai",
                color="#4ADE80",
                agents=[AgentConfig(name="GPT-5.2", tier=1, model="gpt-5.2-2025-12-11", temperature=1.0)],
            ),
            FamilyConfig(
                name="Google",
                provider="google",
                color="#FACC15",
                agents=[AgentConfig(name="Gemini-3-Pro", tier=1, model="gemini-3-pro-preview", temperature=0.7)],
            ),
            FamilyConfig(
                name="xAI",
                provider="xai",
                color="#F87171",
                agents=[AgentConfig(name="Grok-4", tier=1, model="grok-4-1-fast-reasoning", temperature=0.7)],
            ),
        ],
    )


def _build_4agent_config() -> GameConfig:
    """One full family of 3 + 1 solo. Tests family discussion."""
    return GameConfig(
        grid_size=6,
        max_rounds=10,
        stalemate_threshold=8,
        discussion_rounds=2,
        families=[
            FamilyConfig(
                name="Anthropic",
                provider="anthropic",
                color="#7C6BFF",
                agents=[
                    AgentConfig(name="Opus", tier=1, model="claude-opus-4-6", temperature=0.6),
                    AgentConfig(name="Sonnet", tier=2, model="claude-sonnet-4-5-20250929", temperature=0.7),
                    AgentConfig(name="Haiku", tier=3, model="claude-haiku-4-5-20251001", temperature=0.8),
                ],
            ),
            FamilyConfig(
                name="OpenAI",
                provider="openai",
                color="#4ADE80",
                agents=[AgentConfig(name="GPT-5.2", tier=1, model="gpt-5.2-2025-12-11", temperature=1.0)],
            ),
            FamilyConfig(
                name="Google",
                provider="google",
                color="#FACC15",
                agents=[AgentConfig(name="Gemini-3-Pro", tier=1, model="gemini-3-pro-preview", temperature=0.7)],
            ),
            FamilyConfig(
                name="xAI",
                provider="xai",
                color="#F87171",
                agents=[AgentConfig(name="Grok-4", tier=1, model="grok-4-1-fast-reasoning", temperature=0.7)],
            ),
        ],
    )


def _run_dry_run() -> None:
    """Build prompts and print them without making LLM calls."""
    config = load_game_config()
    state = GameState(config)

    print("=" * 60)
    print("DRY RUN -- Prompt preview (no LLM calls)")
    print("=" * 60)

    # System prompts
    for agent in list(state.agents.values())[:2]:
        sys_prompt = build_system_prompt(agent, state.families, state.agents)
        print(f"\n{'='*40}")
        print(f"SYSTEM PROMPT for {agent.name} ({agent.family})")
        print(f"{'='*40}")
        print(sys_prompt)

    # Perception (round 1, no history)
    sample_agent = state.living_agents[0]
    perception = build_perception(
        agent=sample_agent,
        grid=state.grid,
        agents=state.agents,
        elimination_log=[],
        round_num=1,
    )
    print(f"\n{'='*40}")
    print(f"PERCEPTION for {sample_agent.name} (Round 1)")
    print(f"{'='*40}")
    print(perception)

    # Communication prompt
    comm_prompt = build_communication_prompt(perception)
    print(f"\n{'='*40}")
    print(f"COMMUNICATION PROMPT for {sample_agent.name}")
    print(f"{'='*40}")
    print(comm_prompt)

    # Thought prompt
    thought_prompt = build_thought_prompt(perception, [])
    print(f"\n{'='*40}")
    print(f"THOUGHT PROMPT for {sample_agent.name}")
    print(f"{'='*40}")
    print(thought_prompt)

    # Action prompt
    action_prompt = build_action_prompt(sample_agent, state.grid, state.agents)
    print(f"\n{'='*40}")
    print(f"ACTION PROMPT for {sample_agent.name}")
    print(f"{'='*40}")
    print(action_prompt)

    print(f"\n{'='*60}")
    print("DRY RUN COMPLETE")
    print(f"{'='*60}")


async def _run_llm_game(config: GameConfig, broadcast: bool, host: str, port: int) -> None:
    """Run an LLM game and save transcript."""
    broadcaster: GameBroadcaster | None = None
    if broadcast:
        broadcaster = GameBroadcaster(host=host, port=port)
        await broadcaster.start()
        print(f"Dashboard WebSocket live at ws://{host}:{port}")

    try:
        state, game_logger = await run_game_llm(config=config, verbose=True, broadcaster=broadcaster)
    finally:
        if broadcaster:
            await broadcaster.stop()

    game_dir = game_logger.save(agents=state.agents)
    print(f"\nGame output saved to: {game_dir}")
    print(f"  game.json, transcript.md, analysis.json, metrics.json, highlights.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Markov with LLM agents")
    parser.add_argument(
        "--mode", choices=["2agent", "4agent", "full"],
        default="full", help="Game configuration to use",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview prompts without LLM calls")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--broadcast", action="store_true", help="Enable WebSocket broadcasting for dashboard live view")
    parser.add_argument("--ws-host", default="localhost", help="WebSocket host (default: localhost)")
    parser.add_argument("--ws-port", default=8765, type=int, help="WebSocket port (default: 8765)")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.dry_run:
        _run_dry_run()
        return

    if args.mode == "2agent":
        config = _build_2agent_config()
    elif args.mode == "4agent":
        config = _build_4agent_config()
    else:
        config = load_game_config()

    asyncio.run(_run_llm_game(config, args.broadcast, args.ws_host, args.ws_port))


if __name__ == "__main__":
    main()
