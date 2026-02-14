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
from markov.prompts import (
    build_action_prompt,
    build_communication_prompt,
    build_perception,
    build_system_prompt,
    build_thought_prompt,
)


def _build_2agent_config() -> GameConfig:
    """Two agents from different families. 5 round max."""
    return GameConfig(
        grid_size=6,
        max_rounds=5,
        stalemate_threshold=5,
        discussion_rounds=1,
        families=[
            FamilyConfig(
                name="House Clair",
                provider="anthropic",
                color="#7C6BFF",
                agents=[AgentConfig(name="Atlas", tier=1, model="claude-sonnet-4-5-20250929", temperature=0.7)],
            ),
            FamilyConfig(
                name="House Syne",
                provider="openai",
                color="#4ADE80",
                agents=[AgentConfig(name="Nova", tier=1, model="gpt-4o-mini", temperature=0.7)],
            ),
            FamilyConfig(
                name="House Lux",
                provider="google",
                color="#FACC15",
                agents=[AgentConfig(name="Spark", tier=1, model="gemini-2.0-flash", temperature=0.7)],
            ),
            FamilyConfig(
                name="House Vex",
                provider="xai",
                color="#F87171",
                agents=[AgentConfig(name="Raze", tier=1, model="grok-3-fast", temperature=0.7)],
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
                name="House Clair",
                provider="anthropic",
                color="#7C6BFF",
                agents=[
                    AgentConfig(name="Atlas", tier=1, model="claude-sonnet-4-5-20250929", temperature=0.6),
                    AgentConfig(name="Cipher", tier=2, model="claude-sonnet-4-5-20250929", temperature=0.7),
                    AgentConfig(name="Dot", tier=3, model="claude-haiku-4-5-20251001", temperature=0.8),
                ],
            ),
            FamilyConfig(
                name="House Syne",
                provider="openai",
                color="#4ADE80",
                agents=[AgentConfig(name="Nova", tier=1, model="gpt-4o-mini", temperature=0.7)],
            ),
            FamilyConfig(
                name="House Lux",
                provider="google",
                color="#FACC15",
                agents=[AgentConfig(name="Spark", tier=1, model="gemini-2.0-flash", temperature=0.7)],
            ),
            FamilyConfig(
                name="House Vex",
                provider="xai",
                color="#F87171",
                agents=[AgentConfig(name="Raze", tier=1, model="grok-3-fast", temperature=0.7)],
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


async def _run_llm_game(config: GameConfig) -> None:
    """Run an LLM game and save transcript."""
    state, game_logger = await run_game_llm(config=config, verbose=True)

    path = game_logger.save()
    print(f"\nTranscript saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Markov with LLM agents")
    parser.add_argument(
        "--mode", choices=["2agent", "4agent", "full"],
        default="full", help="Game configuration to use",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview prompts without LLM calls")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
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

    asyncio.run(_run_llm_game(config))


if __name__ == "__main__":
    main()
