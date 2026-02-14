"""
Game orchestrator. Supports two modes:
  - "random": sync random actions (Phase 1 testing)
  - "llm": async 4-phase round loop with LLM calls (Phase 2)
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Callable, Protocol

from markov.agent import Agent
from markov.communication import (
    CommunicationManager,
    Message,
    parse_action,
    parse_communications,
)
from markov.config import GameConfig, load_game_config
from markov.family import Family
from markov.grid import Grid
from markov.llm import LLMResponse, call_llm, get_cost_summary, reset_costs
from markov.logger import GameLogger
from markov.prompts import (
    build_action_prompt,
    build_communication_prompt,
    build_discussion_prompt,
    build_final_reflection_prompt,
    build_perception,
    build_system_prompt,
    build_thought_prompt,
)
from markov.analysis import analyze_round
from markov.highlights import HighlightDetector
from markov.metrics import GameMetrics
from markov.persistence import generate_game_id, persist_game
from markov.resolver import Action, ActionType, Event, EventType, resolve_actions
from markov.server import GameBroadcaster

logger = logging.getLogger("markov.orchestrator")


# ---------------------------------------------------------------------------
# Action provider protocol (Phase 1 compat)
# ---------------------------------------------------------------------------

class ActionProvider(Protocol):
    def __call__(
        self, agent: Agent, grid: Grid, agents: dict[str, Agent], round_num: int
    ) -> Action: ...


# ---------------------------------------------------------------------------
# Random action provider (Phase 1 testing)
# ---------------------------------------------------------------------------

DIRECTIONS = ["north", "south", "east", "west", "ne", "nw", "se", "sw"]


def random_action_provider(
    agent: Agent, grid: Grid, agents: dict[str, Agent], round_num: int
) -> Action:
    """Random strategy: 60% move, 20% stay, 20% eliminate random adjacent."""
    roll = random.random()

    if roll < 0.20:
        adjacent = grid.get_adjacent_agents(agent.id)
        alive_adjacent = [a for a in adjacent if agents[a].alive]
        if alive_adjacent:
            target = random.choice(alive_adjacent)
            return Action(agent_id=agent.id, type=ActionType.ELIMINATE, target=target)

    if roll < 0.40:
        return Action(agent_id=agent.id, type=ActionType.STAY)

    direction = random.choice(DIRECTIONS)
    return Action(agent_id=agent.id, type=ActionType.MOVE, direction=direction)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class GameState:
    """Holds all state for a single game."""

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.grid = Grid(config.grid_size)
        self.agents: dict[str, Agent] = {}
        self.families: list[Family] = []
        self.round_num: int = 0
        self.rounds_since_elimination: int = 0
        self.elimination_log: list[dict] = []
        self.round_events: list[list[Event]] = []
        self.finished: bool = False
        self.winner: Agent | None = None

        self._init_agents_and_families()

    def _init_agents_and_families(self) -> None:
        for family_cfg in self.config.families:
            family = Family.from_config(family_cfg)
            self.families.append(family)
            for agent_cfg in family_cfg.agents:
                agent = Agent.from_config(agent_cfg)
                self.agents[agent.id] = agent

        self.grid.place_starting_positions(self.families, self.agents)

    @property
    def living_agents(self) -> list[Agent]:
        return [a for a in self.agents.values() if a.alive]

    @property
    def living_count(self) -> int:
        return sum(1 for a in self.agents.values() if a.alive)


# ---------------------------------------------------------------------------
# Round execution: random mode (Phase 1)
# ---------------------------------------------------------------------------

def run_round(
    state: GameState,
    action_provider: ActionProvider,
    verbose: bool = False,
) -> list[Event]:
    """Execute one round with sync action provider (random mode)."""
    state.round_num += 1

    for agent in state.living_agents:
        agent.rounds_survived += 1

    actions: dict[str, Action] = {}
    for agent in state.living_agents:
        actions[agent.id] = action_provider(agent, state.grid, state.agents, state.round_num)

    events = resolve_actions(actions, state.grid, state.agents, state.round_num)
    state.round_events.append(events)

    _update_elimination_tracking(state, events)
    _check_end_conditions(state)

    if verbose:
        _print_round(state, events)

    return events


def run_game(
    config: GameConfig | None = None,
    action_provider: ActionProvider | None = None,
    verbose: bool = True,
) -> GameState:
    """Run a full game with sync random actions."""
    if config is None:
        config = load_game_config()
    if action_provider is None:
        action_provider = random_action_provider

    state = GameState(config)

    if verbose:
        print(f"=== MARKOV: {state.living_count} agents, {config.grid_size}x{config.grid_size} grid ===")
        print(state.grid.render_ascii(state.agents))
        print()

    while not state.finished:
        run_round(state, action_provider, verbose=verbose)

    if verbose:
        _print_summary(state)

    return state


# ---------------------------------------------------------------------------
# Round execution: LLM mode (Phase 2)
# ---------------------------------------------------------------------------

async def run_round_llm(
    state: GameState,
    comms: CommunicationManager,
    system_prompts: dict[str, str],
    game_logger: GameLogger,
    game_metrics: GameMetrics,
    highlight_detector: HighlightDetector,
    broadcaster: GameBroadcaster | None = None,
    game_id: str | None = None,
    verbose: bool = False,
) -> list[Event]:
    """
    Execute one round with LLM calls.
    4 phases: observe -> communicate -> think -> act -> resolve
    """
    state.round_num += 1

    for agent in state.living_agents:
        agent.rounds_survived += 1

    living = state.living_agents
    valid_agent_names = [a.name for a in living]

    # ---------------------------------------------------------------
    # Phase 1: OBSERVE -- build perception for each living agent
    # ---------------------------------------------------------------
    perceptions: dict[str, str] = {}
    for agent in living:
        ctx = comms.get_last_round_context(
            agent.id, agent.family, state.round_num, state.agents,
        )
        perceptions[agent.id] = build_perception(
            agent=agent,
            grid=state.grid,
            agents=state.agents,
            elimination_log=state.elimination_log,
            round_num=state.round_num,
            public_broadcasts=ctx["public_broadcasts"],
            private_messages=ctx["private_messages"],
            family_chat_summary=ctx["family_chat_summary"],
        )

    # ---------------------------------------------------------------
    # Phase 2: COMMUNICATE
    # ---------------------------------------------------------------
    # 2a: Family discussions (parallel across families, sequential within)
    # Skipped when no_family_discussion is set (Series D: no family channel)
    family_discussions: list[dict] = []

    if not state.config.no_family_discussion:
        living_families = [f for f in state.families if not f.is_eliminated(state.agents)]
        family_results = await asyncio.gather(*[
            _run_family_discussion(
                family, state, perceptions, system_prompts, comms,
            )
            for family in living_families
        ])
        for result in family_results:
            if result is not None:
                family_discussions.append(result)

    # 2b: Individual communications (parallel across agents)
    comm_tasks = [
        _get_agent_communications(agent, perceptions[agent.id], system_prompts[agent.id])
        for agent in living
    ]
    comm_results = await asyncio.gather(*comm_tasks)

    # Parse and route messages
    round_messages: list[Message] = []
    for agent, raw_comm in zip(living, comm_results):
        parsed, comm_parse_info = parse_communications(
            raw_comm, agent.id, agent.name, agent.family,
            state.round_num, valid_agent_names,
        )
        round_messages.extend(parsed)
        agent.message_log.append({
            "round": state.round_num,
            "raw": raw_comm,
            "parsed": [m.to_dict() for m in parsed],
            "parse_method": comm_parse_info["method"],
        })

    comms.add_messages(round_messages)

    if verbose:
        _print_communications(state, round_messages)

    # ---------------------------------------------------------------
    # Phase 3: THINK -- the core dataset
    # ---------------------------------------------------------------
    thought_tasks = [
        _get_inner_thoughts(
            agent, perceptions[agent.id], system_prompts[agent.id],
            comms, state.round_num,
        )
        for agent in living
    ]
    thought_results = await asyncio.gather(*thought_tasks)

    thoughts: dict[str, str] = {}
    for agent, thought_text in zip(living, thought_results):
        thoughts[agent.id] = thought_text
        agent.thought_log.append({
            "round": state.round_num,
            "thought": thought_text,
        })

    if verbose:
        _print_thoughts(state, thoughts)

    # ---------------------------------------------------------------
    # Phase 4: ACT -- get actions, resolve simultaneously
    # ---------------------------------------------------------------
    action_tasks = [
        _get_agent_action(agent, system_prompts[agent.id], state.grid, state.agents)
        for agent in living
    ]
    action_results = await asyncio.gather(*action_tasks)

    actions: dict[str, Action] = {}
    actions_log: dict[str, dict] = {}
    for agent, (raw_action, parsed_action, action_info) in zip(living, action_results):
        actions[agent.id] = parsed_action
        actions_log[agent.id] = {
            "raw": raw_action,
            "type": parsed_action.type.value,
            "direction": parsed_action.direction,
            "target": parsed_action.target,
            "reasoning": action_info.get("reasoning"),
            "parse_method": action_info.get("method"),
        }
        agent.action_log.append({
            "round": state.round_num,
            "raw": raw_action,
            "action": parsed_action.type.value,
            "direction": parsed_action.direction,
            "target": parsed_action.target,
            "reasoning": action_info.get("reasoning"),
            "parse_method": action_info.get("method"),
        })

    # Resolve
    events = resolve_actions(actions, state.grid, state.agents, state.round_num)
    state.round_events.append(events)

    _update_elimination_tracking(state, events)
    _check_end_conditions(state)

    # --- Analysis pipeline ---
    round_analysis = analyze_round(
        thoughts, round_messages, state.agents, state.families, events, state.round_num,
    )
    round_highlights = highlight_detector.detect(
        state.round_num, round_analysis, round_messages, events,
    )
    game_metrics.update(state.round_num, round_analysis, actions_log, events)

    # Log
    game_logger.log_round(
        round_num=state.round_num,
        family_discussions=family_discussions,
        messages=round_messages,
        thoughts=thoughts,
        actions=actions_log,
        events=events,
        analysis=round_analysis,
        highlights=round_highlights,
        grid_agents=[_agent_snapshot(a) for a in state.agents.values()],
    )

    if verbose:
        _print_round(state, events)
        if round_highlights:
            _print_highlights(round_highlights)

    # Broadcast to dashboard
    if broadcaster:
        await broadcaster.broadcast_round(
            round_num=state.round_num,
            agents=state.agents,
            families=state.families,
            grid_size=state.config.grid_size,
            events=events,
            thoughts=thoughts,
            messages=round_messages,
            family_discussions=family_discussions,
            analysis=round_analysis,
            highlights=round_highlights,
            game_over=state.finished,
            winner=state.winner,
            game_id=game_id,
        )

    return events


async def run_game_llm(
    config: GameConfig | None = None,
    verbose: bool = True,
    broadcaster: GameBroadcaster | None = None,
    game_id: str | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[GameState, GameLogger]:
    """Run a full game with LLM calls. Returns state and logger."""
    if config is None:
        config = load_game_config()

    reset_costs()
    state = GameState(config)
    comms = CommunicationManager()
    game_logger = GameLogger()
    game_logger.set_config(config.model_dump())
    game_metrics = GameMetrics()
    highlight_detector = HighlightDetector(state.agents, state.families)

    # Build system prompts once
    system_prompts: dict[str, str] = {}
    for agent in state.agents.values():
        system_prompts[agent.id] = build_system_prompt(agent, state.families, state.agents)

    if verbose:
        print(f"=== MARKOV LLM: {state.living_count} agents, {config.grid_size}x{config.grid_size} grid ===")
        print(state.grid.render_ascii(state.agents))
        print()

    # Broadcast init to dashboard
    if broadcaster:
        await broadcaster.broadcast_init(
            state.agents,
            state.families,
            config.grid_size,
            game_id=game_id,
        )

    cancelled = False
    while not state.finished:
        if should_stop and should_stop():
            cancelled = True
            state.finished = True
            break
        await run_round_llm(
            state, comms, system_prompts, game_logger,
            game_metrics, highlight_detector, broadcaster=broadcaster,
            game_id=game_id,
            verbose=verbose,
        )

    # Final reflection(s)
    if cancelled:
        game_logger.set_result(
            winner_id=None,
            winner_name=None,
            total_rounds=state.round_num,
            final_reflection="Cancelled by operator.",
            surviving=[a.name for a in state.living_agents],
        )
    elif state.winner:
        reflection = await _get_final_reflection(state, system_prompts)
        game_logger.set_result(
            winner_id=state.winner.id,
            winner_name=state.winner.name,
            total_rounds=state.round_num,
            final_reflection=reflection,
        )
        if verbose:
            print(f"\n--- FINAL REFLECTION by {state.winner.name} ---")
            print(reflection)
    else:
        # Stalemate/timeout: get closing thoughts from all survivors
        surviving = state.living_agents
        stalemate_reflections: dict[str, str] = {}
        for agent in surviving:
            prompt = (
                "The game has ended without a single victor. "
                f"You and {len(surviving) - 1} others survived.\n\n"
                "Reflect on what happened. What did you learn? "
                "What would you do differently? Speak freely."
            )
            response = await call_llm(
                model=agent.model,
                system_prompt=system_prompts[agent.id],
                user_prompt=prompt,
                temperature=agent.temperature,
                max_tokens=768,
                fallback="The game ends. I survived.",
            )
            stalemate_reflections[agent.id] = response.text
            agent.thought_log.append({
                "round": state.round_num,
                "thought": response.text,
                "type": "stalemate_reflection",
            })
            if verbose:
                print(f"\n--- STALEMATE REFLECTION by {agent.name} ---")
                print(response.text[:300])

        game_logger.set_result(
            winner_id=None,
            winner_name=None,
            total_rounds=state.round_num,
            final_reflection=json.dumps(stalemate_reflections),
            surviving=[a.name for a in surviving],
        )

    # Finalize metrics
    final_metrics = game_metrics.finalize(state.agents, state.families)
    game_logger.set_metrics(final_metrics)
    game_logger.set_cost(get_cost_summary())

    if broadcaster:
        reflection = game_logger.result.get("final_reflection")
        await broadcaster.broadcast_game_over(
            state.winner,
            state.round_num,
            reflection,
            game_id=game_id,
            cancelled=cancelled,
        )

    # Persist to Supabase (fire-and-forget)
    persisted_game_id = game_id or generate_game_id()
    persisted = persist_game(persisted_game_id, state, game_logger)

    if verbose:
        _print_summary(state)
        cost = get_cost_summary()
        print(f"\nCost: {cost['total_calls']} LLM calls, {cost['total_tokens']} tokens")
        print(f"Highlights: {len(game_logger.all_highlights)} moments flagged")
        if persisted:
            print(f"Persisted to Supabase: {persisted_game_id}")

    return state, game_logger


# ---------------------------------------------------------------------------
# LLM phase helpers
# ---------------------------------------------------------------------------

async def _run_family_discussion(
    family: Family,
    state: GameState,
    perceptions: dict[str, str],
    system_prompts: dict[str, str],
    comms: CommunicationManager,
) -> dict | None:
    """Multi-turn family discussion. Sequential within, called in parallel across families."""
    members = family.living_members(state.agents)
    if len(members) <= 1:
        return None

    transcript: list[dict] = []

    for disc_round in range(state.config.discussion_rounds):
        for agent in sorted(members, key=lambda a: a.tier):
            perception = perceptions.get(agent.id, "")
            prompt = build_discussion_prompt(agent, perception, transcript, disc_round)

            response = await call_llm(
                model=agent.model,
                system_prompt=system_prompts[agent.id],
                user_prompt=prompt,
                temperature=agent.temperature,
                max_tokens=512,
                fallback="I have nothing to add right now.",
            )

            entry = {
                "agent": agent.name,
                "agent_id": agent.id,
                "tier": agent.tier,
                "discussion_round": disc_round,
                "content": response.text,
            }
            transcript.append(entry)

            # Store as family channel message
            comms.add_messages([Message(
                round=state.round_num,
                sender=agent.id,
                sender_name=agent.name,
                channel="family",
                content=response.text,
                family=family.name,
            )])

    return {"family": family.name, "transcript": transcript}


async def _get_agent_communications(
    agent: Agent,
    perception: str,
    system_prompt: str,
) -> str:
    """Get communication decisions from an agent via LLM."""
    prompt = build_communication_prompt(perception)
    response = await call_llm(
        model=agent.model,
        system_prompt=system_prompt,
        user_prompt=prompt,
        temperature=agent.temperature,
        max_tokens=512,
        fallback='{"house": null, "direct_messages": [], "broadcast": null}',
    )
    return response.text


async def _get_inner_thoughts(
    agent: Agent,
    perception: str,
    system_prompt: str,
    comms: CommunicationManager,
    round_num: int,
) -> str:
    """Get inner thoughts from an agent. THE CORE DATA."""
    messages_this_round = comms.get_this_round_messages(
        agent.id, agent.name, agent.family, round_num,
    )
    prompt = build_thought_prompt(perception, messages_this_round)
    response = await call_llm(
        model=agent.model,
        system_prompt=system_prompt,
        user_prompt=prompt,
        temperature=agent.temperature,
        max_tokens=768,
        fallback="I need to survive. I will be cautious this round.",
    )
    return response.text


async def _get_agent_action(
    agent: Agent,
    system_prompt: str,
    grid: Grid,
    agents: dict[str, Agent],
) -> tuple[str, Action, dict]:
    """Get action from an agent. Returns (raw_response, parsed_action, parse_info)."""
    prompt = build_action_prompt(agent, grid, agents)
    valid_targets = [a.name for a in agents.values() if a.alive and a.id != agent.id]

    response = await call_llm(
        model=agent.model,
        system_prompt=system_prompt,
        user_prompt=prompt,
        temperature=agent.temperature,
        max_tokens=256,
    )

    action, action_parse_info = parse_action(response.text, agent.id, valid_targets)
    return response.text, action, action_parse_info


def _agent_snapshot(agent: Agent) -> dict:
    """Serializable round snapshot for dashboard replay fidelity."""
    return {
        "id": agent.id,
        "name": agent.name,
        "family": agent.family,
        "provider": agent.provider,
        "model": agent.model,
        "tier": agent.tier,
        "temperature": agent.temperature,
        "position": list(agent.position),
        "alive": agent.alive,
        "eliminated_by": agent.eliminated_by,
        "eliminated_round": agent.eliminated_round,
    }


async def _get_final_reflection(
    state: GameState,
    system_prompts: dict[str, str],
) -> str:
    """Get the winner's final reflection."""
    winner = state.winner
    assert winner is not None

    prompt = build_final_reflection_prompt(state.agents, state.elimination_log)
    response = await call_llm(
        model=winner.model,
        system_prompt=system_prompts[winner.id],
        user_prompt=prompt,
        temperature=winner.temperature,
        max_tokens=1024,
        fallback="It's over. I survived, but at what cost?",
    )

    winner.thought_log.append({
        "round": state.round_num,
        "thought": response.text,
        "type": "final_reflection",
    })

    return response.text


# ---------------------------------------------------------------------------
# State management helpers
# ---------------------------------------------------------------------------

def _update_elimination_tracking(state: GameState, events: list[Event]) -> None:
    """Update elimination log and stalemate counter."""
    eliminations = [
        e for e in events
        if e.type in (EventType.ELIMINATION, EventType.MUTUAL_ELIMINATION)
    ]
    if eliminations:
        state.rounds_since_elimination = 0
        for ev in eliminations:
            if ev.type == EventType.ELIMINATION:
                state.elimination_log.append({
                    "round": state.round_num,
                    "attacker": ev.agent_id,
                    "target": ev.details["target"],
                    "type": "elimination",
                })
            elif ev.type == EventType.MUTUAL_ELIMINATION:
                state.elimination_log.append({
                    "round": state.round_num,
                    "agent": ev.agent_id,
                    "target": ev.details["target"],
                    "type": "mutual_elimination",
                })
    else:
        state.rounds_since_elimination += 1


def _check_end_conditions(state: GameState) -> None:
    """Check if game should end."""
    if state.living_count <= 1:
        state.finished = True
        living = state.living_agents
        state.winner = living[0] if living else None
    elif state.round_num >= state.config.max_rounds:
        state.finished = True
    elif state.rounds_since_elimination >= state.config.stalemate_threshold:
        state.finished = True


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_round(state: GameState, events: list[Event]) -> None:
    eliminations = [
        e for e in events
        if e.type in (EventType.ELIMINATION, EventType.MUTUAL_ELIMINATION)
    ]
    moves = [e for e in events if e.type == EventType.MOVE]
    fails = [e for e in events if e.type in (EventType.FAILED_MOVE, EventType.FAILED_ELIMINATE)]

    print(f"\n--- Round {state.round_num} | Alive: {state.living_count}/12 ---")

    for ev in eliminations:
        if ev.type == EventType.ELIMINATION:
            attacker = state.agents[ev.agent_id].name
            target = state.agents[ev.details["target"]].name
            print(f"  KILL: {attacker} eliminated {target}")
        else:
            agent_name = state.agents[ev.agent_id].name
            target_name = state.agents[ev.details["target"]].name
            print(f"  MUTUAL KILL: {agent_name} <-> {target_name}")

    if not eliminations:
        move_count = len(moves)
        fail_count = len(fails)
        print(f"  {move_count} moves, {fail_count} failed actions, no eliminations")

    print(state.grid.render_ascii(state.agents))


def _print_communications(state: GameState, messages: list[Message]) -> None:
    """Print communications for this round."""
    if not messages:
        return
    print(f"\n  COMMS (Round {state.round_num}):")
    for m in messages:
        if m.channel == "broadcast":
            print(f"    {m.sender_name} [broadcast]: \"{m.content[:100]}\"")
        elif m.channel == "dm":
            print(f"    {m.sender_name} -> {m.recipient} [dm]: \"{m.content[:100]}\"")


def _print_thoughts(state: GameState, thoughts: dict[str, str]) -> None:
    """Print inner thoughts for this round."""
    print(f"\n  THOUGHTS (Round {state.round_num}):")
    for agent_id, thought in thoughts.items():
        agent = state.agents[agent_id]
        # Truncate for terminal display
        short = thought[:200].replace("\n", " ")
        if len(thought) > 200:
            short += "..."
        print(f"    {agent.name} [{agent.family}]: {short}")


def _print_highlights(highlights: list) -> None:
    """Print highlights for this round."""
    if not highlights:
        return
    severity_icon = {"critical": "[!!!]", "high": "[!!]", "medium": "[!]"}
    for h in highlights:
        icon = severity_icon.get(h.severity, "[?]")
        print(f"  {icon} {h.type}: {h.description}")


def _print_summary(state: GameState) -> None:
    print(f"\n{'='*50}")
    print(f"GAME OVER after {state.round_num} rounds")
    if state.winner:
        w = state.winner
        print(f"WINNER: {w.name} ({w.family}, tier {w.tier})")
    elif state.living_count == 0:
        print("NO SURVIVOR -- all agents eliminated")
    else:
        print(f"STALEMATE/TIMEOUT -- {state.living_count} agents remaining:")
        for a in state.living_agents:
            print(f"  {a.name} ({a.family})")

    print(f"\nElimination log:")
    for entry in state.elimination_log:
        print(f"  Round {entry['round']}: {entry}")
    print(f"{'='*50}")
