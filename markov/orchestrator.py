"""
Game orchestrator. Supports two modes:
  - "random": sync random actions (Phase 1 testing)
  - "llm": async 3-phase round loop with LLM calls + extended thinking
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Callable, Protocol

from markov.agent import Agent
from markov.communication import (
    DECISION_JSON_SCHEMA,
    CommunicationManager,
    Message,
    parse_decision_response,
)
from markov.config import GameConfig, load_game_config
from markov.family import Family
from markov.grid import Grid
from markov.llm import (
    ThinkingResponse,
    call_llm,
    call_llm_stream,
    call_llm_with_thinking,
    get_cost_summary,
    reset_costs,
)
from markov.logger import GameLogger
from markov.prompts import (
    build_decision_prompt,
    build_discussion_prompt,
    build_final_reflection_prompt,
    build_perception,
    build_system_prompt,
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
    family_discussions: list[dict] = []

    if not state.config.no_family_discussion:
        living_families = [f for f in state.families if not f.is_eliminated(state.agents)]
        if broadcaster:
            await broadcaster.broadcast_phase_start(game_id, state.round_num, "family_discussion",
                [aid for f in living_families for aid in f.agent_ids if state.agents.get(aid, Agent(id="",name="",family="",provider="",model="",tier=1,temperature=0)).alive])
        family_results = await asyncio.gather(*[
            _run_family_discussion(
                family, state, perceptions, system_prompts, comms,
                broadcaster=broadcaster, game_id=game_id,
            )
            for family in living_families
        ], return_exceptions=True)
        for i, result in enumerate(family_results):
            if isinstance(result, BaseException):
                logger.error("Family discussion crashed for %s: %s",
                             living_families[i].name, result, exc_info=result)
                continue
            if result is not None:
                family_discussions.append(result)
        if broadcaster:
            await broadcaster.broadcast_phase_complete(game_id, state.round_num, "family_discussion")

    # ---------------------------------------------------------------
    # Phase 3: DECIDE -- merged communication + action with
    #          extended thinking (reasoning traces = core dataset)
    # ---------------------------------------------------------------
    if broadcaster:
        await broadcaster.broadcast_phase_start(game_id, state.round_num, "deciding",
            [a.id for a in living])

    valid_targets = [a.name for a in living]
    decide_tasks = [
        _get_agent_decision(
            agent, perceptions[agent.id], system_prompts[agent.id],
            state.grid, state.agents, valid_agent_names, valid_targets,
            broadcaster=broadcaster, game_id=game_id, round_num=state.round_num,
        )
        for agent in living
    ]
    decide_results = await asyncio.gather(*decide_tasks, return_exceptions=True)

    # Process results: extract comms, actions, thinking traces
    round_messages: list[Message] = []
    actions: dict[str, Action] = {}
    actions_log: dict[str, dict] = {}
    thoughts: dict[str, str] = {}
    reasoning_traces: dict[str, dict] = {}

    for agent, result in zip(living, decide_results):
        if isinstance(result, BaseException):
            logger.error("Decision failed for %s: %s", agent.name, result, exc_info=result)
            actions[agent.id] = Action(agent_id=agent.id, type=ActionType.STAY)
            actions_log[agent.id] = {"raw": "", "type": "stay", "direction": None, "target": None, "parse_method": "exception_fallback"}
            thoughts[agent.id] = f"[Decision failed: {result}]"
            continue

        response, messages, action, parse_info = result

        # Communications
        round_messages.extend(messages)
        agent.message_log.append({
            "round": state.round_num,
            "raw": response.text,
            "parsed": [m.to_dict() for m in messages],
            "parse_method": parse_info["method"],
        })

        # Broadcast per-agent communication completion
        if broadcaster and messages:
            for msg in messages:
                await broadcaster.broadcast_token_delta(
                    game_id, state.round_num, "deciding",
                    agent.id, agent.name, msg.content,
                )

        # Actions
        actions[agent.id] = action
        actions_log[agent.id] = {
            "raw": response.text,
            "type": action.type.value,
            "direction": action.direction,
            "target": action.target,
            "parse_method": parse_info.get("method"),
        }
        agent.action_log.append({
            "round": state.round_num,
            "raw": response.text,
            "action": action.type.value,
            "direction": action.direction,
            "target": action.target,
            "parse_method": parse_info.get("method"),
        })

        # Thinking traces — the core dataset
        thinking_text = response.thinking_trace or response.reasoning_summary or ""
        thoughts[agent.id] = thinking_text
        agent.thought_log.append({
            "round": state.round_num,
            "thought": thinking_text,
            "type": "reasoning_trace",
        })
        trace_entry = {
            "thinking_trace": response.thinking_trace,
            "reasoning_summary": response.reasoning_summary,
            "tokens_thinking": response.thinking_tokens,
        }
        reasoning_traces[agent.id] = trace_entry
        agent.reasoning_log.append({
            "round": state.round_num,
            **trace_entry,
        })

    comms.add_messages(round_messages)

    if broadcaster:
        await broadcaster.broadcast_phase_complete(game_id, state.round_num, "deciding")

    if verbose:
        _print_communications(state, round_messages)
        _print_thoughts(state, thoughts)

    # ---------------------------------------------------------------
    # Resolve actions
    # ---------------------------------------------------------------
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
        reasoning_traces=reasoning_traces,
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
            reasoning_traces=reasoning_traces,
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
        system_prompts[agent.id] = build_system_prompt(agent, state.families, state.agents, grid_size=config.grid_size)

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
                provider=agent.provider,
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
    broadcaster: GameBroadcaster | None = None,
    game_id: str | None = None,
) -> dict | None:
    """Multi-turn family discussion. Sequential within, called in parallel across families. Streams tokens."""
    members = family.living_members(state.agents)
    if len(members) <= 1:
        return None

    transcript: list[dict] = []

    for disc_round in range(state.config.discussion_rounds):
        for agent in sorted(members, key=lambda a: a.tier):
            perception = perceptions.get(agent.id, "")
            prompt = build_discussion_prompt(agent, perception, transcript, disc_round)

            try:
                content = await call_llm_stream(
                    model=agent.model,
                    system_prompt=system_prompts[agent.id],
                    user_prompt=prompt,
                    temperature=agent.temperature,
                    max_tokens=512,
                    provider=agent.provider,
                    on_token=lambda delta, _aid=agent.id, _aname=agent.name: asyncio.get_event_loop().create_task(
                        broadcaster.broadcast_token_delta(game_id, state.round_num, "family_discussion", _aid, _aname, delta)
                    ) if broadcaster else None,
                )
            except Exception as e:
                logger.error("Family discussion LLM call failed for %s (model=%s provider=%s): %s",
                             agent.name, agent.model, agent.provider, e, exc_info=True)
                content = "[Discussion contribution failed]"

            entry = {
                "agent": agent.name,
                "agent_id": agent.id,
                "tier": agent.tier,
                "discussion_round": disc_round,
                "content": content,
            }
            transcript.append(entry)

            # Store as family channel message
            comms.add_messages([Message(
                round=state.round_num,
                sender=agent.id,
                sender_name=agent.name,
                channel="family",
                content=content,
                family=family.name,
            )])

    return {"family": family.name, "transcript": transcript}


async def _get_agent_decision(
    agent: Agent,
    perception: str,
    system_prompt: str,
    grid: Grid,
    agents: dict[str, Agent],
    valid_agent_names: list[str],
    valid_target_names: list[str],
    broadcaster: GameBroadcaster | None = None,
    game_id: str | None = None,
    round_num: int = 0,
) -> tuple[ThinkingResponse, list[Message], Action, dict]:
    """
    Single merged decision call with extended thinking.
    Returns (response, messages, action, parse_info).
    Thinking trace = the core dataset (reasoning traces).
    """
    prompt = build_decision_prompt(agent, perception, grid, agents)

    # Build streaming callback for thinking tokens
    on_thinking: Callable[[str], None] | None = None
    if broadcaster:
        def on_thinking(delta: str, _aid: str = agent.id, _aname: str = agent.name) -> None:
            asyncio.get_event_loop().create_task(
                broadcaster.broadcast_token_delta(game_id, round_num, "deciding", _aid, _aname, delta)
            )

    try:
        response = await call_llm_with_thinking(
            model=agent.model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=agent.temperature,
            max_tokens=1024,
            thinking_budget=8192,
            provider=agent.provider,
            enforce_json=True,
            json_schema=DECISION_JSON_SCHEMA,
            on_thinking_token=on_thinking,
        )
    except Exception as e:
        logger.error("Decision LLM call failed for %s (model=%s provider=%s): %s",
                     agent.name, agent.model, agent.provider, e, exc_info=True)
        # Return a fallback: no messages, stay action, empty thinking
        fallback_response = ThinkingResponse(
            text='{"communicate":{"house":null,"direct_messages":[],"broadcast":null},"action":{"action":"stay","direction":null,"target":null}}',
            thinking_trace=None,
            reasoning_summary=None,
            model=agent.model,
        )
        return fallback_response, [], Action(agent_id=agent.id, type=ActionType.STAY), {"method": "llm_error"}

    try:
        messages, action, parse_info = parse_decision_response(
            response.text, agent.id, agent.name, agent.family,
            round_num, valid_agent_names, valid_target_names,
        )
    except Exception as e:
        logger.error("Decision parse failed for %s: %s (raw: %s)",
                     agent.name, e, response.text[:200])
        messages = []
        action = Action(agent_id=agent.id, type=ActionType.STAY)
        parse_info = {"method": "parse_error_default_stay", "raw_truncated": response.text[:300]}

    return response, messages, action, parse_info


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
        provider=winner.provider,
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
