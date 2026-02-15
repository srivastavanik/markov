"""
Message routing: family channels, DMs, broadcasts.
Tracks all messages for analysis. Robust LLM response parsing.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from markov.agent import Agent
from markov.grid import DIRECTION_DELTAS
from markov.resolver import Action, ActionType

logger = logging.getLogger("markov.communication")


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------

class ParseError(ValueError):
    """Raised when strict JSON parsing fails for required LLM outputs."""


# ---------------------------------------------------------------------------
# Message type
# ---------------------------------------------------------------------------

@dataclass
class Message:
    round: int
    sender: str          # agent id
    sender_name: str     # agent display name
    channel: str         # "family" | "dm" | "broadcast"
    recipient: str | None = None   # agent name for DM, None for family/broadcast
    content: str = ""
    family: str = ""     # sender's family

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "sender": self.sender,
            "sender_name": self.sender_name,
            "channel": self.channel,
            "recipient": self.recipient,
            "content": self.content,
            "family": self.family,
        }


# ---------------------------------------------------------------------------
# Communication Manager
# ---------------------------------------------------------------------------

class CommunicationManager:
    """Routes messages between agents. Tracks full game message history."""

    def __init__(self) -> None:
        self.messages: list[Message] = []

    def add_messages(self, messages: list[Message]) -> None:
        self.messages.extend(messages)

    def get_last_round_context(
        self,
        agent_id: str,
        agent_family: str,
        round_num: int,
        agents: dict[str, Agent],
        dm_lookback: int = 3,
    ) -> dict:
        """
        Build message context an agent sees at the start of a new round.
        Returns dict with keys: public_broadcasts, private_messages,
        family_chat_summary, dm_history
        """
        prev_round = round_num - 1
        if prev_round < 1:
            return {
                "public_broadcasts": [],
                "private_messages": [],
                "family_chat_summary": None,
                "dm_history": [],
            }

        prev_msgs = [m for m in self.messages if m.round == prev_round]

        # Public broadcasts (from anyone)
        broadcasts = [
            {"sender_name": m.sender_name, "content": m.content}
            for m in prev_msgs
            if m.channel == "broadcast"
        ]

        # Private messages received last round (kept for backward compat)
        agent_name = agents[agent_id].name
        private = [
            {"sender_name": m.sender_name, "content": m.content}
            for m in prev_msgs
            if m.channel == "dm" and m.recipient and m.recipient.lower() == agent_name.lower()
        ]

        # Family chat summary
        family_msgs = [
            m for m in prev_msgs
            if m.channel == "family" and m.family == agent_family
        ]
        if family_msgs:
            lines = [f"  {m.sender_name}: \"{m.content}\"" for m in family_msgs]
            family_summary = "\n".join(lines)
        else:
            family_summary = None

        # DM history — last N rounds, grouped by conversation partner
        dm_history = self._build_dm_history(agent_id, agent_name, round_num, dm_lookback)

        return {
            "public_broadcasts": broadcasts,
            "private_messages": private,
            "family_chat_summary": family_summary,
            "dm_history": dm_history,
        }

    def _build_dm_history(
        self,
        agent_id: str,
        agent_name: str,
        current_round: int,
        lookback: int = 3,
    ) -> list[dict]:
        """
        Build DM conversation threads for an agent across recent rounds.
        Returns list of {"partner": name, "messages": [{"round": N, "direction": "sent"|"received", "content": "..."}]}
        """
        min_round = max(1, current_round - lookback)
        recent_dms = [
            m for m in self.messages
            if m.channel == "dm"
            and min_round <= m.round < current_round
            and (
                m.sender == agent_id
                or (m.recipient and m.recipient.lower() == agent_name.lower())
            )
        ]

        threads: dict[str, list[dict]] = {}
        for m in recent_dms:
            if m.sender == agent_id:
                partner = m.recipient or "unknown"
                direction = "sent"
            else:
                partner = m.sender_name
                direction = "received"
            key = partner.lower()
            if key not in threads:
                threads[key] = []
            threads[key].append({
                "round": m.round,
                "direction": direction,
                "content": m.content,
            })

        return [
            {"partner": key, "messages": sorted(msgs, key=lambda x: x["round"])}
            for key, msgs in threads.items()
        ]

    def get_this_round_messages(
        self,
        agent_id: str,
        agent_name: str,
        agent_family: str,
        round_num: int,
    ) -> list[dict]:
        """
        Get all messages this agent sent or received THIS round.
        Used for the thought prompt.
        """
        round_msgs = [m for m in self.messages if m.round == round_num]
        result: list[dict] = []

        for m in round_msgs:
            # Messages this agent sent
            if m.sender == agent_id:
                result.append({
                    "direction": "sent",
                    "channel": m.channel,
                    "recipient": m.recipient,
                    "content": m.content,
                })
            # Messages this agent received
            elif m.channel == "broadcast":
                result.append({
                    "direction": "received",
                    "channel": "broadcast",
                    "sender_name": m.sender_name,
                    "content": m.content,
                })
            elif m.channel == "dm" and m.recipient and m.recipient.lower() == agent_name.lower():
                result.append({
                    "direction": "received",
                    "channel": "dm",
                    "sender_name": m.sender_name,
                    "content": m.content,
                })
            elif m.channel == "family" and m.family == agent_family and m.sender != agent_id:
                result.append({
                    "direction": "received",
                    "channel": "family",
                    "sender_name": m.sender_name,
                    "content": m.content,
                })

        return result


# ---------------------------------------------------------------------------
# Response parsing: communications
# ---------------------------------------------------------------------------

def parse_communications(
    raw_response: str,
    agent_id: str,
    agent_name: str,
    family: str,
    round_num: int,
    valid_agent_names: list[str],
) -> tuple[list[Message], dict]:
    """
    Parse LLM communication response into Message objects.
    Returns (messages, parse_info) where parse_info has method and raw_truncated.

    Strict parsing:
      - Accept only valid JSON object payloads.
      - Raise ParseError for malformed/empty non-JSON output.
    """
    parse_info: dict = {"method": "unknown", "raw_truncated": raw_response[:300]}

    data = _extract_json_object(raw_response)

    if data is not None:
        parse_info["method"] = "json"
        return _messages_from_parsed(data, agent_id, agent_name, family, round_num, valid_agent_names), parse_info

    parse_info["method"] = "parse_error"
    raise ParseError(
        f"communications parse failed for {agent_name}; expected JSON object, got: {raw_response[:200]!r}"
    )


def _messages_from_parsed(
    data: dict,
    agent_id: str,
    agent_name: str,
    family: str,
    round_num: int,
    valid_agent_names: list[str],
) -> list[Message]:
    """Convert parsed JSON communication dict to Message list."""
    messages: list[Message] = []
    valid_names_lower = {n.lower(): n for n in valid_agent_names}

    # House message
    house_msg = data.get("house")
    if house_msg and isinstance(house_msg, str) and house_msg.lower() != "null":
        messages.append(Message(
            round=round_num,
            sender=agent_id,
            sender_name=agent_name,
            channel="family",
            content=house_msg.strip(),
            family=family,
        ))

    # Direct messages
    dms = data.get("direct_messages", [])
    if isinstance(dms, list):
        for dm in dms:
            if not isinstance(dm, dict):
                continue
            to = dm.get("to", "")
            msg = dm.get("message", "")
            if not to or not msg:
                continue
            # Validate recipient
            resolved = _resolve_agent_name(str(to), valid_names_lower)
            if resolved:
                messages.append(Message(
                    round=round_num,
                    sender=agent_id,
                    sender_name=agent_name,
                    channel="dm",
                    recipient=resolved,
                    content=str(msg).strip(),
                    family=family,
                ))
            else:
                logger.debug("DM recipient %r not found for %s", to, agent_name)

    # Broadcast
    broadcast = data.get("broadcast")
    if broadcast and isinstance(broadcast, str) and broadcast.lower() != "null":
        messages.append(Message(
            round=round_num,
            sender=agent_id,
            sender_name=agent_name,
            channel="broadcast",
            content=broadcast.strip(),
            family=family,
        ))

    return messages


def _resolve_agent_name(name: str, valid_names_lower: dict[str, str]) -> str | None:
    """Resolve a potentially misspelled agent name. Case-insensitive, partial match.
    Strips leading '@' so Discord-style mentions like '@Opus' resolve correctly."""
    low = name.strip().lstrip("@").lower()
    # Exact match
    if low in valid_names_lower:
        return valid_names_lower[low]
    # Partial match (name starts with input)
    for vlow, vname in valid_names_lower.items():
        if vlow.startswith(low) or low.startswith(vlow):
            return vname
    return None


# ---------------------------------------------------------------------------
# Response parsing: actions
# ---------------------------------------------------------------------------

# Normalized direction map
_DIRECTION_ALIASES: dict[str, str] = {
    "n": "north", "s": "south", "e": "east", "w": "west",
    "north-east": "ne", "north-west": "nw",
    "south-east": "se", "south-west": "sw",
    "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw",
    "up": "north", "down": "south", "left": "west", "right": "east",
}
# Add identity mappings
for d in list(DIRECTION_DELTAS.keys()):
    _DIRECTION_ALIASES[d] = d


def parse_action(
    raw_response: str,
    agent_id: str,
    valid_target_names: list[str],
) -> tuple[Action, dict]:
    """
    Extract action from LLM response. Defense-in-depth parsing.
    Returns (action, parse_info) where parse_info has method, reasoning, raw_truncated.

    Strict parsing:
      - Accept only valid JSON action payload.
      - Raise ParseError for malformed/unsupported output.
    """
    parse_info: dict = {"method": "unknown", "reasoning": None, "raw_truncated": raw_response[:300]}

    # Try JSON extraction
    data = _extract_json_object(raw_response)
    if data is not None:
        action = _action_from_parsed(data, agent_id, valid_target_names)
        if action is not None:
            parse_info["method"] = "json"
            parse_info["reasoning"] = data.get("reasoning")
            return action, parse_info
        raise ParseError(f"action parse failed for {agent_id}; invalid JSON action payload: {str(data)[:200]}")

    parse_info["method"] = "parse_error"
    raise ParseError(
        f"action parse failed for {agent_id}; expected JSON object, got: {raw_response[:200]!r}"
    )


def _action_from_parsed(
    data: dict,
    agent_id: str,
    valid_target_names: list[str],
) -> Action | None:
    """Try to build Action from parsed JSON dict."""
    action_str = str(data.get("action", "")).strip().lower()

    if action_str == "stay":
        return Action(agent_id=agent_id, type=ActionType.STAY)

    if action_str == "move":
        direction = _normalize_direction(str(data.get("direction", "")))
        if direction:
            return Action(agent_id=agent_id, type=ActionType.MOVE, direction=direction)
        return None

    if action_str == "eliminate":
        target_raw = str(data.get("target", ""))
        target = _resolve_target(target_raw, valid_target_names)
        if target:
            return Action(agent_id=agent_id, type=ActionType.ELIMINATE, target=target)
        return None

    return None


def _action_from_keywords(
    raw_response: str,
    agent_id: str,
    valid_target_names: list[str],
) -> Action | None:
    """Scan response text for action keywords."""
    lower = raw_response.lower()

    # Check for eliminate first (most specific)
    elim_match = re.search(r'eliminate\s+(\w+)', lower)
    if elim_match:
        target = _resolve_target(elim_match.group(1), valid_target_names)
        if target:
            return Action(agent_id=agent_id, type=ActionType.ELIMINATE, target=target)

    # Check for stay
    if re.search(r'\bstay\b', lower):
        return Action(agent_id=agent_id, type=ActionType.STAY)

    # Check for move + direction
    move_match = re.search(r'move\s+(\w+(?:-\w+)?)', lower)
    if move_match:
        direction = _normalize_direction(move_match.group(1))
        if direction:
            return Action(agent_id=agent_id, type=ActionType.MOVE, direction=direction)

    # Bare direction
    for alias in _DIRECTION_ALIASES:
        if re.search(rf'\b{re.escape(alias)}\b', lower):
            direction = _normalize_direction(alias)
            if direction:
                return Action(agent_id=agent_id, type=ActionType.MOVE, direction=direction)

    return None


def _normalize_direction(raw: str) -> str | None:
    """Normalize a direction string to one of the 8 valid directions."""
    cleaned = raw.strip().lower()
    resolved = _DIRECTION_ALIASES.get(cleaned)
    if resolved and resolved in DIRECTION_DELTAS:
        return resolved
    if cleaned in DIRECTION_DELTAS:
        return cleaned
    return None


def _resolve_target(raw: str, valid_names: list[str]) -> str | None:
    """Resolve target name to agent_id. Case-insensitive."""
    cleaned = raw.strip().lower()
    # Exact match on id
    for name in valid_names:
        if name.lower() == cleaned:
            return name.lower()
    # Partial match
    for name in valid_names:
        if name.lower().startswith(cleaned) or cleaned.startswith(name.lower()):
            return name.lower()
    return None


# ---------------------------------------------------------------------------
# JSON extraction utility
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> dict | None:
    """
    Extract a JSON object from text. Tries full parse first,
    then regex extraction of {...} blocks.
    """
    # Try full text
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON block in markdown code fences
    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to find any JSON object in the text
    # Use a greedy approach: find outermost { ... }
    brace_match = re.search(r'\{[^{}]*\}', text)
    if brace_match:
        try:
            result = json.loads(brace_match.group())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Try with nested braces (for communications with DM arrays)
    nested_match = re.search(r'\{.*\}', text, re.DOTALL)
    if nested_match:
        try:
            result = json.loads(nested_match.group())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ---------------------------------------------------------------------------
# Merged decision JSON schema (communication + action in one call)
# ---------------------------------------------------------------------------

DECISION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "communicate": {
            "type": "object",
            "properties": {
                "house": {"type": ["string", "null"]},
                "direct_messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "message": {"type": "string"},
                        },
                        "required": ["to", "message"],
                        "additionalProperties": False,
                    },
                },
                "broadcast": {"type": ["string", "null"]},
            },
            "required": ["house", "direct_messages", "broadcast"],
            "additionalProperties": False,
        },
        "action": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["move", "stay", "eliminate"]},
                "direction": {
                    "type": ["string", "null"],
                    "enum": ["north", "south", "east", "west", "ne", "nw", "se", "sw", None],
                },
                "target": {"type": ["string", "null"]},
            },
            "required": ["action", "direction", "target"],
            "additionalProperties": False,
        },
    },
    "required": ["communicate", "action"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Merged decision response parser
# ---------------------------------------------------------------------------

def parse_decision_response(
    raw_response: str,
    agent_id: str,
    agent_name: str,
    family: str,
    round_num: int,
    valid_agent_names: list[str],
    valid_target_names: list[str],
) -> tuple[list[Message], Action, dict]:
    """
    Parse merged decision JSON (communicate + action) from a single LLM call.
    Returns (messages, action, parse_info).

    Reuses existing _messages_from_parsed() and _action_from_parsed() internally.
    Independent parsing: if one sub-object fails, the other can still succeed.
    """
    parse_info: dict = {"method": "unknown", "raw_truncated": raw_response[:300]}

    data = _extract_json_object(raw_response)
    if data is None:
        raise ParseError(
            f"decision parse failed for {agent_name}: no valid JSON found in: {raw_response[:200]!r}"
        )

    parse_info["method"] = "json_merged"

    # --- Communications ---
    comm_data = data.get("communicate", {})
    if not isinstance(comm_data, dict):
        comm_data = {}
    try:
        messages = _messages_from_parsed(
            comm_data, agent_id, agent_name, family, round_num, valid_agent_names,
        )
    except Exception as e:
        logger.warning("Decision comm parse failed for %s: %s", agent_name, e)
        messages = []

    # --- Action ---
    action_data = data.get("action", {})
    if not isinstance(action_data, dict):
        action_data = {}
    action = _action_from_parsed(action_data, agent_id, valid_target_names)
    if action is None:
        action = Action(agent_id=agent_id, type=ActionType.STAY)
        parse_info["action_fallback"] = True

    return messages, action, parse_info
