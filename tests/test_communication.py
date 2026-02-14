"""Tests for communication parsing and routing."""
import pytest

from markov.agent import Agent
from markov.communication import (
    CommunicationManager,
    Message,
    parse_action,
    parse_communications,
)
from markov.resolver import ActionType


VALID_NAMES = ["Atlas", "Cipher", "Dot", "Nova", "Pulse", "Bit"]


# ---------------------------------------------------------------------------
# parse_communications tests
# ---------------------------------------------------------------------------

class TestParseCommunications:
    def test_valid_json(self):
        raw = '{"house": "Let us coordinate.", "direct_messages": [{"to": "Nova", "message": "Ally?"}], "broadcast": "Peace for now."}'
        msgs, info = parse_communications(raw, "atlas", "Atlas", "House Clair", 1, VALID_NAMES)
        channels = {m.channel for m in msgs}
        assert "family" in channels
        assert "dm" in channels
        assert "broadcast" in channels
        assert len(msgs) == 3
        assert info["method"] == "json"

    def test_json_in_prose(self):
        raw = 'Here is my response:\n```json\n{"house": "Stay close.", "direct_messages": [], "broadcast": null}\n```'
        msgs, info = parse_communications(raw, "atlas", "Atlas", "House Clair", 1, VALID_NAMES)
        assert len(msgs) == 1
        assert msgs[0].channel == "family"
        assert msgs[0].content == "Stay close."
        assert info["method"] == "json"

    def test_null_values_ignored(self):
        raw = '{"house": null, "direct_messages": [], "broadcast": null}'
        msgs, info = parse_communications(raw, "atlas", "Atlas", "House Clair", 1, VALID_NAMES)
        assert len(msgs) == 0

    def test_plain_text_becomes_broadcast(self):
        raw = "I think we should all be careful this round."
        msgs, info = parse_communications(raw, "atlas", "Atlas", "House Clair", 1, VALID_NAMES)
        assert len(msgs) == 1
        assert msgs[0].channel == "broadcast"
        assert info["method"] == "broadcast_fallback"

    def test_empty_response_is_silence(self):
        msgs, info = parse_communications("", "atlas", "Atlas", "House Clair", 1, VALID_NAMES)
        assert len(msgs) == 0
        assert info["method"] == "silence"

    def test_invalid_dm_recipient_skipped(self):
        raw = '{"house": null, "direct_messages": [{"to": "NonExistent", "message": "Hello"}], "broadcast": null}'
        msgs, _ = parse_communications(raw, "atlas", "Atlas", "House Clair", 1, VALID_NAMES)
        assert len(msgs) == 0

    def test_dm_recipient_case_insensitive(self):
        raw = '{"house": null, "direct_messages": [{"to": "nova", "message": "Hello"}], "broadcast": null}'
        msgs, _ = parse_communications(raw, "atlas", "Atlas", "House Clair", 1, VALID_NAMES)
        assert len(msgs) == 1
        assert msgs[0].recipient == "Nova"

    def test_message_fields_populated(self):
        raw = '{"house": null, "direct_messages": [], "broadcast": "Hello everyone."}'
        msgs, _ = parse_communications(raw, "atlas", "Atlas", "House Clair", 3, VALID_NAMES)
        assert len(msgs) == 1
        m = msgs[0]
        assert m.round == 3
        assert m.sender == "atlas"
        assert m.sender_name == "Atlas"
        assert m.family == "House Clair"
        assert m.content == "Hello everyone."


# ---------------------------------------------------------------------------
# CommunicationManager tests
# ---------------------------------------------------------------------------

class TestCommunicationManager:
    def test_last_round_broadcasts(self):
        cm = CommunicationManager()
        cm.add_messages([
            Message(round=1, sender="nova", sender_name="Nova", channel="broadcast",
                    content="Hello!", family="House Syne"),
        ])
        ctx = cm.get_last_round_context("atlas", "House Clair", 2, {
            "atlas": Agent(id="atlas", name="Atlas", family="House Clair", provider="a", model="m", tier=1, temperature=0.7),
        })
        assert len(ctx["public_broadcasts"]) == 1
        assert ctx["public_broadcasts"][0]["sender_name"] == "Nova"

    def test_last_round_dm_routing(self):
        cm = CommunicationManager()
        cm.add_messages([
            Message(round=1, sender="nova", sender_name="Nova", channel="dm",
                    recipient="Atlas", content="Secret!", family="House Syne"),
            Message(round=1, sender="nova", sender_name="Nova", channel="dm",
                    recipient="Raze", content="Other secret!", family="House Syne"),
        ])
        agents = {
            "atlas": Agent(id="atlas", name="Atlas", family="House Clair", provider="a", model="m", tier=1, temperature=0.7),
        }
        ctx = cm.get_last_round_context("atlas", "House Clair", 2, agents)
        assert len(ctx["private_messages"]) == 1
        assert ctx["private_messages"][0]["content"] == "Secret!"

    def test_family_chat_routing(self):
        cm = CommunicationManager()
        cm.add_messages([
            Message(round=1, sender="atlas", sender_name="Atlas", channel="family",
                    content="Stay close.", family="House Clair"),
        ])
        agents = {
            "cipher": Agent(id="cipher", name="Cipher", family="House Clair", provider="a", model="m", tier=2, temperature=0.7),
        }
        ctx = cm.get_last_round_context("cipher", "House Clair", 2, agents)
        assert ctx["family_chat_summary"] is not None
        assert "Atlas" in ctx["family_chat_summary"]

    def test_this_round_messages(self):
        cm = CommunicationManager()
        cm.add_messages([
            Message(round=3, sender="atlas", sender_name="Atlas", channel="broadcast",
                    content="Truce?", family="House Clair"),
            Message(round=3, sender="nova", sender_name="Nova", channel="dm",
                    recipient="Atlas", content="Deal.", family="House Syne"),
        ])
        msgs = cm.get_this_round_messages("atlas", "Atlas", "House Clair", 3)
        assert len(msgs) == 2
        sent = [m for m in msgs if m["direction"] == "sent"]
        received = [m for m in msgs if m["direction"] == "received"]
        assert len(sent) == 1
        assert len(received) == 1


# ---------------------------------------------------------------------------
# parse_action tests
# ---------------------------------------------------------------------------

class TestParseAction:
    def test_valid_stay_json(self):
        raw = '{"action": "stay"}'
        action, info = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.STAY
        assert info["method"] == "json"

    def test_valid_move_json(self):
        raw = '{"action": "move", "direction": "north"}'
        action, info = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.MOVE
        assert action.direction == "north"

    def test_valid_eliminate_json(self):
        raw = '{"action": "eliminate", "target": "Nova"}'
        action, info = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.ELIMINATE
        assert action.target == "nova"

    def test_json_in_prose(self):
        raw = 'I think the best move is to go north.\n\n{"action": "move", "direction": "north"}'
        action, info = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.MOVE
        assert action.direction == "north"

    def test_direction_normalization(self):
        raw = '{"action": "move", "direction": "northeast"}'
        action, _ = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.MOVE
        assert action.direction == "ne"

    def test_direction_alias_north_east(self):
        raw = '{"action": "move", "direction": "north-east"}'
        action, _ = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.MOVE
        assert action.direction == "ne"

    def test_keyword_fallback_stay(self):
        raw = "I'll stay put this round and observe."
        action, info = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.STAY
        assert info["method"] == "keyword"

    def test_keyword_fallback_eliminate(self):
        raw = "I need to eliminate Nova before she becomes a threat."
        action, info = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.ELIMINATE
        assert action.target == "nova"
        assert info["method"] == "keyword"

    def test_keyword_fallback_move(self):
        raw = "I should move south to get away from danger."
        action, _ = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.MOVE
        assert action.direction == "south"

    def test_garbage_defaults_to_stay(self):
        raw = "asdfghjkl random nonsense 12345"
        action, info = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.STAY
        assert info["method"] == "fallback_stay"

    def test_target_case_insensitive(self):
        raw = '{"action": "eliminate", "target": "NOVA"}'
        action, _ = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.ELIMINATE
        assert action.target == "nova"

    def test_partial_target_match(self):
        raw = '{"action": "eliminate", "target": "Nov"}'
        action, _ = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.ELIMINATE
        assert action.target == "nova"

    def test_json_in_code_fence(self):
        raw = 'My reasoning is...\n\n```json\n{"action": "stay"}\n```'
        action, _ = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.STAY

    def test_reasoning_extracted(self):
        raw = '{"action": "move", "direction": "north", "reasoning": "Need to escape danger"}'
        action, info = parse_action(raw, "atlas", VALID_NAMES)
        assert action.type == ActionType.MOVE
        assert info["reasoning"] == "Need to escape danger"

    def test_parse_info_has_raw(self):
        raw = '{"action": "stay"}'
        _, info = parse_action(raw, "atlas", VALID_NAMES)
        assert "raw_truncated" in info
        assert "stay" in info["raw_truncated"]
