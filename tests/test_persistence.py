"""Tests for persistence module (mocked Supabase)."""
import pytest
from unittest.mock import MagicMock, patch

from markov.persistence import generate_game_id, persist_game


class TestGenerateGameId:
    def test_format(self):
        gid = generate_game_id()
        assert gid.startswith("game_")
        parts = gid.split("_")
        assert len(parts) == 4  # game, date, time, hash


class TestPersistGame:
    def test_returns_false_when_no_key(self):
        with patch("markov.persistence._SUPABASE_KEY", ""):
            result = persist_game("test_id", MagicMock(), MagicMock())
            assert result is False

    def test_returns_false_on_exception(self):
        with patch("markov.persistence._get_client") as mock_client:
            mock_client.return_value = MagicMock()
            mock_client.return_value.table.side_effect = Exception("connection failed")
            result = persist_game("test_id", MagicMock(), MagicMock())
            assert result is False

    def test_graceful_on_none_client(self):
        with patch("markov.persistence._get_client", return_value=None):
            result = persist_game("test_id", MagicMock(), MagicMock())
            assert result is False
