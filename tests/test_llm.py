"""Tests for LLM wrapper: retry, fail-fast, cost tracking (mocked native SDKs)."""
from unittest.mock import AsyncMock, patch

import pytest

from markov.llm import (
    LLMCallError,
    _call_openai,
    _extract_text_from_response_output,
    _openai_reasoning_arg,
    call_llm,
    get_cost_summary,
    reset_costs,
)


@pytest.fixture(autouse=True)
def clean_costs():
    """Reset cost tracker before each test."""
    reset_costs()
    yield
    reset_costs()


class TestCallLLM:
    @pytest.mark.asyncio
    async def test_successful_call_anthropic(self):
        with patch("markov.llm._call_anthropic", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = ("test output", 100, 50)
            result = await call_llm("claude-opus-4-6", "system", "user", provider="anthropic")
            assert result.text == "test output"
            assert result.prompt_tokens == 100
            assert result.completion_tokens == 50
            assert not result.failed
            mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_successful_call_openai(self):
        with patch("markov.llm._call_openai", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = ("openai output", 80, 40)
            result = await call_llm("gpt-5.2-2025-12-11", "system", "user", provider="openai")
            assert result.text == "openai output"
            assert result.prompt_tokens == 80
            mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_successful_call_google(self):
        with patch("markov.llm._call_google", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = ("gemini output", 60, 30)
            result = await call_llm("gemini-3-pro-preview", "system", "user", provider="google")
            assert result.text == "gemini output"
            mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_successful_call_xai(self):
        with patch("markov.llm._call_openai", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = ("grok output", 70, 35)
            result = await call_llm("grok-4-1-fast-reasoning", "system", "user", provider="xai")
            assert result.text == "grok output"
            mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retry_on_first_failure(self):
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timeout")
            return ("retry output", 10, 5)

        with patch("markov.llm._call_anthropic", new_callable=AsyncMock, side_effect=side_effect):
            result = await call_llm("claude-opus-4-6", "system", "user", provider="anthropic")
            assert result.text == "retry output"
            assert call_count == 2
            assert not result.failed

    @pytest.mark.asyncio
    async def test_raises_on_total_failure(self):
        async def fail(*args, **kwargs):
            raise RuntimeError("boom")

        with patch("markov.llm._call_anthropic", new_callable=AsyncMock, side_effect=fail):
            with pytest.raises(LLMCallError):
                await call_llm("claude-opus-4-6", "system", "user", provider="anthropic")

    @pytest.mark.asyncio
    async def test_cost_accumulation(self):
        with patch("markov.llm._call_anthropic", new_callable=AsyncMock) as mock_a:
            mock_a.return_value = ("a", 100, 50)
            await call_llm("model-a", "s", "u", provider="anthropic")

        with patch("markov.llm._call_openai", new_callable=AsyncMock) as mock_o:
            mock_o.return_value = ("b", 200, 100)
            await call_llm("model-b", "s", "u", provider="openai")

        summary = get_cost_summary()
        assert summary["total_calls"] == 2
        assert summary["total_prompt_tokens"] == 300
        assert summary["total_completion_tokens"] == 150
        assert summary["total_tokens"] == 450

    @pytest.mark.asyncio
    async def test_failure_counted(self):
        async def fail(*args, **kwargs):
            raise RuntimeError("boom")

        with patch("markov.llm._call_anthropic", new_callable=AsyncMock, side_effect=fail):
            with pytest.raises(LLMCallError):
                await call_llm("test-model", "s", "u", provider="anthropic")
            summary = get_cost_summary()
            assert summary["total_failures"] == 1

    @pytest.mark.asyncio
    async def test_raises_on_empty_content(self):
        with patch("markov.llm._call_anthropic", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = ("", 10, 5)
            with pytest.raises(LLMCallError):
                await call_llm("test-model", "s", "u", provider="anthropic")

    @pytest.mark.asyncio
    async def test_reset_costs(self):
        with patch("markov.llm._call_anthropic", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = ("x", 50, 25)
            await call_llm("m", "s", "u", provider="anthropic")
            assert get_cost_summary()["total_calls"] == 1
            reset_costs()
            assert get_cost_summary()["total_calls"] == 0

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        with pytest.raises(LLMCallError, match="unknown provider"):
            await call_llm("some-model", "s", "u", provider="fakeprovider")


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.output_text = content
        self.output = []
        self.usage = {"input_tokens": 3, "output_tokens": 4}


class _FakeResponses:
    def __init__(self) -> None:
        self.last_kwargs = {}

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse("ok")


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


class TestOpenAIParamShaping:
    @pytest.mark.asyncio
    async def test_omits_temperature_for_gpt5_family(self):
        client = _FakeClient()
        await _call_openai(
            model="gpt-5-2025-08-07",
            system_prompt="s",
            user_prompt="u",
            temperature=0.7,
            max_tokens=64,
            timeout=30,
            client=client,
        )
        assert "temperature" not in client.responses.last_kwargs

    @pytest.mark.asyncio
    async def test_keeps_temperature_for_non_gpt5_models(self):
        client = _FakeClient()
        await _call_openai(
            model="gpt-4.1-mini",
            system_prompt="s",
            user_prompt="u",
            temperature=0.2,
            max_tokens=64,
            timeout=30,
            client=client,
        )
        assert client.responses.last_kwargs["temperature"] == 0.2

    def test_extract_text_handles_reasoning_blocks_without_content(self):
        output = [
            {"type": "reasoning", "content": None},
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "{\"action\":\"stay\"}"},
                ],
            },
        ]
        assert _extract_text_from_response_output(output) == "{\"action\":\"stay\"}"

    def test_openai_reasoning_effort_by_model(self):
        assert _openai_reasoning_arg("openai", "gpt-5.2-2025-12-11") == {"effort": "medium"}
        assert _openai_reasoning_arg("openai", "gpt-5-mini-2025-08-07") == {"effort": "minimal"}
        assert _openai_reasoning_arg("xai", "grok-4") is None
