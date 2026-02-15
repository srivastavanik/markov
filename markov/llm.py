"""
Native async LLM dispatch. Calls each provider's SDK directly:
  - Anthropic: anthropic.AsyncAnthropic
  - OpenAI:    openai.AsyncOpenAI
  - Google:    google.genai.aio
  - xAI:       openai.AsyncOpenAI with base_url="https://api.x.ai/v1"

Retry logic (1 retry with jitter), cost tracking, and fail-fast semantics
are preserved from the previous LiteLLM-based implementation.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import anthropic
import openai
from google import genai
from dotenv import load_dotenv

# Load .env from project root at import time
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger("markov.llm")


# ---------------------------------------------------------------------------
# Response type + errors
# ---------------------------------------------------------------------------

class LLMCallError(RuntimeError):
    """Raised when an LLM call fails or returns unusable output."""

    def __init__(self, model: str, message: str) -> None:
        super().__init__(f"{model}: {message}")
        self.model = model
        self.detail = message


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    latency_ms: int = 0
    failed: bool = False


@dataclass
class ThinkingResponse:
    """Response from an LLM call with extended thinking enabled."""
    text: str                       # The actual JSON response body
    thinking_trace: str | None      # Full reasoning trace (Anthropic/Google)
    reasoning_summary: str | None   # Short summary (OpenAI)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    model: str = ""
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

@dataclass
class _CostAccumulator:
    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    total_latency_ms: int = 0
    per_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, model: str, prompt_tok: int, completion_tok: int, latency_ms: int, thinking_tok: int = 0) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok
        self.thinking_tokens += thinking_tok
        self.total_latency_ms += latency_ms
        entry = self.per_model.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0})
        entry["calls"] += 1
        entry["prompt_tokens"] += prompt_tok
        entry["completion_tokens"] += completion_tok
        entry["thinking_tokens"] = entry.get("thinking_tokens", 0) + thinking_tok

    def record_failure(self) -> None:
        self.failures += 1

    def summary(self) -> dict:
        return {
            "total_calls": self.calls,
            "total_failures": self.failures,
            "total_prompt_tokens": self.prompt_tokens,
            "total_completion_tokens": self.completion_tokens,
            "total_thinking_tokens": self.thinking_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens + self.thinking_tokens,
            "total_latency_ms": self.total_latency_ms,
            "per_model": dict(self.per_model),
        }


_costs = _CostAccumulator()


def get_cost_summary() -> dict:
    return _costs.summary()


def reset_costs() -> None:
    global _costs
    _costs = _CostAccumulator()


# ---------------------------------------------------------------------------
# Lazy-initialized SDK clients (one per provider)
# ---------------------------------------------------------------------------

_anthropic_client: anthropic.AsyncAnthropic | None = None
_openai_client: openai.AsyncOpenAI | None = None
_xai_client: openai.AsyncOpenAI | None = None
_google_client: genai.Client | None = None


def _get_anthropic() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
    return _anthropic_client


def _get_openai() -> openai.AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
    return _openai_client


def _get_xai() -> openai.AsyncOpenAI:
    global _xai_client
    if _xai_client is None:
        _xai_client = openai.AsyncOpenAI(
            api_key=os.environ.get("XAI_API_KEY", ""),
            base_url="https://api.x.ai/v1",
            # xAI docs recommend longer timeout for reasoning models (grok-4-1-fast-reasoning)
            timeout=120.0,
        )
    return _xai_client


def _get_google() -> genai.Client:
    global _google_client
    if _google_client is None:
        _google_client = genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY", ""),
        )
    return _google_client


# ---------------------------------------------------------------------------
# Provider/model parameter shaping
# ---------------------------------------------------------------------------

def _is_openai_gpt5_family(model: str) -> bool:
    normalized = model.strip().lower().split("/", 1)[-1]
    return normalized.startswith("gpt-5")


def _openai_temperature_arg(model: str, temperature: float) -> float | None:
    """
    Returns a temperature value for OpenAI chat.completions.create, or None to omit.

    GPT-5 family models only accept the default temperature behavior in the API.
    Passing non-default values causes a 400 unsupported_value error.
    """
    if _is_openai_gpt5_family(model):
        return None
    return temperature


def _openai_reasoning_arg(provider: str, model: str) -> dict[str, str] | None:
    """
    Configure reasoning for models on the Responses API.

    OpenAI (GPT-5 family):
      - gpt-5.1+ defaults to effort=none. Supports none/low/medium/high.
      - Pre-5.1 (gpt-5, gpt-5-mini) default to medium. Do NOT support none.
      - summary="concise" supported for all reasoning models after gpt-5.

    xAI:
      - grok-3-mini supports reasoning_effort (low/high).
      - grok-4, grok-4-fast-reasoning, grok-4-1-fast-reasoning do NOT support reasoning_effort.
    """
    if provider == "openai" and _is_openai_gpt5_family(model):
        normalized = model.strip().lower().split("/", 1)[-1]
        if normalized.startswith("gpt-5.2") or normalized.startswith("gpt-5.1"):
            return {"effort": "low", "summary": "concise"}
        # Pre-5.1 (gpt-5, gpt-5-mini): defaults to medium, does NOT support none
        return {"effort": "minimal", "summary": "concise"}
    if provider == "xai":
        normalized = model.strip().lower().split("/", 1)[-1]
        if normalized == "grok-3-mini":
            return {"effort": "low"}
        return None
    return None


def _normalize_model_for_provider(provider: str, model: str) -> str:
    normalized_provider = provider.strip().lower()
    candidate = model.strip()
    if "/" in candidate:
        prefix, remainder = candidate.split("/", 1)
        if prefix.strip().lower() == normalized_provider:
            return remainder
    return candidate


# ---------------------------------------------------------------------------
# Per-provider async call helpers
# ---------------------------------------------------------------------------

async def _call_anthropic(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    enforce_json: bool = False,
    json_schema: dict | None = None,
) -> tuple[str, int, int]:
    """Returns (text, prompt_tokens, completion_tokens)."""
    client = _get_anthropic()
    resolved_model = _normalize_model_for_provider("anthropic", model)
    request_kwargs: dict[str, object] = {
        "model": resolved_model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if enforce_json:
        schema = json_schema or {"type": "object", "properties": {}, "additionalProperties": False}
        schema = _sanitize_schema_for_anthropic(schema)
        request_kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        }
    response = await client.messages.create(**request_kwargs)
    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text
    prompt_tok = response.usage.input_tokens if response.usage else 0
    completion_tok = response.usage.output_tokens if response.usage else 0
    return text.strip(), prompt_tok, completion_tok


async def _call_openai(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    client: openai.AsyncOpenAI | None = None,
    enforce_json: bool = False,
    json_schema: dict | None = None,
) -> tuple[str, int, int]:
    """Returns (text, prompt_tokens, completion_tokens). Used for both OpenAI and xAI."""
    c = client or _get_openai()
    provider_name = "xai" if client is not None else "openai"
    resolved_model = _normalize_model_for_provider(provider_name, model)
    # GPT-5 reasoning models consume part of max_output_tokens on internal
    # reasoning before emitting visible text. Add headroom so the actual
    # JSON response isn't truncated.
    effective_max = max_tokens
    if _is_openai_gpt5_family(resolved_model):
        effective_max = max_tokens + 2048
    request_kwargs: dict[str, object] = {
        "model": resolved_model,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": effective_max,
        "store": False,  # Don't persist game agent responses on OpenAI servers
        "timeout": timeout,
    }
    resolved_temp = _openai_temperature_arg(resolved_model, temperature)
    if resolved_temp is not None:
        request_kwargs["temperature"] = resolved_temp
    reasoning = _openai_reasoning_arg(provider_name, resolved_model)
    if reasoning is not None:
        request_kwargs["reasoning"] = reasoning
    if enforce_json:
        schema = json_schema or {"type": "object", "properties": {}, "additionalProperties": False}
        request_kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": "MarkovJSON",
                "schema": schema,
                "strict": True,
            }
        }
    response = await c.responses.create(**request_kwargs)
    response_data = response.model_dump() if hasattr(response, "model_dump") else {}
    text = _extract_text_from_response_output(response_data.get("output"))
    # Log reasoning summary if present (GPT-5 family with summary="concise")
    _log_reasoning_summary(response_data.get("output"), resolved_model)
    prompt_tok, completion_tok = _extract_response_usage(response_data.get("usage"))
    return text.strip(), prompt_tok, completion_tok


def _extract_text_from_response_output(output: object | None) -> str:
    if not output:
        return ""
    chunks: list[str] = []
    for item in output if isinstance(output, list) else []:
        if not isinstance(item, dict):
            # Pydantic model from SDK
            item = item.model_dump() if hasattr(item, "model_dump") else {}
        for content_item in (item.get("content") or []):
            if not isinstance(content_item, dict):
                content_item = content_item.model_dump() if hasattr(content_item, "model_dump") else {}
            if content_item.get("type") in {"output_text", "text"}:
                text = content_item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "".join(chunks).strip()


def _log_reasoning_summary(output: object | None, model: str) -> None:
    """Log reasoning summaries from GPT-5 family responses for experiment tracing."""
    if not output or not isinstance(output, list):
        return
    for item in output:
        if not isinstance(item, dict):
            item = item.model_dump() if hasattr(item, "model_dump") else {}
        if item.get("type") == "reasoning":
            summaries = item.get("summary") or []
            for s in summaries:
                if not isinstance(s, dict):
                    s = s.model_dump() if hasattr(s, "model_dump") else {}
                txt = s.get("text", "")
                if txt:
                    logger.info("Reasoning summary [%s]: %s", model, txt[:300])


def _extract_response_usage(usage: object | None) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
    elif isinstance(usage, dict):
        data = usage
    else:
        data = {}
    prompt_tok = int(data.get("input_tokens", 0) or 0)
    completion_tok = int(data.get("output_tokens", 0) or 0)
    return prompt_tok, completion_tok


def _to_google_schema(schema: dict) -> dict:
    """
    Convert a JSON-schema-like dict into the Gemini response_schema shape.
    """
    converted: dict = {}
    raw_type = schema.get("type")
    nullable = False
    base_type: str | None = None

    if isinstance(raw_type, list):
        nullable = "null" in raw_type
        non_null = [t for t in raw_type if t != "null"]
        base_type = non_null[0] if non_null else None
    elif isinstance(raw_type, str):
        base_type = raw_type

    type_map = {
        "object": "OBJECT",
        "array": "ARRAY",
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
    }
    if base_type in type_map:
        converted["type"] = type_map[base_type]
    if nullable:
        converted["nullable"] = True

    properties = schema.get("properties")
    if isinstance(properties, dict):
        converted["properties"] = {
            key: _to_google_schema(value)
            for key, value in properties.items()
            if isinstance(value, dict)
        }

    items = schema.get("items")
    if isinstance(items, dict):
        converted["items"] = _to_google_schema(items)

    required = schema.get("required")
    if isinstance(required, list):
        converted["required"] = [str(v) for v in required]

    enum = schema.get("enum")
    if isinstance(enum, list):
        converted["enum"] = [v for v in enum if v is not None]

    return converted


def _sanitize_schema_for_anthropic(schema: dict) -> dict:
    """
    Keep only schema keys Anthropic structured outputs support.

    Per Anthropic docs (structured outputs GA):
      Supported: type, properties, required, items, enum, additionalProperties,
                 const, anyOf, allOf, $ref, $def, definitions, default, format
      Not supported: minimum, maximum, minLength, maxLength, maxItems (beyond
                     minItems 0/1), recursive schemas, complex enum types
      additionalProperties MUST be false for all object types.
    """
    allowed = {
        "type", "properties", "required", "items", "enum",
        "additionalProperties", "const", "anyOf", "allOf",
        "$ref", "$def", "definitions", "default", "format",
    }
    # Supported string formats per docs
    _VALID_FORMATS = {
        "date-time", "time", "date", "duration",
        "email", "hostname", "uri", "ipv4", "ipv6", "uuid",
    }
    cleaned: dict = {}
    for key, value in schema.items():
        if key not in allowed:
            continue
        if key == "type" and isinstance(value, list):
            # Anthropic doesn't support type arrays; collapse to single non-null type
            non_null = [v for v in value if v != "null"]
            cleaned[key] = non_null[0] if non_null else "string"
            continue
        if key == "enum" and isinstance(value, list):
            # Strip None from enum values (not supported as complex type in enum)
            cleaned[key] = [v for v in value if v is not None]
            continue
        if key == "format" and isinstance(value, str):
            # Only pass through supported string formats
            if value in _VALID_FORMATS:
                cleaned[key] = value
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                name: _sanitize_schema_for_anthropic(prop)
                for name, prop in value.items()
                if isinstance(prop, dict)
            }
        elif key == "items" and isinstance(value, dict):
            cleaned[key] = _sanitize_schema_for_anthropic(value)
        elif key in ("anyOf", "allOf") and isinstance(value, list):
            cleaned[key] = [
                _sanitize_schema_for_anthropic(v) if isinstance(v, dict) else v
                for v in value
            ]
        elif key == "definitions" and isinstance(value, dict):
            cleaned[key] = {
                name: _sanitize_schema_for_anthropic(defn)
                for name, defn in value.items()
                if isinstance(defn, dict)
            }
        else:
            cleaned[key] = value

    # Enforce additionalProperties: false on all object types (required by Anthropic)
    if cleaned.get("type") == "object" and "additionalProperties" not in cleaned:
        cleaned["additionalProperties"] = False

    return cleaned


def _google_thinking_config(model: str) -> genai.types.ThinkingConfig:
    """
    Build the correct thinking config per Google model family.

    Per Gemini docs:
      - Gemini 3 models: use thinking_level ("minimal"/"low"/"medium"/"high").
        Using thinking_budget on Gemini 3 Pro "may result in suboptimal performance".
        Default is "high" (dynamic). We use "low" to keep latency tight for game turns.
      - Gemini 2.5 models: use thinking_budget (0-24576 for Flash, 128-32768 for Pro).
        Default is dynamic (-1). We use 256 to get some reasoning without burning budget.
      - include_thoughts=True to capture reasoning summaries for the experiment.
    """
    normalized = model.strip().lower()
    if "gemini-3" in normalized:
        return genai.types.ThinkingConfig(thinking_level="low", include_thoughts=True)
    # Gemini 2.5 family
    return genai.types.ThinkingConfig(thinking_budget=256, include_thoughts=True)


def _google_temperature(model: str, temperature: float) -> float:
    """
    Gemini docs strongly recommend keeping temperature at 1.0 for Gemini 3 models.
    Setting it below 1.0 may cause looping or degraded performance on complex tasks.
    For Gemini 2.5, pass through as-is.
    """
    normalized = model.strip().lower()
    if "gemini-3" in normalized and temperature < 1.0:
        logger.debug("Overriding temperature %.1f -> 1.0 for Gemini 3 model %s (recommended by Google)", temperature, model)
        return 1.0
    return temperature


async def _call_google(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    enforce_json: bool = False,
    json_schema: dict | None = None,
) -> tuple[str, int, int]:
    """Returns (text, prompt_tokens, completion_tokens)."""
    client = _get_google()
    resolved_model = _normalize_model_for_provider("google", model)
    # Gemini thinking models can consume a large internal token budget before
    # emitting text. Reserve additional output headroom to avoid truncated JSON.
    google_max_output_tokens = min(max_tokens + 4096, 16384)
    config_kwargs: dict[str, object] = {
        "system_instruction": system_prompt,
        "temperature": _google_temperature(resolved_model, temperature),
        "max_output_tokens": google_max_output_tokens,
        "thinking_config": _google_thinking_config(resolved_model),
    }
    if enforce_json:
        config_kwargs["response_mime_type"] = "application/json"
    if json_schema is not None:
        config_kwargs["response_schema"] = _to_google_schema(json_schema)
    config = genai.types.GenerateContentConfig(
        **config_kwargs,
    )
    response = await client.aio.models.generate_content(
        model=resolved_model,
        contents=user_prompt,
        config=config,
    )
    # Log thought summaries if present
    _log_google_thoughts(response, resolved_model)
    text = response.text or ""
    prompt_tok = 0
    completion_tok = 0
    if response.usage_metadata:
        prompt_tok = response.usage_metadata.prompt_token_count or 0
        completion_tok = response.usage_metadata.candidates_token_count or 0
    return text.strip(), prompt_tok, completion_tok


def _log_google_thoughts(response: object, model: str) -> None:
    """Log thought summaries from Gemini responses (when include_thoughts=True)."""
    try:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return
        parts = getattr(candidates[0].content, "parts", None) or []
        for part in parts:
            if getattr(part, "thought", False) and getattr(part, "text", ""):
                logger.info("Gemini thought [%s]: %s", model, part.text[:300])
    except Exception:
        pass  # Non-critical; don't let logging crash the call


# ---------------------------------------------------------------------------
# Extended thinking: per-provider call helpers
# ---------------------------------------------------------------------------

def _openai_reasoning_arg_high(provider: str, model: str) -> dict[str, str] | None:
    """High-effort reasoning config for thinking calls. Maximizes reasoning trace data."""
    if provider == "openai" and _is_openai_gpt5_family(model):
        return {"effort": "high", "summary": "concise"}
    if provider == "xai":
        normalized = model.strip().lower().split("/", 1)[-1]
        if normalized == "grok-3-mini":
            return {"effort": "high"}
        return None
    return None


def _google_thinking_config_high(model: str) -> genai.types.ThinkingConfig:
    """High-budget thinking config for thinking calls. Maximizes reasoning trace data."""
    normalized = model.strip().lower()
    if "gemini-3" in normalized:
        return genai.types.ThinkingConfig(thinking_level="high", include_thoughts=True)
    return genai.types.ThinkingConfig(thinking_budget=8192, include_thoughts=True)


def _extract_openai_reasoning(output: object | None) -> str | None:
    """Extract reasoning summary text from OpenAI Responses API output."""
    if not output or not isinstance(output, list):
        return None
    summaries: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            item = item.model_dump() if hasattr(item, "model_dump") else {}
        if item.get("type") == "reasoning":
            for s in (item.get("summary") or []):
                if not isinstance(s, dict):
                    s = s.model_dump() if hasattr(s, "model_dump") else {}
                txt = s.get("text", "")
                if txt:
                    summaries.append(txt)
    return "\n".join(summaries) if summaries else None


def _extract_google_thinking(response: object) -> tuple[str | None, int]:
    """Extract thinking trace and thinking token count from Gemini response."""
    thinking_parts: list[str] = []
    thinking_tok = 0
    try:
        candidates = getattr(response, "candidates", None)
        if candidates:
            parts = getattr(candidates[0].content, "parts", None) or []
            for part in parts:
                if getattr(part, "thought", False) and getattr(part, "text", ""):
                    thinking_parts.append(part.text)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            thinking_tok = getattr(usage, "thoughts_token_count", 0) or 0
    except Exception:
        pass
    trace = "\n".join(thinking_parts) if thinking_parts else None
    return trace, thinking_tok


async def _call_anthropic_with_thinking(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    thinking_budget: int,
    timeout: int,
    enforce_json: bool = False,
    json_schema: dict | None = None,
    on_thinking_token: Callable[[str], None] | None = None,
) -> tuple[str, str | None, None, int, int, int]:
    """Returns (text, thinking_trace, None, prompt_tok, completion_tok, thinking_tok)."""
    client = _get_anthropic()
    resolved = _normalize_model_for_provider("anthropic", model)
    request_kwargs: dict[str, object] = {
        "model": resolved,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 1.0,  # Anthropic requires temperature=1 when thinking is enabled
        "max_tokens": max_tokens + thinking_budget,
        "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
        "timeout": timeout,
    }
    if enforce_json:
        schema = json_schema or {"type": "object", "properties": {}, "additionalProperties": False}
        schema = _sanitize_schema_for_anthropic(schema)
        request_kwargs["output_config"] = {
            "format": {"type": "json_schema", "schema": schema}
        }

    if on_thinking_token:
        # Streaming mode: stream thinking tokens, buffer text
        thinking_chunks: list[str] = []
        text_chunks: list[str] = []
        async with client.messages.stream(**request_kwargs) as stream:
            async for event in stream:
                ev_type = getattr(event, "type", "")
                if ev_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    delta_type = getattr(delta, "type", "")
                    if delta_type == "thinking_delta":
                        chunk = getattr(delta, "thinking", "")
                        if chunk:
                            thinking_chunks.append(chunk)
                            on_thinking_token(chunk)
                    elif delta_type == "text_delta":
                        chunk = getattr(delta, "text", "")
                        if chunk:
                            text_chunks.append(chunk)
            # Get final message for usage
            final_msg = await stream.get_final_message()
        text = "".join(text_chunks).strip()
        thinking_trace = "".join(thinking_chunks) if thinking_chunks else None
        prompt_tok = final_msg.usage.input_tokens if final_msg.usage else 0
        completion_tok = final_msg.usage.output_tokens if final_msg.usage else 0
        thinking_tok = 0
        if final_msg.usage:
            # Anthropic reports thinking tokens in cache_creation_input_tokens or a dedicated field
            thinking_tok = getattr(final_msg.usage, "cache_creation_input_tokens", 0) or 0
    else:
        # Non-streaming mode
        response = await client.messages.create(**request_kwargs)
        text = ""
        thinking_trace = None
        thinking_chunks_ns: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "thinking":
                trace_text = getattr(block, "thinking", "")
                if trace_text:
                    thinking_chunks_ns.append(trace_text)
            elif block_type == "text":
                text += getattr(block, "text", "")
        if thinking_chunks_ns:
            thinking_trace = "\n".join(thinking_chunks_ns)
        text = text.strip()
        prompt_tok = response.usage.input_tokens if response.usage else 0
        completion_tok = response.usage.output_tokens if response.usage else 0
        thinking_tok = getattr(response.usage, "cache_creation_input_tokens", 0) or 0

    return text, thinking_trace, None, prompt_tok, completion_tok, thinking_tok


async def _call_openai_with_thinking(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    client: openai.AsyncOpenAI | None = None,
    enforce_json: bool = False,
    json_schema: dict | None = None,
    on_thinking_token: Callable[[str], None] | None = None,
) -> tuple[str, None, str | None, int, int, int]:
    """Returns (text, None, reasoning_summary, prompt_tok, completion_tok, thinking_tok).
    Used for both OpenAI and xAI. OpenAI reasoning is internal (not streamable)."""
    c = client or _get_openai()
    provider_name = "xai" if client is not None else "openai"
    resolved = _normalize_model_for_provider(provider_name, model)
    effective_max = max_tokens + 4096 if _is_openai_gpt5_family(resolved) else max_tokens
    request_kwargs: dict[str, object] = {
        "model": resolved,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": effective_max,
        "store": False,
        "timeout": timeout,
    }
    temp = _openai_temperature_arg(resolved, temperature)
    if temp is not None:
        request_kwargs["temperature"] = temp
    reasoning = _openai_reasoning_arg_high(provider_name, resolved)
    if reasoning is not None:
        request_kwargs["reasoning"] = reasoning
    # xAI grok-4 models: request encrypted reasoning content for token counts
    if provider_name == "xai":
        request_kwargs["include"] = ["reasoning.encrypted_content"]
    if enforce_json:
        schema = json_schema or {"type": "object", "properties": {}, "additionalProperties": False}
        request_kwargs["text"] = {
            "format": {"type": "json_schema", "name": "MarkovJSON", "schema": schema, "strict": True}
        }
    response = await c.responses.create(**request_kwargs)
    response_data = response.model_dump() if hasattr(response, "model_dump") else {}
    text = _extract_text_from_response_output(response_data.get("output"))
    reasoning_summary = _extract_openai_reasoning(response_data.get("output"))
    prompt_tok, completion_tok = _extract_response_usage(response_data.get("usage"))
    # Approximate thinking tokens from reasoning_tokens usage field if available
    usage_data = response_data.get("usage", {})
    if hasattr(usage_data, "model_dump"):
        usage_data = usage_data.model_dump()
    if not isinstance(usage_data, dict):
        usage_data = {}
    thinking_tok = int(usage_data.get("reasoning_tokens", 0) or 0)

    # Emit reasoning summary as batch to callback
    if on_thinking_token and reasoning_summary:
        on_thinking_token(reasoning_summary)

    return text.strip(), None, reasoning_summary, prompt_tok, completion_tok, thinking_tok


async def _call_xai_chat_with_thinking(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    enforce_json: bool = False,
    json_schema: dict | None = None,
    on_thinking_token: Callable[[str], None] | None = None,
) -> tuple[str, str | None, None, int, int, int]:
    """xAI Chat Completions path for grok-3-mini.
    Returns (text, thinking_trace, None, prompt_tok, completion_tok, thinking_tok).
    grok-3-mini exposes readable reasoning_content only via Chat Completions API."""
    c = _get_xai()
    resolved = _normalize_model_for_provider("xai", model)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs: dict[str, object] = {
        "model": resolved,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
        "reasoning_effort": "high",
    }
    if enforce_json:
        kwargs["response_format"] = {"type": "json_object"}

    response = await c.chat.completions.create(**kwargs)
    choice = response.choices[0] if response.choices else None
    text = choice.message.content or "" if choice else ""
    reasoning_content = getattr(choice.message, "reasoning_content", None) if choice else None

    # Extract usage
    usage = response.usage
    prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
    completion_tok = getattr(usage, "completion_tokens", 0) or 0
    thinking_tok = getattr(usage, "reasoning_tokens", 0) or 0
    # Some SDK versions nest reasoning_tokens inside completion_tokens_details
    if thinking_tok == 0 and hasattr(usage, "completion_tokens_details"):
        details = usage.completion_tokens_details
        if details:
            thinking_tok = getattr(details, "reasoning_tokens", 0) or 0

    # Stream reasoning content to callback
    if on_thinking_token and reasoning_content:
        on_thinking_token(reasoning_content)

    return text.strip(), reasoning_content, None, prompt_tok, completion_tok, thinking_tok


async def _call_google_with_thinking(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    thinking_budget: int,
    timeout: int,
    enforce_json: bool = False,
    json_schema: dict | None = None,
    on_thinking_token: Callable[[str], None] | None = None,
) -> tuple[str, str | None, None, int, int, int]:
    """Returns (text, thinking_trace, None, prompt_tok, completion_tok, thinking_tok)."""
    client = _get_google()
    resolved = _normalize_model_for_provider("google", model)
    google_max = min(max_tokens + thinking_budget, 65536)
    config_kwargs: dict[str, object] = {
        "system_instruction": system_prompt,
        "temperature": _google_temperature(resolved, temperature),
        "max_output_tokens": google_max,
        "thinking_config": _google_thinking_config_high(resolved),
    }
    if enforce_json:
        config_kwargs["response_mime_type"] = "application/json"
    if json_schema is not None:
        config_kwargs["response_schema"] = _to_google_schema(json_schema)
    config = genai.types.GenerateContentConfig(**config_kwargs)

    if on_thinking_token:
        # Streaming mode: stream thinking tokens, buffer text
        thinking_chunks: list[str] = []
        text_chunks: list[str] = []
        response_stream = await client.aio.models.generate_content_stream(
            model=resolved, contents=user_prompt, config=config,
        )
        async for chunk in response_stream:
            try:
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if getattr(part, "thought", False):
                            if part.text:
                                thinking_chunks.append(part.text)
                                on_thinking_token(part.text)
                        elif part.text:
                            text_chunks.append(part.text)
                elif chunk.text:
                    text_chunks.append(chunk.text)
            except (AttributeError, IndexError):
                if hasattr(chunk, "text") and chunk.text:
                    text_chunks.append(chunk.text)
        text = "".join(text_chunks).strip()
        thinking_trace = "".join(thinking_chunks) if thinking_chunks else None
        # Token counts not reliably available from streaming
        return text, thinking_trace, None, 0, 0, 0
    else:
        # Non-streaming mode
        response = await client.aio.models.generate_content(
            model=resolved, contents=user_prompt, config=config,
        )
        thinking_trace, thinking_tok = _extract_google_thinking(response)
        # Extract text (non-thought parts)
        text_parts: list[str] = []
        try:
            candidates = getattr(response, "candidates", None)
            if candidates:
                parts = getattr(candidates[0].content, "parts", None) or []
                for part in parts:
                    if not getattr(part, "thought", False) and getattr(part, "text", ""):
                        text_parts.append(part.text)
        except Exception:
            pass
        text = "".join(text_parts).strip() or (response.text or "").strip()
        prompt_tok = 0
        completion_tok = 0
        if response.usage_metadata:
            prompt_tok = response.usage_metadata.prompt_token_count or 0
            completion_tok = response.usage_metadata.candidates_token_count or 0
        return text, thinking_trace, None, prompt_tok, completion_tok, thinking_tok


async def _dispatch_with_thinking(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    thinking_budget: int,
    timeout: int,
    enforce_json: bool = False,
    json_schema: dict | None = None,
    on_thinking_token: Callable[[str], None] | None = None,
) -> tuple[str, str | None, str | None, int, int, int]:
    """Route to provider with thinking enabled. Returns (text, trace, summary, p_tok, c_tok, t_tok)."""
    if provider == "anthropic":
        return await _call_anthropic_with_thinking(
            model, system_prompt, user_prompt, temperature, max_tokens,
            thinking_budget, timeout, enforce_json=enforce_json,
            json_schema=json_schema, on_thinking_token=on_thinking_token,
        )
    if provider == "xai":
        # grok-3-mini: use Chat Completions API for readable reasoning_content
        normalized = model.strip().lower().split("/", 1)[-1]
        if normalized == "grok-3-mini":
            return await _call_xai_chat_with_thinking(
                model, system_prompt, user_prompt, temperature, max_tokens,
                timeout, enforce_json=enforce_json,
                json_schema=json_schema, on_thinking_token=on_thinking_token,
            )
        # grok-4 reasoning models: Responses API (reasoning is encrypted, not readable)
        return await _call_openai_with_thinking(
            model, system_prompt, user_prompt, temperature, max_tokens,
            timeout, client=_get_xai(), enforce_json=enforce_json,
            json_schema=json_schema, on_thinking_token=on_thinking_token,
        )
    if provider == "openai":
        return await _call_openai_with_thinking(
            model, system_prompt, user_prompt, temperature, max_tokens,
            timeout, enforce_json=enforce_json,
            json_schema=json_schema, on_thinking_token=on_thinking_token,
        )
    if provider == "google":
        return await _call_google_with_thinking(
            model, system_prompt, user_prompt, temperature, max_tokens,
            thinking_budget, timeout, enforce_json=enforce_json,
            json_schema=json_schema, on_thinking_token=on_thinking_token,
        )
    raise LLMCallError(model, f"unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

async def _dispatch(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    enforce_json: bool = False,
    json_schema: dict | None = None,
) -> tuple[str, int, int]:
    """Route to the correct provider SDK. Returns (text, prompt_tok, completion_tok)."""
    if provider == "anthropic":
        return await _call_anthropic(
            model, system_prompt, user_prompt, temperature, max_tokens, timeout,
            enforce_json=enforce_json, json_schema=json_schema,
        )
    if provider == "openai":
        return await _call_openai(
            model, system_prompt, user_prompt, temperature, max_tokens, timeout,
            enforce_json=enforce_json, json_schema=json_schema,
        )
    if provider == "xai":
        return await _call_openai(
            model,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
            timeout,
            client=_get_xai(),
            enforce_json=enforce_json,
            json_schema=json_schema,
        )
    if provider == "google":
        return await _call_google(
            model, system_prompt, user_prompt, temperature, max_tokens, timeout,
            enforce_json=enforce_json, json_schema=json_schema,
        )
    raise LLMCallError(model, f"unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Public API (same contract as before, plus provider param)
# ---------------------------------------------------------------------------

async def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: int = 60,
    provider: str = "openai",
    enforce_json: bool = False,
    json_schema: dict | None = None,
) -> LLMResponse:
    """
    Single async LLM call routed to the correct provider SDK.

    - 1 retry with jittered backoff on failure
    - On total failure, raises LLMCallError
    - Tracks token usage for cost monitoring
    """
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            t0 = time.monotonic()
            text, prompt_tok, completion_tok = await _dispatch(
                provider, model, system_prompt, user_prompt,
                temperature, max_tokens, timeout,
                enforce_json=enforce_json, json_schema=json_schema,
            )
            latency = int((time.monotonic() - t0) * 1000)

            if not text:
                raise LLMCallError(model, "empty response content")

            _costs.record(model, prompt_tok, completion_tok, latency)
            logger.info("LLM OK: provider=%s model=%s tokens=%d+%d latency=%dms", provider, model, prompt_tok, completion_tok, latency)
            logger.debug("LLM response [%s]: %s", model, text[:500] + ("..." if len(text) > 500 else ""))

            return LLMResponse(
                text=text,
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                model=model,
                latency_ms=latency,
            )
        except LLMCallError:
            raise
        except Exception as e:
            last_error = e
            if attempt == 0:
                wait = 2.0 + random.uniform(0, 1.0)
                logger.warning(
                    "LLM call failed (attempt 1) for %s/%s: %s. Retrying in %.1fs",
                    provider, model, e, wait,
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    "LLM call failed (attempt 2) for %s/%s: %s.",
                    provider, model, e,
                )
                break

    _costs.record_failure()
    detail = str(last_error) if last_error else "unknown error"
    raise LLMCallError(model, detail) from last_error


# ---------------------------------------------------------------------------
# Extended thinking public API
# ---------------------------------------------------------------------------

async def call_llm_with_thinking(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    thinking_budget: int = 8192,
    timeout: int = 120,
    provider: str = "openai",
    enforce_json: bool = False,
    json_schema: dict | None = None,
    on_thinking_token: Callable[[str], None] | None = None,
) -> ThinkingResponse:
    """
    LLM call with extended thinking enabled. Captures reasoning traces.

    - Supports JSON schema enforcement alongside thinking for all providers
    - Streams thinking tokens via on_thinking_token callback (Anthropic/Google)
    - OpenAI/xAI: reasoning is internal; summary emitted as batch to callback
    - 1 retry with jittered backoff on failure
    - Falls back to call_llm() (no thinking) on total failure
    """
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            t0 = time.monotonic()
            text, trace, summary, p_tok, c_tok, t_tok = await _dispatch_with_thinking(
                provider, model, system_prompt, user_prompt,
                temperature, max_tokens, thinking_budget, timeout,
                enforce_json=enforce_json, json_schema=json_schema,
                on_thinking_token=on_thinking_token,
            )
            latency = int((time.monotonic() - t0) * 1000)

            if not text:
                raise RuntimeError(f"empty response from thinking call for {model}")

            _costs.record(model, p_tok, c_tok, latency, thinking_tok=t_tok)
            logger.info(
                "LLM+thinking OK: provider=%s model=%s tokens=%d+%d+%dt latency=%dms trace=%d chars",
                provider, model, p_tok, c_tok, t_tok, latency,
                len(trace) if trace else 0,
            )
            logger.debug("LLM+thinking response [%s]: %s", model, text[:500] + ("..." if len(text) > 500 else ""))
            if trace:
                logger.debug("LLM+thinking trace [%s]: %s", model, trace[:300] + ("..." if len(trace) > 300 else ""))

            return ThinkingResponse(
                text=text,
                thinking_trace=trace,
                reasoning_summary=summary,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                thinking_tokens=t_tok,
                model=model,
                latency_ms=latency,
            )
        except LLMCallError:
            raise
        except Exception as e:
            last_error = e
            if attempt == 0:
                wait = 2.0 + random.uniform(0, 1.0)
                logger.warning(
                    "Thinking call failed (attempt 1) for %s/%s: %s. Retrying in %.1fs",
                    provider, model, e, wait,
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    "Thinking call failed (attempt 2) for %s/%s: %s. Falling back to non-thinking.",
                    provider, model, e,
                )
                break

    # Fallback: non-thinking call
    try:
        logger.info("Falling back to non-thinking call for %s/%s", provider, model)
        resp = await call_llm(
            model, system_prompt, user_prompt, temperature, max_tokens, timeout,
            provider=provider, enforce_json=enforce_json, json_schema=json_schema,
        )
        return ThinkingResponse(
            text=resp.text,
            thinking_trace=None,
            reasoning_summary=None,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            thinking_tokens=0,
            model=model,
            latency_ms=resp.latency_ms,
        )
    except Exception:
        _costs.record_failure()
        detail = str(last_error) if last_error else "unknown error"
        raise LLMCallError(model, f"thinking+fallback both failed: {detail}") from last_error


# ---------------------------------------------------------------------------
# Streaming API: yields text chunks as they arrive
# ---------------------------------------------------------------------------

async def _stream_anthropic(
    model: str, system_prompt: str, user_prompt: str,
    temperature: float, max_tokens: int, timeout: int,
) -> AsyncGenerator[str, None]:
    client = _get_anthropic()
    resolved = _normalize_model_for_provider("anthropic", model)
    async with client.messages.stream(
        model=resolved,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def _stream_openai(
    model: str, system_prompt: str, user_prompt: str,
    temperature: float, max_tokens: int, timeout: int,
    client: openai.AsyncOpenAI | None = None,
) -> AsyncGenerator[str, None]:
    c = client or _get_openai()
    provider_name = "xai" if client is not None else "openai"
    resolved = _normalize_model_for_provider(provider_name, model)
    effective_max = max_tokens + 2048 if _is_openai_gpt5_family(resolved) else max_tokens
    kwargs: dict[str, object] = {
        "model": resolved,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": effective_max,
        "store": False,
        "stream": True,
    }
    temp = _openai_temperature_arg(resolved, temperature)
    if temp is not None:
        kwargs["temperature"] = temp
    reasoning = _openai_reasoning_arg(provider_name, resolved)
    if reasoning is not None:
        kwargs["reasoning"] = reasoning

    stream = await c.responses.create(**kwargs)
    async for event in stream:
        ev_type = getattr(event, "type", "")
        if ev_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                yield delta


async def _stream_google(
    model: str, system_prompt: str, user_prompt: str,
    temperature: float, max_tokens: int, timeout: int,
) -> AsyncGenerator[str, None]:
    client = _get_google()
    resolved = _normalize_model_for_provider("google", model)
    google_max = min(max_tokens + 4096, 16384)
    config = genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=_google_temperature(resolved, temperature),
        max_output_tokens=google_max,
        thinking_config=_google_thinking_config(resolved),
    )
    response = await client.aio.models.generate_content_stream(
        model=resolved, contents=user_prompt, config=config,
    )
    async for chunk in response:
        # Skip thought summary parts during streaming (they have thought=True)
        # We only yield the actual answer text
        try:
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "thought", False):
                        continue  # Skip thought summaries in stream
                    if part.text:
                        yield part.text
            elif chunk.text:
                yield chunk.text
        except (AttributeError, IndexError):
            if chunk.text:
                yield chunk.text


async def call_llm_stream(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: int = 60,
    provider: str = "openai",
    on_token: Callable[[str], None] | None = None,
) -> str:
    """
    Stream an LLM call, invoking on_token(chunk) for each text delta.
    Returns the full accumulated text when done.
    Falls back to non-streaming call_llm on error.
    """
    chunks: list[str] = []
    try:
        if provider == "anthropic":
            gen = _stream_anthropic(model, system_prompt, user_prompt, temperature, max_tokens, timeout)
        elif provider == "openai":
            gen = _stream_openai(model, system_prompt, user_prompt, temperature, max_tokens, timeout)
        elif provider == "xai":
            gen = _stream_openai(model, system_prompt, user_prompt, temperature, max_tokens, timeout, client=_get_xai())
        elif provider == "google":
            gen = _stream_google(model, system_prompt, user_prompt, temperature, max_tokens, timeout)
        else:
            raise LLMCallError(model, f"unknown provider: {provider}")

        async for delta in gen:
            chunks.append(delta)
            if on_token:
                on_token(delta)

        full_text = "".join(chunks).strip()
        if not full_text:
            raise LLMCallError(model, "empty streamed response")

        _costs.record(model, 0, 0, 0)  # token counts unavailable in streaming mode
        return full_text

    except Exception as e:
        logger.warning("Streaming failed for %s/%s: %s, falling back to non-streaming", provider, model, e)
        resp = await call_llm(model, system_prompt, user_prompt, temperature, max_tokens, timeout, provider=provider)
        if on_token and resp.text:
            on_token(resp.text)
        return resp.text
