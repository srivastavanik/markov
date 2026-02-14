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
from dataclasses import dataclass, field
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

@dataclass
class _CostAccumulator:
    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: int = 0
    per_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, model: str, prompt_tok: int, completion_tok: int, latency_ms: int) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok
        self.total_latency_ms += latency_ms
        entry = self.per_model.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        entry["calls"] += 1
        entry["prompt_tokens"] += prompt_tok
        entry["completion_tokens"] += completion_tok

    def record_failure(self) -> None:
        self.failures += 1

    def summary(self) -> dict:
        return {
            "total_calls": self.calls,
            "total_failures": self.failures,
            "total_prompt_tokens": self.prompt_tokens,
            "total_completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
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
    Keep GPT-5 family focused on answer tokens instead of consuming the
    full budget on internal reasoning in short-turn game calls.
    """
    if provider == "openai" and _is_openai_gpt5_family(model):
        normalized = model.strip().lower().split("/", 1)[-1]
        if normalized.startswith("gpt-5.2"):
            return {"effort": "medium"}
        return {"effort": "minimal"}
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
    """
    allowed = {"type", "properties", "required", "items", "enum", "additionalProperties"}
    cleaned: dict = {}
    for key, value in schema.items():
        if key not in allowed:
            continue
        if key == "type" and isinstance(value, list):
            non_null = [v for v in value if v != "null"]
            cleaned[key] = non_null[0] if non_null else "string"
            continue
        if key == "enum" and isinstance(value, list):
            cleaned[key] = [v for v in value if v is not None]
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                name: _sanitize_schema_for_anthropic(prop)
                for name, prop in value.items()
                if isinstance(prop, dict)
            }
        elif key == "items" and isinstance(value, dict):
            cleaned[key] = _sanitize_schema_for_anthropic(value)
        else:
            cleaned[key] = value
    return cleaned


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
    google_max_output_tokens = min(max_tokens + 1024, 4096)
    config_kwargs: dict[str, object] = {
        "system_instruction": system_prompt,
        "temperature": temperature,
        "max_output_tokens": google_max_output_tokens,
        "thinking_config": genai.types.ThinkingConfig(thinking_budget=128),
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
    text = response.text or ""
    prompt_tok = 0
    completion_tok = 0
    if response.usage_metadata:
        prompt_tok = response.usage_metadata.prompt_token_count or 0
        completion_tok = response.usage_metadata.candidates_token_count or 0
    return text.strip(), prompt_tok, completion_tok


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
