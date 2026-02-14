"""
LiteLLM async wrapper. Handles multi-provider dispatch, retries, cost tracking.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import litellm
from dotenv import load_dotenv

# Load .env from project root at import time
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Suppress LiteLLM's verbose default logging
litellm.suppress_debug_info = True

logger = logging.getLogger("markov.llm")


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

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
# Core LLM call
# ---------------------------------------------------------------------------

_DEFAULT_FALLBACK = '{"action": "stay"}'


async def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: int = 60,
    fallback: str = _DEFAULT_FALLBACK,
) -> LLMResponse:
    """
    Single async LLM call via LiteLLM.

    - 1 retry with jittered backoff on failure
    - On total failure, returns fallback text
    - Tracks token usage for cost monitoring
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(2):
        try:
            t0 = time.monotonic()
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            latency = int((time.monotonic() - t0) * 1000)

            text = response.choices[0].message.content or ""
            usage = response.usage
            prompt_tok = usage.prompt_tokens if usage else 0
            completion_tok = usage.completion_tokens if usage else 0

            _costs.record(model, prompt_tok, completion_tok, latency)
            logger.info("LLM OK: model=%s tokens=%d+%d latency=%dms", model, prompt_tok, completion_tok, latency)

            return LLMResponse(
                text=text,
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                model=model,
                latency_ms=latency,
            )
        except Exception as e:
            if attempt == 0:
                wait = 2.0 + random.uniform(0, 1.0)
                logger.warning("LLM call failed (attempt 1) for %s: %s. Retrying in %.1fs", model, e, wait)
                await asyncio.sleep(wait)
            else:
                logger.error("LLM call failed (attempt 2) for %s: %s. Using fallback.", model, e)
                _costs.record_failure()
                return LLMResponse(text=fallback, model=model, failed=True)

    # Unreachable, but satisfies type checker
    _costs.record_failure()
    return LLMResponse(text=fallback, model=model, failed=True)
