"""LLM client abstraction over OpenRouter (OpenAI-compatible).

Owner: Div

Provides:
  - LLMClient interface with chat()
  - OpenRouterClient: real calls via openai SDK, model rotation on failure
  - MockLLMClient: scripted responses for testing
  - get_llm_client(): auto-selects based on env
  - LLMUsageTracker: per-minute/per-day budget tracking
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

log = logging.getLogger("frostline.llm")

USAGE_PATH = Path("data/state/llm_usage.json")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    usage: dict  # {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    model: str = ""
    cost_estimate: float = 0.0


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        ...


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

class LLMUsageTracker:
    """Track LLM requests per minute and per day. Persisted to JSON."""

    def __init__(self):
        self._data = self._load()

    def _load(self) -> dict:
        if USAGE_PATH.exists():
            try:
                return json.loads(USAGE_PATH.read_text())
            except Exception:
                pass
        return {"requests": [], "total_requests": 0, "total_tokens": 0}

    def _save(self):
        try:
            USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            USAGE_PATH.write_text(json.dumps(self._data))
        except Exception as e:
            log.warning("Failed to save usage: %s", e)

    def record(self, model: str, tokens: int):
        now = time.time()
        self._data["requests"].append({"ts": now, "model": model, "tokens": tokens})
        self._data["total_requests"] = self._data.get("total_requests", 0) + 1
        self._data["total_tokens"] = self._data.get("total_tokens", 0) + tokens
        # Keep only last 24h of request timestamps
        cutoff = now - 86400
        self._data["requests"] = [r for r in self._data["requests"] if r["ts"] > cutoff]
        self._save()

    def requests_last_minute(self) -> int:
        now = time.time()
        return sum(1 for r in self._data.get("requests", []) if r["ts"] > now - 60)

    def requests_last_day(self) -> int:
        return len(self._data.get("requests", []))

    def summary(self) -> dict:
        return {
            "requests_last_minute": self.requests_last_minute(),
            "requests_last_day": self.requests_last_day(),
            "total_requests": self._data.get("total_requests", 0),
            "total_tokens": self._data.get("total_tokens", 0),
        }


_usage_tracker = LLMUsageTracker()


def get_usage_tracker() -> LLMUsageTracker:
    return _usage_tracker


# ---------------------------------------------------------------------------
# OpenRouter client with model rotation
# ---------------------------------------------------------------------------

class OpenRouterClient(LLMClient):
    def __init__(self, api_key: str, model: str,
                 model_pool: list[str] | None = None):
        self.model = model
        self._pool = model_pool or [model]
        self._current_pool_idx = 0
        # Set pool index to match the primary model
        for i, m in enumerate(self._pool):
            if m == model:
                self._current_pool_idx = i
                break
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/divakargaba/frostline",
                "X-Title": "Frostline",
            },
            timeout=45.0,
        )
        self._total_tokens = 0
        self._total_cost = 0.0

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        # Try models in pool order with exponential backoff
        models_tried = 0
        max_models = len(self._pool)
        last_err = None

        while models_tried < max_models:
            model = self._pool[self._current_pool_idx]
            kwargs["model"] = model

            for attempt in range(2):  # 2 attempts per model
                try:
                    resp = self._client.chat.completions.create(**kwargs)

                    # Check for empty choices
                    if not resp.choices:
                        log.warning("Empty choices from %s, rotating", model)
                        break  # Try next model

                    msg = resp.choices[0].message
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                        "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                    }

                    tool_calls: list[ToolCall] = []
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            args = tc.function.arguments
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {}
                            tool_calls.append(ToolCall(
                                id=tc.id,
                                name=tc.function.name,
                                arguments=args,
                            ))

                    self._total_tokens += usage["total_tokens"]
                    self.model = model  # Track which model actually answered
                    _usage_tracker.record(model, usage["total_tokens"])
                    log.info("LLM [%s]: %d tokens (total: %d)",
                             model, usage["total_tokens"], self._total_tokens)

                    return LLMResponse(
                        content=msg.content,
                        tool_calls=tool_calls,
                        usage=usage,
                        model=model,
                        cost_estimate=0.0,  # Free models
                    )

                except Exception as e:
                    last_err = e
                    code = getattr(e, "status_code", 0)
                    if code == 429:
                        # Check Retry-After header
                        retry_after = getattr(e, "headers", {})
                        wait = 2 ** (attempt + 1)  # Exponential backoff: 2, 4
                        if hasattr(e, "response") and hasattr(e.response, "headers"):
                            ra = e.response.headers.get("Retry-After")
                            if ra:
                                try:
                                    wait = min(int(ra), 10)
                                except ValueError:
                                    pass
                        log.warning("429 from %s, waiting %ds", model, wait)
                        time.sleep(wait)
                        if attempt == 0:
                            continue  # Retry same model once
                        break  # Then rotate
                    elif code in (500, 502, 503):
                        if attempt == 0:
                            log.warning("%d from %s, retrying in 2s", code, model)
                            time.sleep(2)
                            continue
                        break
                    elif code == 403:
                        log.warning("403 from %s, skipping", model)
                        break
                    else:
                        log.error("LLM error from %s: %s", model, e)
                        break

            # Rotate to next model
            self._current_pool_idx = (self._current_pool_idx + 1) % max_models
            models_tried += 1
            log.info("Rotating to model %s", self._pool[self._current_pool_idx])

        # All models exhausted
        log.error("All models in pool exhausted")
        return LLMResponse(
            content=None,
            tool_calls=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="exhausted",
        )


@dataclass
class MockLLMClient(LLMClient):
    """Scripted mock for testing. Each chat() pops the next planned response."""
    planned: list[LLMResponse] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if self.planned:
            return self.planned.pop(0)
        return LLMResponse(
            content="Mock response — no more planned responses.",
            tool_calls=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="mock",
        )


def get_llm_client() -> LLMClient:
    """Return OpenRouterClient if credentials are set, else MockLLMClient."""
    from backend.config import OPENROUTER_API_KEY, OPENROUTER_MODEL, LLM_MODE

    if LLM_MODE == "mock":
        log.warning("LLM_MODE=mock — using MockLLMClient")
        return MockLLMClient()

    if OPENROUTER_API_KEY:
        model = OPENROUTER_MODEL or "nvidia/nemotron-3-super-120b-a12b:free"
        pool_str = os.getenv("OPENROUTER_FREE_MODELS", "")
        pool = [m.strip() for m in pool_str.split(",") if m.strip()] if pool_str else [model]
        log.info("Using OpenRouter model %s (pool: %d models)", model, len(pool))
        return OpenRouterClient(OPENROUTER_API_KEY, model, model_pool=pool)

    log.warning("No OPENROUTER_API_KEY — falling back to MockLLMClient")
    return MockLLMClient()
