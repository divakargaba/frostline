"""LLM client abstraction over OpenRouter (OpenAI-compatible).

Owner: Div

Provides:
  - LLMClient interface with chat()
  - OpenRouterClient: real calls via openai SDK
  - MockLLMClient: scripted responses for testing
  - get_llm_client(): auto-selects based on env
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

log = logging.getLogger("frostline.llm")

# Rough cost per million tokens — updated per model as needed
_DEFAULT_COST_PER_M = {"input": 0.0, "output": 0.0}


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


class OpenRouterClient(LLMClient):
    def __init__(self, api_key: str, model: str,
                 cost_per_m: dict | None = None):
        self.model = model
        self._cost = cost_per_m or _DEFAULT_COST_PER_M
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/divakargaba/frostline",
                "X-Title": "Frostline",
            },
            timeout=30.0,
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
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        last_err = None
        for attempt in range(2):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                last_err = e
                code = getattr(e, "status_code", 0)
                if attempt == 0 and code in (429, 500, 502, 503):
                    log.warning("OpenRouter %s, retrying in 2s", code)
                    time.sleep(2)
                    continue
                raise
        else:
            raise last_err  # type: ignore[misc]

        if not resp.choices:
            return LLMResponse(
                content=None,
                tool_calls=[],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self.model,
            )
        msg = resp.choices[0].message
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "total_tokens": resp.usage.total_tokens if resp.usage else 0,
        }

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            import json
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        cost = (
            usage["prompt_tokens"] * self._cost.get("input", 0)
            + usage["completion_tokens"] * self._cost.get("output", 0)
        ) / 1_000_000
        self._total_tokens += usage["total_tokens"]
        self._total_cost += cost
        log.info("LLM call: %d tokens, ~$%.6f (total: %d tok, ~$%.6f)",
                 usage["total_tokens"], cost, self._total_tokens, self._total_cost)

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            usage=usage,
            model=self.model,
            cost_estimate=cost,
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
    if OPENROUTER_API_KEY and OPENROUTER_MODEL:
        log.info("Using OpenRouter model %s", OPENROUTER_MODEL)
        return OpenRouterClient(OPENROUTER_API_KEY, OPENROUTER_MODEL)
    log.warning("No OPENROUTER_API_KEY/MODEL — falling back to MockLLMClient")
    return MockLLMClient()
