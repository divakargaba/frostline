"""LLM client abstraction — multi-provider with Gemini, Groq, OpenRouter.

Owner: Div

Provides:
  - LLMClient interface with chat()
  - ProviderPool: tries Gemini -> Groq -> OpenRouter, rotates on failure
  - OpenAICompatClient: generic client for any OpenAI-compatible endpoint
  - MockLLMClient: scripted responses for testing
  - get_llm_client(): auto-selects based on env
  - LLMUsageTracker: per-provider budget tracking
  - Pacing: minimum interval between calls per provider (LLM_MIN_INTERVAL_S)
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
LLM_MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_S", "4"))


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
    provider: str = ""


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
# Usage tracking (per-provider)
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
        return {"requests": [], "total_requests": 0, "total_tokens": 0,
                "by_provider": {}}

    def _save(self):
        try:
            USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            USAGE_PATH.write_text(json.dumps(self._data))
        except Exception as e:
            log.warning("Failed to save usage: %s", e)

    def record(self, model: str, tokens: int, provider: str = "unknown"):
        now = time.time()
        self._data["requests"].append({
            "ts": now, "model": model, "tokens": tokens, "provider": provider,
        })
        self._data["total_requests"] = self._data.get("total_requests", 0) + 1
        self._data["total_tokens"] = self._data.get("total_tokens", 0) + tokens
        # Per-provider counts
        bp = self._data.setdefault("by_provider", {})
        prov = bp.setdefault(provider, {"requests": 0, "tokens": 0})
        prov["requests"] += 1
        prov["tokens"] += tokens
        # Keep only last 24h
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
            "by_provider": self._data.get("by_provider", {}),
        }


_usage_tracker = LLMUsageTracker()


def get_usage_tracker() -> LLMUsageTracker:
    return _usage_tracker


# ---------------------------------------------------------------------------
# Generic OpenAI-compatible client (works for Gemini, Groq, OpenRouter)
# ---------------------------------------------------------------------------

class OpenAICompatClient(LLMClient):
    """Client for any OpenAI-compatible API endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 provider: str, headers: dict | None = None,
                 model_pool: list[str] | None = None,
                 timeout: float = 45.0):
        self.model = model
        self.provider = provider
        self._pool = model_pool or [model]
        self._current_pool_idx = 0
        for i, m in enumerate(self._pool):
            if m == model:
                self._current_pool_idx = i
                break
        extra_headers = headers or {}
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=extra_headers,
            timeout=timeout,
            max_retries=0,  # we handle retries ourselves
        )
        self._total_tokens = 0
        self._last_call_ts = 0.0
        self.disabled = False
        self._disable_reason = ""

    def _pace(self):
        """Enforce minimum interval between calls."""
        elapsed = time.time() - self._last_call_ts
        if elapsed < LLM_MIN_INTERVAL_S:
            wait = LLM_MIN_INTERVAL_S - elapsed
            log.debug("Pacing %s: waiting %.1fs", self.provider, wait)
            time.sleep(wait)
        self._last_call_ts = time.time()

    def check_health(self) -> bool:
        """Quick health check — one cheap call. Returns True if provider works."""
        try:
            resp = self._client.chat.completions.create(
                model=self._pool[0],
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                timeout=10,
            )
            if resp.choices:
                return True
            return False
        except Exception as e:
            code = getattr(e, "status_code", 0)
            log.warning("Health check failed for %s: %s (code=%s)", self.provider, e, code)
            # Don't disable on 429 — rate limits are transient
            # The chat() method will handle retries when actually called
            return False

    def disable(self, reason: str = ""):
        self.disabled = True
        self._disable_reason = reason
        log.warning("Provider %s disabled: %s", self.provider, reason)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        if self.disabled:
            return LLMResponse(
                content=None, tool_calls=[],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model="disabled", provider=self.provider,
            )

        self._pace()

        kwargs: dict[str, Any] = dict(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        models_tried = 0
        max_models = len(self._pool)

        while models_tried < max_models:
            model = self._pool[self._current_pool_idx]
            kwargs["model"] = model

            max_retries = 4  # generous retries for 429s
            for attempt in range(max_retries):
                try:
                    resp = self._client.chat.completions.create(**kwargs)

                    if not resp.choices:
                        log.warning("Empty choices from %s/%s, rotating", self.provider, model)
                        break

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
                                    log.warning("Malformed tool args from %s: %s", model, args)
                                    args = {}
                            tool_calls.append(ToolCall(
                                id=tc.id,
                                name=tc.function.name,
                                arguments=args,
                            ))

                    self._total_tokens += usage["total_tokens"]
                    self.model = model
                    _usage_tracker.record(model, usage["total_tokens"], provider=self.provider)
                    log.info("LLM [%s/%s]: %d tokens", self.provider, model, usage["total_tokens"])

                    return LLMResponse(
                        content=msg.content,
                        tool_calls=tool_calls,
                        usage=usage,
                        model=model,
                        cost_estimate=0.0,
                        provider=self.provider,
                    )

                except Exception as e:
                    code = getattr(e, "status_code", 0)
                    if code == 429:
                        # Honor Retry-After header, default to exponential backoff 20-60s
                        wait = min(20 * (2 ** attempt), 60)
                        if hasattr(e, "response") and hasattr(e.response, "headers"):
                            ra = e.response.headers.get("Retry-After")
                            if ra:
                                try:
                                    wait = min(max(int(ra), 20), 60)
                                except ValueError:
                                    pass
                        log.warning("429 from %s/%s, retry %d/%d, waiting %ds",
                                    self.provider, model, attempt + 1, max_retries, wait)
                        time.sleep(wait)
                        continue  # always retry, don't disable
                    elif code in (500, 502, 503):
                        if attempt < 2:
                            time.sleep(2)
                            continue
                        break
                    elif code == 403:
                        log.warning("403 from %s/%s, skipping", self.provider, model)
                        break
                    else:
                        log.error("LLM error from %s/%s: %s", self.provider, model, e)
                        break

            self._current_pool_idx = (self._current_pool_idx + 1) % max_models
            models_tried += 1

        log.error("All models exhausted for provider %s", self.provider)
        return LLMResponse(
            content=None, tool_calls=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="exhausted", provider=self.provider,
        )


# ---------------------------------------------------------------------------
# Provider pool — tries providers in order, skips disabled
# ---------------------------------------------------------------------------

class ProviderPool(LLMClient):
    """Tries multiple providers in order. Skips disabled providers."""

    def __init__(self, providers: list[OpenAICompatClient]):
        self.providers = providers
        self.model = providers[0].model if providers else "none"

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        for provider in self.providers:
            if provider.disabled:
                log.debug("Skipping disabled provider %s", provider.provider)
                continue
            resp = provider.chat(messages, tools, tool_choice, temperature, max_tokens)
            if resp.model != "exhausted" and resp.model != "disabled":
                self.model = resp.model
                return resp
            log.warning("Provider %s exhausted, trying next", provider.provider)

        log.error("All providers exhausted")
        return LLMResponse(
            content=None, tool_calls=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="exhausted", provider="all",
        )

    def provider_status(self) -> list[dict]:
        """Status of each provider for /health."""
        return [
            {
                "provider": p.provider,
                "model": p.model,
                "disabled": p.disabled,
                "reason": p._disable_reason if p.disabled else "",
            }
            for p in self.providers
        ]


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _build_provider_pool() -> ProviderPool | None:
    """Build provider pool from env vars: Gemini -> Groq -> OpenRouter."""
    providers: list[OpenAICompatClient] = []

    # Gemini (OpenAI-compatible endpoint)
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if gemini_key:
        providers.append(OpenAICompatClient(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_key,
            model=gemini_model,
            provider="gemini",
            timeout=30.0,
        ))

    # Groq
    groq_key = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if groq_key:
        providers.append(OpenAICompatClient(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            model=groq_model,
            provider="groq",
            timeout=30.0,
        ))

    # OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    if openrouter_key:
        pool_str = os.getenv("OPENROUTER_FREE_MODELS", "")
        pool = [m.strip() for m in pool_str.split(",") if m.strip()] if pool_str else [openrouter_model]
        providers.append(OpenAICompatClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            model=openrouter_model,
            provider="openrouter",
            headers={
                "HTTP-Referer": "https://github.com/divakargaba/frostline",
                "X-Title": "Frostline",
            },
            model_pool=pool,
            timeout=45.0,
        ))

    if not providers:
        return None
    return ProviderPool(providers)


# Cached pool instance (built once per process)
_provider_pool: ProviderPool | None = None
_pool_checked = False


def _get_or_build_pool() -> ProviderPool | None:
    global _provider_pool, _pool_checked
    if _provider_pool is not None:
        return _provider_pool
    if _pool_checked:
        return None
    _pool_checked = True
    _provider_pool = _build_provider_pool()
    if _provider_pool:
        # Quick health check on each provider at startup (non-blocking: don't disable on failure)
        for p in _provider_pool.providers:
            ok = p.check_health()
            if not ok:
                log.warning("Provider %s failed health check (may be transient)", p.provider)
            else:
                log.info("Provider %s healthy (%s)", p.provider, p.model)
    return _provider_pool


def get_llm_client() -> LLMClient:
    """Return ProviderPool if any credentials set, else MockLLMClient."""
    from backend.config import LLM_MODE

    if LLM_MODE == "mock":
        log.warning("LLM_MODE=mock — using MockLLMClient")
        return MockLLMClient()

    pool = _get_or_build_pool()
    if pool and any(not p.disabled for p in pool.providers):
        active = [p.provider for p in pool.providers if not p.disabled]
        log.info("Using ProviderPool: %s", ", ".join(active))
        return pool

    log.warning("No active LLM providers — falling back to MockLLMClient")
    return MockLLMClient()


def reset_provider_pool():
    """Reset the cached pool (for testing)."""
    global _provider_pool, _pool_checked
    _provider_pool = None
    _pool_checked = False


# Keep OpenRouterClient as alias for backward compatibility
OpenRouterClient = OpenAICompatClient


# ---------------------------------------------------------------------------
# Fleet transport adapter — wraps ProviderPool for Mico's investigate()
# ---------------------------------------------------------------------------

def make_fleet_transport():
    """Return an async transport function for fleet_llm.investigate().

    Uses our ProviderPool (Gemini -> Groq -> OpenRouter) and converts
    the LLMResponse back to the raw OpenAI response dict that investigate expects.
    Falls back to None if no providers are configured (investigate will use
    its own deterministic fallback).
    """
    from backend.config import LLM_MODE
    if LLM_MODE == "mock":
        return None  # let fleet_llm use its own deterministic fallback

    pool = _get_or_build_pool()
    if pool is None or not any(not p.disabled for p in pool.providers):
        return None

    async def transport(payload, timeout):
        """Adapter: translate between fleet_llm payload format and our pool."""
        import asyncio
        messages = payload.get("messages", [])
        tools = payload.get("tools")
        tool_choice = payload.get("tool_choice", "auto")
        temperature = payload.get("temperature", 0)
        max_tokens = payload.get("max_tokens", 1800)

        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: pool.chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice if isinstance(tool_choice, str) else "auto",
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )

        if resp.model in ("exhausted", "disabled"):
            from src.fleet_llm import ProviderFailure
            raise ProviderFailure(503, 0, False)

        # Convert LLMResponse -> raw OpenAI dict for fleet_llm
        tool_calls_raw = []
        for tc in resp.tool_calls:
            tool_calls_raw.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                },
            })

        return {
            "model": resp.model,
            "choices": [{
                "message": {
                    "content": resp.content,
                    "tool_calls": tool_calls_raw or None,
                },
                "finish_reason": "tool_calls" if tool_calls_raw else "stop",
            }],
            "usage": {
                "prompt_tokens": resp.usage.get("prompt_tokens", 0),
                "completion_tokens": resp.usage.get("completion_tokens", 0),
                "cost": 0,
            },
            "id": f"pool-{resp.provider}-{int(time.time())}",
        }

    return transport
