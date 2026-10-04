"""Multi-provider LLM pool (Gemini -> Groq -> OpenRouter) and the fleet investigation transport.

Owner: Div

Provides:
  - OpenAICompatClient: one OpenAI-compatible endpoint with a model pool, pacing and retries
  - ProviderPool: tries providers in order, skipping disabled/exhausted ones
  - configured_providers(): provider names with a key set (never the keys)
  - LLMUsageTracker: per-provider request/token accounting
  - make_fleet_transport(): async transport for src.fleet_llm.investigate()

LLM_MODE=mock disables every provider (tests, offline runs).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

log = logging.getLogger("frostline.llm")

USAGE_PATH = Path("data/state/llm_usage.json")
LLM_MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_S", "4"))
EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
OPENROUTER_HEADERS = {"HTTP-Referer": "https://github.com/divakargaba/frostline", "X-Title": "Frostline"}
# name, key env, model env, default model, base url, timeout (s)
PROVIDERS = [
    ("gemini", "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta/openai/", 30.0),
    ("groq", "GROQ_API_KEY", "GROQ_MODEL", "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1", 30.0),
    ("openrouter", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free", "https://openrouter.ai/api/v1", 45.0),
]


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


def _unavailable(model, provider):
    return LLMResponse(content=None, tool_calls=[], usage=dict(EMPTY_USAGE), model=model, provider=provider)


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
        return {"requests": [], "total_requests": 0, "total_tokens": 0, "by_provider": {}}

    def _save(self):
        try:
            USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            USAGE_PATH.write_text(json.dumps(self._data))
        except Exception as e:
            log.warning("Failed to save usage: %s", e)

    def record(self, model: str, tokens: int, provider: str = "unknown"):
        now = time.time()
        self._data["requests"].append({"ts": now, "model": model, "tokens": tokens, "provider": provider})
        self._data["total_requests"] = self._data.get("total_requests", 0) + 1
        self._data["total_tokens"] = self._data.get("total_tokens", 0) + tokens
        prov = self._data.setdefault("by_provider", {}).setdefault(provider, {"requests": 0, "tokens": 0})
        prov["requests"] += 1
        prov["tokens"] += tokens
        self._data["requests"] = [r for r in self._data["requests"] if r["ts"] > now - 86400]
        self._save()

    def requests_last_minute(self) -> int:
        now = time.time()
        return sum(1 for r in self._data.get("requests", []) if r["ts"] > now - 60)

    def requests_last_day(self) -> int:
        return len(self._data.get("requests", []))

    def summary(self) -> dict:
        return {"requests_last_minute": self.requests_last_minute(), "requests_last_day": self.requests_last_day(),
                "total_requests": self._data.get("total_requests", 0), "total_tokens": self._data.get("total_tokens", 0),
                "by_provider": self._data.get("by_provider", {})}


_usage_tracker = LLMUsageTracker()


def get_usage_tracker() -> LLMUsageTracker:
    return _usage_tracker


class OpenAICompatClient:
    """Client for any OpenAI-compatible API endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str, provider: str, headers: dict | None = None,
                 model_pool: list[str] | None = None, timeout: float = 45.0):
        self.model = model
        self.provider = provider
        self._pool = model_pool or [model]
        self._current_pool_idx = self._pool.index(model) if model in self._pool else 0
        self._client = OpenAI(base_url=base_url, api_key=api_key, default_headers=headers or {},
                              timeout=timeout, max_retries=0)  # retries handled below
        self._last_call_ts = 0.0
        self.disabled = False
        self._disable_reason = ""

    def _pace(self):
        """Enforce minimum interval between calls."""
        elapsed = time.time() - self._last_call_ts
        if elapsed < LLM_MIN_INTERVAL_S:
            time.sleep(LLM_MIN_INTERVAL_S - elapsed)
        self._last_call_ts = time.time()

    def disable(self, reason: str = ""):
        self.disabled = True
        self._disable_reason = reason
        log.warning("Provider %s disabled: %s", self.provider, reason)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: Any = "auto",
             temperature: float = 0, max_tokens: int = 1024) -> LLMResponse:
        if self.disabled:
            return _unavailable("disabled", self.provider)
        self._pace()
        kwargs: dict[str, Any] = dict(messages=messages, temperature=temperature, max_tokens=max_tokens)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        for _ in range(len(self._pool)):
            model = kwargs["model"] = self._pool[self._current_pool_idx]
            for attempt in range(4):
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                    if not resp.choices:
                        log.warning("Empty choices from %s/%s, rotating", self.provider, model)
                        break
                    msg = resp.choices[0].message
                    usage = {"prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                             "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                             "total_tokens": resp.usage.total_tokens if resp.usage else 0}
                    tool_calls = []
                    for tc in msg.tool_calls or []:
                        args = tc.function.arguments
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                log.warning("Malformed tool args from %s: %s", model, args)
                                args = {}
                        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
                    self.model = model
                    _usage_tracker.record(model, usage["total_tokens"], provider=self.provider)
                    log.info("LLM [%s/%s]: %d tokens", self.provider, model, usage["total_tokens"])
                    return LLMResponse(content=msg.content, tool_calls=tool_calls, usage=usage, model=model, provider=self.provider)
                except Exception as e:
                    code = getattr(e, "status_code", 0)
                    if code == 429:
                        # Honor Retry-After, default to exponential backoff 20-60s.
                        wait = min(20 * (2 ** attempt), 60)
                        retry_after = getattr(getattr(e, "response", None), "headers", {}).get("Retry-After")
                        if retry_after:
                            try:
                                wait = min(max(int(retry_after), 20), 60)
                            except ValueError:
                                pass
                        log.warning("429 from %s/%s, retry %d/4, waiting %ds", self.provider, model, attempt + 1, wait)
                        time.sleep(wait)
                        continue
                    if code == 400 and isinstance(kwargs.get("tool_choice"), dict):
                        # Some OpenAI-compatible endpoints only accept string tool_choice values.
                        kwargs["tool_choice"] = "required"
                        continue
                    if code in (500, 502, 503) and attempt < 2:
                        time.sleep(2)
                        continue
                    log.error("LLM error from %s/%s: %s", self.provider, model, e)
                    break
            self._current_pool_idx = (self._current_pool_idx + 1) % len(self._pool)
        log.error("All models exhausted for provider %s", self.provider)
        return _unavailable("exhausted", self.provider)


class ProviderPool:
    """Tries multiple providers in order. Skips disabled providers."""

    def __init__(self, providers: list[OpenAICompatClient]):
        self.providers = providers
        self.model = providers[0].model if providers else "none"

    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: Any = "auto",
             temperature: float = 0, max_tokens: int = 1024) -> LLMResponse:
        for provider in self.providers:
            if provider.disabled:
                continue
            resp = provider.chat(messages, tools, tool_choice, temperature, max_tokens)
            if resp.model not in ("exhausted", "disabled"):
                self.model = resp.model
                return resp
            log.warning("Provider %s exhausted, trying next", provider.provider)
        log.error("All providers exhausted")
        return _unavailable("exhausted", "all")

    def provider_status(self) -> list[dict]:
        return [{"provider": p.provider, "model": p.model, "disabled": p.disabled,
                 "reason": p._disable_reason if p.disabled else ""} for p in self.providers]


def llm_mocked() -> bool:
    return os.getenv("LLM_MODE", "").strip().lower() == "mock"


def configured_providers() -> list[str]:
    """Provider names, in pool order, whose API key is set. Never returns key values."""
    if llm_mocked():
        return []
    return [name for name, key, *_ in PROVIDERS if os.getenv(key, "").strip()]


def _build_provider_pool() -> ProviderPool | None:
    providers = []
    for name, key, model_env, default, base_url, timeout in PROVIDERS:
        api_key = os.getenv(key, "").strip()
        if not api_key:
            continue
        model = os.getenv(model_env, "").strip() or default
        pool = None
        if name == "openrouter":
            pool = [m.strip() for m in os.getenv("OPENROUTER_FREE_MODELS", "").split(",") if m.strip()] or None
        providers.append(OpenAICompatClient(base_url=base_url, api_key=api_key, model=model, provider=name,
                                            headers=OPENROUTER_HEADERS if name == "openrouter" else None,
                                            model_pool=pool, timeout=timeout))
    return ProviderPool(providers) if providers else None


_provider_pool: ProviderPool | None = None
_pool_key: tuple | None = None


def _get_or_build_pool() -> ProviderPool | None:
    """Build the pool once per distinct provider configuration (no network calls)."""
    global _provider_pool, _pool_key
    if llm_mocked():
        return None
    key = tuple(os.getenv(env, "") for spec in PROVIDERS for env in spec[1:3]) + (os.getenv("OPENROUTER_FREE_MODELS", ""),)
    if key != _pool_key:
        _provider_pool, _pool_key = _build_provider_pool(), key
    return _provider_pool


def reset_provider_pool():
    """Reset the cached pool (for testing)."""
    global _provider_pool, _pool_key
    _provider_pool = _pool_key = None


def make_fleet_transport():
    """Return an async transport for fleet_llm.investigate() backed by the provider pool.

    Converts each LLMResponse back into the raw OpenAI response dict investigate() expects.
    Returns None in mock mode or when no provider is configured.
    """
    pool = _get_or_build_pool()
    if pool is None or all(p.disabled for p in pool.providers):
        return None

    async def transport(payload, timeout):
        from src.fleet_llm import ProviderFailure
        resp = await asyncio.to_thread(pool.chat, messages=payload.get("messages", []), tools=payload.get("tools"),
                                       tool_choice=payload.get("tool_choice", "auto"), temperature=payload.get("temperature", 0),
                                       max_tokens=payload.get("max_tokens", 1800))
        if resp.model in ("exhausted", "disabled"):
            raise ProviderFailure(503)
        calls = [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments}}
                 for tc in resp.tool_calls]
        return {"id": f"pool-{resp.provider}-{int(time.time())}", "model": resp.model, "provider": resp.provider,
                "choices": [{"message": {"content": resp.content, "tool_calls": calls or None}, "finish_reason": "tool_calls" if calls else "stop"}],
                "usage": {"prompt_tokens": resp.usage.get("prompt_tokens", 0), "completion_tokens": resp.usage.get("completion_tokens", 0), "cost": 0}}

    return transport
