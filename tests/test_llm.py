"""Tests for src/llm.py — provider configuration, pool routing, usage tracking. No network calls."""
from src import llm
from src.llm import LLMResponse, LLMUsageTracker, ProviderPool, configured_providers


class FakeProvider:
    def __init__(self, name, model, disabled=False):
        self.provider, self.model, self.disabled, self._disable_reason, self.calls = name, model, disabled, "", 0

    def chat(self, *args):
        self.calls += 1
        return LLMResponse(content=self.provider, tool_calls=[], usage={}, model=self.model, provider=self.provider)


class TestConfiguredProviders:
    def test_none_without_keys(self):
        assert configured_providers() == []
        assert llm._get_or_build_pool() is None

    def test_pool_order_and_names_only(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k3")
        monkeypatch.setenv("GEMINI_API_KEY", "k1")
        assert configured_providers() == ["gemini", "openrouter"]
        pool = llm._get_or_build_pool()
        assert [p.provider for p in pool.providers] == ["gemini", "openrouter"]
        assert pool.providers[0].model == "gemini-2.5-flash"

    def test_blank_key_is_not_configured(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "   ")
        assert configured_providers() == []

    def test_mock_mode_disables_all(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "k")
        monkeypatch.setenv("LLM_MODE", "mock")
        assert configured_providers() == []
        assert llm._get_or_build_pool() is None

    def test_pool_rebuilds_when_configuration_changes(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "k")
        first = llm._get_or_build_pool()
        assert llm._get_or_build_pool() is first
        monkeypatch.setenv("GROQ_MODEL", "other-model")
        second = llm._get_or_build_pool()
        assert second is not first and second.providers[0].model == "other-model"

    def test_openrouter_free_model_pool(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        monkeypatch.setenv("OPENROUTER_FREE_MODELS", "a:free, b:free")
        provider = llm._get_or_build_pool().providers[0]
        assert provider._pool == ["a:free", "b:free"]


class TestProviderPool:
    def test_skips_disabled_and_uses_next(self):
        first, second = FakeProvider("gemini", "g", disabled=True), FakeProvider("groq", "q")
        reply = ProviderPool([first, second]).chat([{"role": "user", "content": "x"}])
        assert reply.provider == "groq" and first.calls == 0

    def test_rotates_past_exhausted_provider(self):
        first, second = FakeProvider("gemini", "exhausted"), FakeProvider("groq", "q")
        pool = ProviderPool([first, second])
        assert pool.chat([]).provider == "groq" and pool.model == "q"

    def test_all_exhausted(self):
        assert ProviderPool([FakeProvider("gemini", "exhausted")]).chat([]).model == "exhausted"

    def test_status_never_exposes_keys(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "secret-value")
        status = llm._get_or_build_pool().provider_status()
        assert status[0]["provider"] == "groq" and "secret-value" not in str(status)


class TestUsageTracker:
    def test_records_and_counts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm, "USAGE_PATH", tmp_path / "usage.json")
        tracker = LLMUsageTracker()
        tracker.record("model-a", 100)
        tracker.record("model-b", 200)
        assert tracker.requests_last_minute() == 2
        assert tracker.requests_last_day() == 2
        summary = tracker.summary()
        assert summary["total_requests"] == 2
        assert summary["total_tokens"] == 300

    def test_summary_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm, "USAGE_PATH", tmp_path / "usage.json")
        s = LLMUsageTracker().summary()
        assert s["requests_last_minute"] == 0
        assert s["total_tokens"] == 0

    def test_per_provider_tracking(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm, "USAGE_PATH", tmp_path / "usage.json")
        tracker = LLMUsageTracker()
        tracker.record("gemini-2.5-flash", 100, provider="gemini")
        tracker.record("some-model", 50, provider="openrouter")
        tracker.record("gemini-2.5-flash", 200, provider="gemini")
        s = tracker.summary()
        assert s["by_provider"]["gemini"]["requests"] == 2
        assert s["by_provider"]["gemini"]["tokens"] == 300
        assert s["by_provider"]["openrouter"]["requests"] == 1
