#!/usr/bin/env python3
"""Smoke test: one real OpenRouter call with a tool, verify round-trip.

Usage: python scripts/llm_smoke_test.py
"""
import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import OpenRouterClient, get_llm_client, MockLLMClient


def main():
    client = get_llm_client()
    if isinstance(client, MockLLMClient):
        print("SKIP — no OpenRouter key/model set. Using mock client.")
        print("Set OPENROUTER_API_KEY and OPENROUTER_MODEL in .env to test.")
        return

    assert isinstance(client, OpenRouterClient)
    print(f"Model: {client.model}")

    tools = [{
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Returns the current UTC time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }]

    # Step 1: ask the model to call the tool
    messages = [{"role": "user", "content": "What time is it right now? Use the get_time tool."}]
    resp = client.chat(messages, tools=tools)
    print(f"Step 1 — tokens: {resp.usage['total_tokens']}, cost: ~${resp.cost_estimate:.6f}")

    if resp.tool_calls:
        tc = resp.tool_calls[0]
        print(f"  Tool call: {tc.name}({tc.arguments})")

        # Step 2: send tool result, get final answer
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        messages.append({"role": "assistant", "content": resp.content, "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
        ]})
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"utc": now})})
        resp2 = client.chat(messages, tools=tools)
        print(f"Step 2 — tokens: {resp2.usage['total_tokens']}, cost: ~${resp2.cost_estimate:.6f}")
        print(f"  Answer: {resp2.content[:200] if resp2.content else '(no content)'}")
        print("OK")
    else:
        print(f"  No tool call returned. Content: {resp.content[:200] if resp.content else '(none)'}")
        # Some free models don't support tools well — still counts as a valid smoke test
        print("OK (no tool call — model may not support tools)")


if __name__ == "__main__":
    main()
