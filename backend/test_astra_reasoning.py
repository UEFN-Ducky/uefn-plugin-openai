from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for k in list(sys.modules):
    if k == "backend" or k.startswith("backend."):
        del sys.modules[k]
for p in _here.parents:
    cand = p / "UEFN-Ducky-Release" / "ducky_app"
    if (cand / "backend" / "agent").is_dir():
        sys.path.insert(0, str(_here))
        sys.path.insert(0, str(cand))
        break

from backend.agent.providers.base import ProviderMessage, ToolCallRequest
from openai_provider import (
    responses_effort,
    to_responses_input,
    to_responses_tools,
    uses_responses_api,
)


def test_uses_responses_and_effort():
    assert uses_responses_api("gpt-6-astra")
    assert uses_responses_api("GPT-6-mini")
    assert not uses_responses_api("gpt-4o")
    assert responses_effort("off") == "low"
    assert responses_effort("high") == "high"


def test_tools_and_input_roundtrip():
    tools = to_responses_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    assert tools == [
        {
            "type": "function",
            "name": "ping",
            "description": "Ping",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    msgs = [
        ProviderMessage(role="user", content="first"),
        ProviderMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCallRequest(id="t1", name="ping", arguments={"x": 1})],
            thinking_blocks=[{"type": "reasoning", "id": "rs_1"}],
        ),
        ProviderMessage(role="tool", tool_call_id="t1", content='{"ok":true}'),
        ProviderMessage(role="assistant", content="done"),
    ]
    out = to_responses_input(msgs)
    assert out[0] == {"role": "user", "content": "first"}
    assert out[1] == {"type": "reasoning", "id": "rs_1"}
    assert out[2]["type"] == "function_call"
    assert out[2]["call_id"] == "t1"
    assert out[3] == {"type": "function_call_output", "call_id": "t1", "output": '{"ok":true}'}
    assert out[4] == {"role": "assistant", "content": "done"}


if __name__ == "__main__":
    test_uses_responses_and_effort()
    test_tools_and_input_roundtrip()
    print("ok")
