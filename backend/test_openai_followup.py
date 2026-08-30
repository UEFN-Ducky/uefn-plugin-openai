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
from openai_provider import OpenAIProvider


def test_followup_comes_after_previous_reply():
    msgs = [
        ProviderMessage(role="user", content="first"),
        ProviderMessage(
            role="assistant",
            content="Checking",
            tool_calls=[ToolCallRequest(id="t1", name="ping", arguments={})],
        ),
        ProviderMessage(role="tool", tool_call_id="t1", content='{"ok":true}'),
        ProviderMessage(role="assistant", content="Here is the answer."),
        ProviderMessage(role="user", content="follow up"),
    ]
    out = OpenAIProvider("k", "gpt-4o")._to_openai_messages("sys", msgs)
    roles = [m["role"] for m in out if m["role"] != "system"]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]
    assert out[-2]["content"] == "Here is the answer."
    assert out[-1]["content"] == "follow up"
