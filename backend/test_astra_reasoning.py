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

from openai_provider import (
    apply_chat_reasoning,
    chat_completions_blocks_tools_with_reasoning,
    is_tools_plus_reasoning_error,
)


def test_astra_forces_none_with_tools():
    assert chat_completions_blocks_tools_with_reasoning("gpt-6-astra")
    assert chat_completions_blocks_tools_with_reasoning("GPT-6-mini")
    assert not chat_completions_blocks_tools_with_reasoning("gpt-4o")
    kw = apply_chat_reasoning({"model": "gpt-6-astra"}, model="gpt-6-astra", has_tools=True)
    assert kw["reasoning_effort"] == "none"
    kw = apply_chat_reasoning({"model": "gpt-4o"}, model="gpt-4o", has_tools=True)
    assert "reasoning_effort" not in kw
    kw = apply_chat_reasoning({"model": "gpt-6-astra"}, model="gpt-6-astra", has_tools=False)
    assert "reasoning_effort" not in kw


def test_conflict_error_detect():
    err = Exception(
        "Function tools with reasoning_effort are not supported for gpt-6-astra "
        "in /v1/chat/completions."
    )
    assert is_tools_plus_reasoning_error(err)
    assert not is_tools_plus_reasoning_error(Exception("rate limit exceeded"))


if __name__ == "__main__":
    test_astra_forces_none_with_tools()
    test_conflict_error_detect()
    print("ok")
