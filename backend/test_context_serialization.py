"""Lossless, compact replay of tool arguments on both OpenAI API paths."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

# The plugin and app both have a top-level backend package. Prefer the app's
# package while importing the provider module from this plugin checkout.
_here = Path(__file__).resolve().parent
for _parent in _here.parents:
    _app = _parent / "UEFN-Ducky" / "ducky_app"
    if (_app / "backend" / "agent").is_dir():
        for _name in list(sys.modules):
            if _name == "backend" or _name.startswith("backend."):
                del sys.modules[_name]
        sys.path[:0] = [str(_app), str(_here)]
        break

from backend.agent.providers.base import ProviderMessage, ToolCallRequest
from openai_provider import OpenAIProvider, _responses_usage, to_responses_input


def test_unicode_tool_arguments_are_compact_and_lossless(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    args = {"path": "Verse/Économie/中文.verse", "text": 'Prix : 3 €, "oui"\n  conserver les espaces',
            "values": [1, False, None, {"label": "énergie"}]}
    messages = [ProviderMessage(role="assistant", tool_calls=[
        ToolCallRequest(id="call-1", name="workspace_write_file", arguments=args)
    ])]
    responses_args = to_responses_input(messages)[0]["arguments"]
    chat_args = OpenAIProvider("unused", "gpt-4o")._to_openai_messages("sys", messages)[1]["tool_calls"][0]["function"]["arguments"]
    for serialized in (responses_args, chat_args):
        assert json.loads(serialized) == args
        assert serialized == json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        assert len(serialized) < len(json.dumps(args))


def test_responses_reports_cache_writes_and_reads():
    usage = SimpleNamespace(input_tokens=1000, output_tokens=20,
                            input_tokens_details=SimpleNamespace(cached_tokens=600, cache_write_tokens=200))
    assert _responses_usage(usage) == {"input_tokens": 1000, "output_tokens": 20,
                                       "cache_read_tokens": 600, "cache_write_tokens": 200}
    usage.input_tokens_details = None
    assert _responses_usage(usage)["cache_write_tokens"] == 0
    assert _responses_usage(None) == {}


def test_responses_accounting_with_older_app_parser(monkeypatch):
    import openai_provider

    monkeypatch.setattr(openai_provider, "parse_openai_usage", lambda usage: {
        "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0
    })
    expected = {"input_tokens": 1000, "output_tokens": 20,
                "cache_read_tokens": 600, "cache_write_tokens": 200}
    data = {"input_tokens": 1000, "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 200}}
    assert _responses_usage(data) == expected
    assert _responses_usage(SimpleNamespace(model_dump=lambda **kwargs: data)) == expected
    assert _responses_usage(SimpleNamespace(input_tokens=1000, output_tokens=20,
        input_tokens_details=SimpleNamespace(cached_tokens=600, cache_write_tokens=200))) == expected
