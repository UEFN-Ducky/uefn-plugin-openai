from __future__ import annotations

import sys
import types
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

import model_fetch
from openai_provider import OpenAIProvider


class _Client:
    made: list = []

    def __init__(self, **kw):
        _Client.made.append(kw)
        self.models = types.SimpleNamespace(list=lambda: [])


def test_client_model_list_and_reasoning_follow_the_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    _Client.made.clear()
    OpenAIProvider("sk", "gpt-4o")._client()
    proxy = OpenAIProvider("sk", "deepseek-reasoner", base_url="https://api.deepseek.com/v1")
    proxy._client()
    assert _Client.made == [{"api_key": "sk"}, {"api_key": "sk", "base_url": "https://api.deepseek.com/v1"}]
    assert proxy._reasoning_field == "reasoning_content"
    assert OpenAIProvider("sk", "gpt-4o")._reasoning_field == ""

    dash: list = []
    monkeypatch.setattr(model_fetch, "_fetch_openai_dashboard", lambda key: dash.append(key) or {})
    monkeypatch.setattr(model_fetch, "_pricing_catalog", lambda: {})
    _Client.made.clear()
    model_fetch.fetch_models("sk-proxy", base_url="https://proxy.example/v1", verify=True)
    assert _Client.made == [{"api_key": "sk-proxy", "base_url": "https://proxy.example/v1"}]
    assert dash == []  # a proxy's key never goes to api.openai.com
