"""Context comes from the window field / official docs — never max_tokens (output cap)."""

from __future__ import annotations

import importlib
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

# Load the plugin as a package while keeping the host's backend namespace.
_package = types.ModuleType("openai_context_test")
_package.__path__ = [str(_here)]
sys.modules[_package.__name__] = _package
_model_fetch = importlib.import_module("openai_context_test.model_fetch")
_context_from_record = _model_fetch._context_from_record
_openai_info_from_dashboard = _model_fetch._openai_info_from_dashboard
parse_openai_docs_md = _model_fetch.parse_openai_docs_md


ASTRA_MD = """
# GPT-6 Astra
Model ID: `gpt-6-astra`
- 1,050,000 context window
- 128,000 max output tokens

| Metric | Price | Unit |
| --- | ---: | --- |
| Input | $10 | 1M tokens |
| Cached input | $1 | 1M tokens |
| Cache writes | $12.5 | 1M tokens |
| Output | $50 | 1M tokens |
"""


def test_max_tokens_is_not_context():
    rec = {"max_tokens": 128000, "features": ["function_calling"]}
    assert _context_from_record(rec) is None
    info = _openai_info_from_dashboard(rec, "gpt-6-astra")
    assert info.context_limit is None


def test_context_window_wins_over_max_tokens():
    rec = {"max_tokens": 128000, "context_window": 1_050_000, "features": ["function_calling"]}
    assert _context_from_record(rec) == 1_050_000
    info = _openai_info_from_dashboard(rec, "gpt-6-astra")
    assert info.context_limit == 1_050_000


def test_docs_md_is_the_live_source():
    spec = parse_openai_docs_md(ASTRA_MD)
    assert spec["context_limit"] == 1_050_000
    assert spec["price_in"] == 10
    assert spec["price_out"] == 50
    assert spec["price_cached_in"] == 1
    info = _openai_info_from_dashboard({"max_tokens": 128000, "features": []}, "gpt-6-astra", docs=spec)
    assert info.context_limit == 1_050_000
    assert info.price_in == 10
    assert info.price_out == 50
