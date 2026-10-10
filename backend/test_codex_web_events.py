"""Representative wire fixtures; no CLI/provider or live web calls."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def stream():
    spec = importlib.util.spec_from_file_location("_web_event_adapter", Path(__file__).with_name("codex_adapter.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []
    return module._CodexStream("fixture-chat", "fixture-run", events.append), events


def emit(stream, item, start=None):
    parser, events = stream
    if start is not None:
        parser.on_line(json.dumps({"type": "item.started", "item": {"id": "web", "type": "web_search", **start}}))
    parser.on_line(json.dumps({"type": "item.completed", "item": {"id": "web", "type": "web_search", **item}}))
    return events[-1]["tool"], parser.blocks[-1]


@pytest.mark.parametrize("action", [
    {"type": "search", "query": "Codex MCP", "queries": ["Codex MCP"]},
    {"type": "open_page", "url": "https://developers.openai.com/codex/mcp"},
    {"type": "find_in_page", "url": "https://developers.openai.com/codex/mcp", "pattern": "MCP"},
])
def test_completion_retains_action_and_content_after_minimal_start(stream, action):
    tool, block = emit(stream, {"action": action, "content": "MCP configuration reference", "sources": [{"title": "Codex MCP", "url": "https://developers.openai.com/codex/mcp"}]}, start={})
    assert tool["arguments"]["action"] == action
    data = json.loads(tool["result"])
    assert data["text"] == "MCP configuration reference"
    assert data["sources"][0]["url"] == "https://developers.openai.com/codex/mcp"
    assert "results" not in data
    assert block["arguments"] == tool["arguments"]
    assert block["result"]["data"] == tool["result"]
    assert [(e["type"], e["conv_id"], e["run_id"]) for e in stream[1]] == [("tool", "fixture-chat", "fixture-run"), ("tool_done", "fixture-chat", "fixture-run")]


@pytest.mark.parametrize("item", [{}, {"query": "Codex MCP"}, {"action": [], "content": 123, "sources": "bad", "results": None}])
def test_missing_or_malformed_detail_is_not_empty_search(stream, item):
    tool, _ = emit(stream, item)
    data = json.loads(tool["result"])
    assert "results" not in data and "text" not in data
    assert tool["status"] == "success"


def test_explicit_empty_results_preserved(stream):
    tool, _ = emit(stream, {"query": "absent", "results": []})
    assert json.loads(tool["result"])["results"] == []


@pytest.mark.parametrize("status", ["failed", "error", "cancelled", "canceled"])
def test_terminal_failure_and_cancel_are_not_success(stream, status):
    tool, block = emit(stream, {"status": status, "error": {"message": "Page unavailable"}})
    assert tool["status"] == ("cancelled" if status in ("cancelled", "canceled") else "error")
    assert not block["result"]["ok"] and not stream[1][-1]["success"]
    assert json.loads(tool["result"])["error"] == "Page unavailable"


def test_start_metadata_survives_completion_without_action(stream):
    action = {"type": "find_in_page", "url": "https://example.com", "pattern": "MCP"}
    tool, _ = emit(stream, {"text": "MCP appears here."}, start={"action": action})
    assert tool["arguments"]["action"] == action


def test_text_blocks_and_source_rows_are_preserved_without_malformed_fields(stream):
    tool, _ = emit(stream, {"content": [{"type": "text", "text": "Page summary"}, {"type": "image", "text": "not text"}], "results": [{"title": "Docs", "url": "https://example.com", "snippet": "Summary"}, None], "action": {"type": "open_page", "url": 12}})
    data = json.loads(tool["result"])
    assert data["text"] == "Page summary"
    assert data["results"] == [{"title": "Docs", "url": "https://example.com", "snippet": "Summary"}]
    assert "url" not in tool["arguments"]["action"]


def test_unresolved_cancel_retains_existing_terminal_error_and_detail(stream):
    parser, events = stream
    parser.on_line(json.dumps({"type": "item.started", "item": {"type": "web_search", "id": "web"}}))
    parser.finish_unresolved_tools(cancelled=True)
    assert events[-1]["tool"]["status"] == "error"
    assert events[-1]["tool"]["result"] == "Cancelled before the tool finished."
