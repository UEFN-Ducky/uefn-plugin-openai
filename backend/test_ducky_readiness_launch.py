"""Exercise Ducky readiness at the actual adapter launch boundary."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def launch_env(tmp_path, monkeypatch):
    for key in ("APPDATA", "LOCALAPPDATA", "USERPROFILE", "HOME", "CODEX_HOME"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    here = Path(__file__).resolve().parent
    for parent in here.parents:
        core = parent / "UEFN-Ducky-Release" / "ducky_app"
        if (core / "backend" / "agent").is_dir():
            monkeypatch.syspath_prepend(str(core))
            break
    else:
        pytest.fail("UEFN-Ducky-Release core checkout is required")

    # Pytest may have imported this plugin's backend package during collection.
    for name in list(sys.modules):
        if name == "backend" or name.startswith("backend."):
            monkeypatch.delitem(sys.modules, name)

    # Load the plugin modules without running its registration entrypoint.
    package_name = "_ducky_readiness_openai"
    package = types.ModuleType(package_name)
    package.__path__ = [str(here)]
    monkeypatch.setitem(sys.modules, package_name, package)
    adapter = importlib.import_module(f"{package_name}.codex_adapter")
    updater = importlib.import_module(f"{package_name}.cli_update")
    monkeypatch.setattr(updater, "resolve_bin", lambda _: "codex.exe")
    update = Mock(side_effect=AssertionError("Readiness must not trigger CLI installation"))
    monkeypatch.setattr(updater, "update_cli", update)
    monkeypatch.setattr(adapter, "codex_extra_dirs", lambda _: [])
    process = Mock()
    monkeypatch.setattr(adapter, "run_streaming_process", process)
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"uefn": {
        "command": "node", "args": ["bridge.mjs"],
    }}}), encoding="utf-8")
    events = []
    kwargs = dict(
        prompt="Inspect tools", system_prompt="System context", cwd=str(tmp_path),
        conv_id="test-conversation", run_id="test-run", model="auto",
        mcp_config_path=str(config), extra_args="", cli_path="codex.exe",
        env={}, push=events.append,
    )
    yield adapter, process, update, config, events, kwargs
    for name in list(sys.modules):
        if name.startswith(package_name + "."):
            del sys.modules[name]


@pytest.mark.parametrize("session_id", ["", "existing-thread"])
@pytest.mark.parametrize("failure", [
    "empty_path", "missing_file", "invalid_json", "missing_server",
    "missing_command", "blocked_directory", "blocked_config",
])
def test_ducky_readiness_failed_profile_never_launches(launch_env, tmp_path, session_id, failure):
    adapter, process, update, config, events, kwargs = launch_env
    if failure == "empty_path":
        kwargs["mcp_config_path"] = ""
    elif failure == "missing_file":
        config.unlink()
    elif failure == "invalid_json":
        config.write_text("{broken", encoding="utf-8")
    elif failure == "missing_server":
        config.write_text('{"mcpServers": {}}', encoding="utf-8")
    elif failure == "missing_command":
        config.write_text('{"mcpServers": {"uefn": {}}}', encoding="utf-8")
    elif failure == "blocked_directory":
        (tmp_path / ".codex").write_text("not a directory", encoding="utf-8")
    elif failure == "blocked_config":
        (tmp_path / ".codex" / "config.toml").mkdir(parents=True)
    process.side_effect = AssertionError("Agent launched without a written MCP profile")

    result = adapter.CodexAdapter().launch(**kwargs, session_id=session_id)

    assert not result.ok
    assert result.status == "error"
    assert result.error.startswith("Ducky tools unavailable: ")
    assert "profile" in result.error.lower()
    assert result.upstream_session_id == session_id
    process.assert_not_called()
    update.assert_not_called()
    assert not any("Starting Codex" in event.get("text", "") for event in events)


@pytest.mark.parametrize("session_id", ["", "existing-thread"])
@pytest.mark.parametrize("mcp_failure", [False, True])
def test_ducky_readiness_normal_launch_keeps_mcp_required(launch_env, tmp_path, session_id, mcp_failure):
    adapter, process, update, config, events, kwargs = launch_env

    def run(**call):
        # The profile is written before the process can be created.
        profile = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
        assert "[mcp_servers.uefn]" in profile
        assert "startup_timeout_sec = 60.0" in profile
        assert "required" not in profile  # Requirement belongs to Ducky turns only.
        argv = call["argv"]
        assert "mcp_servers.uefn.required=true" in argv
        assert "tools.mcp__uefn__" in argv[-1]
        assert (argv[2:4] == ["resume", session_id]) if session_id else "resume" not in argv
        if not mcp_failure:
            call["on_line"](json.dumps({"type": "item.completed", "item": {
                "id": "reply", "type": "agent_message", "text": "Tools available",
            }}))
        return types.SimpleNamespace(
            cancelled=False, timed_out=False, returncode=1 if mcp_failure else 0,
            stderr_tail="Required MCP server uefn failed to initialize" if mcp_failure else "",
            raw_tail="",
        )

    process.side_effect = run
    result = adapter.CodexAdapter().launch(**kwargs, session_id=session_id)

    assert result.ok is not mcp_failure
    if mcp_failure:
        assert "Required MCP server uefn" in result.error
    else:
        assert result.reply_text == "Tools available"
    # In particular, an MCP failure cannot retry as a shell-only turn.
    process.assert_called_once()
    update.assert_not_called()
