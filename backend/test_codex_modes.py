"""Native sandbox configuration; MCP write authorization is a separate boundary."""
from __future__ import annotations

import importlib
import inspect
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
        pytest.fail("Core checkout is required")
    for name in list(sys.modules):
        if name == "backend" or name.startswith("backend."):
            monkeypatch.delitem(sys.modules, name)
    name = "_ducky_codex_modes"
    package = types.ModuleType(name)
    package.__path__ = [str(here)]
    monkeypatch.setitem(sys.modules, name, package)
    adapter = importlib.import_module(name + ".codex_adapter")
    updater = importlib.import_module(name + ".cli_update")
    monkeypatch.setattr(updater, "resolve_bin", Mock(return_value="codex.exe"))
    monkeypatch.setattr(updater, "update_cli", Mock(side_effect=AssertionError("Unexpected install")))
    monkeypatch.setattr(adapter, "codex_extra_dirs", lambda _: [tmp_path / "other-project"])
    calls, events = [], []

    def run(**call):
        calls.append(call)
        call["on_line"](json.dumps({"type": "thread.started", "thread_id": "thread-one"}))
        call["on_line"](json.dumps({"type": "item.completed", "item": {
            "id": "reply", "type": "agent_message", "text": "Done",
        }}))
        return types.SimpleNamespace(cancelled=False, timed_out=False, returncode=0,
                                     stderr_tail="", raw_tail="")

    monkeypatch.setattr(adapter, "run_streaming_process", Mock(side_effect=run))
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"uefn": {
        "command": "node", "args": ["bridge.mjs"],
    }}}), encoding="utf-8")
    kwargs = dict(prompt="Inspect", system_prompt="System", cwd=str(tmp_path),
                  conv_id="conversation", run_id="run", model="auto",
                  mcp_config_path=str(config), extra_args="--full-auto",
                  cli_path="codex.exe", env={}, push=events.append)
    yield adapter, updater, kwargs, calls, events
    for module in list(sys.modules):
        if module.startswith(name + "."):
            del sys.modules[module]


def assert_mode_argv(argv, mode, session_id):
    assert "mcp_servers.uefn.required=true" in argv
    assert 'mcp_servers.uefn.default_tools_approval_mode="approve"' in argv
    assert "tools.mcp__uefn__" in argv[-1]
    assert "--full-auto" not in argv
    if mode == "agent":
        assert "--dangerously-bypass-approvals-and-sandbox" in argv
    else:
        assert argv[1:4] == ["exec", "--sandbox", "read-only"]
        assert "--dangerously-bypass-approvals-and-sandbox" not in argv
        assert "--approve-for-me" not in argv
        assert "--add-dir" not in argv
        assert not any("workspace-write" in s or "writable_roots" in s for s in argv[:-1])
        assert 'approval_policy="never"' in argv
    if session_id:
        index = argv.index("resume")
        assert argv[index + 1] == session_id
        assert "--sandbox" not in argv[index:]
    else:
        assert "resume" not in argv


@pytest.mark.parametrize("mode", ["ask", "plan", "agent"])
@pytest.mark.parametrize("session_id", ["", "thread-one"])
def test_launch_modes(launch_env, mode, session_id):
    adapter, updater, kwargs, calls, events = launch_env
    result = adapter.CodexAdapter().launch(**kwargs, mode=mode, session_id=session_id)
    assert result.ok
    assert result.requested_mode == result.effective_mode == mode
    assert result.upstream_session_id == "thread-one"
    assert len(calls) == 1
    assert_mode_argv(calls[0]["argv"], mode, session_id)
    assert events and all(e["requested_mode"] == mode for e in events)
    # Streaming progress does not yet confirm the final CLI outcome.
    assert all(e["effective_mode"] == "" for e in events)
    updater.update_cli.assert_not_called()


def test_core_contract_and_mode_changes(launch_env):
    adapter, _, kwargs, calls, _ = launch_env
    instance = adapter.CodexAdapter()
    assert set(instance.capabilities.supported_modes) == {"ask", "plan", "agent"}
    assert "mode" in inspect.signature(instance.launch).parameters
    for mode in ("agent", "ask", "plan", "agent", "ask"):
        result = instance.launch(**kwargs, mode=mode, session_id="same-thread")
        assert result.ok and result.effective_mode == mode
        assert_mode_argv(calls[-1]["argv"], mode, "same-thread")


@pytest.mark.parametrize("extra", [
    "--dangerously-bypass-approvals-and-sandbox", "--approve-for-me", "--sandbox=workspace-write",
    "-s danger-full-access", "-sworkspace-write", "--config=sandbox_mode=workspace-write",
    '-c sandbox_mode="danger-full-access"', "-csandbox_mode=workspace-write",
    '--config default_permissions=":danger-full-access"', "--profile=write", "-pwrite",
    "--add-dir=C:/", "--ignore-user-config", "--disable sandbox", "--",
    "--output-last-message=out.txt", "--worktree", "--full-auto --bad-option",
    "-c mcp_servers.uefn.required=false", '-c mcp_servers.uefn.enabled=false', '"unclosed',
])
@pytest.mark.parametrize("mode", ["ask", "plan"])
@pytest.mark.parametrize("session_id", ["", "thread-one"])
def test_restricted_overrides_fail_before_side_effects(launch_env, monkeypatch, extra, mode, session_id):
    adapter, updater, kwargs, calls, events = launch_env
    writer = Mock(side_effect=AssertionError("Unexpected config write"))
    monkeypatch.setattr(adapter, "write_codex_uefn_profile", writer)
    kwargs["extra_args"] = extra
    result = adapter.CodexAdapter().launch(**kwargs, mode=mode, session_id=session_id)
    assert not result.ok and result.status == "error"
    assert result.requested_mode == mode and result.effective_mode == ""
    assert result.error == "Codex mode or extra arguments are unsupported for a restricted launch."
    assert result.upstream_session_id == session_id
    assert not calls and not events
    writer.assert_not_called()
    updater.resolve_bin.assert_not_called()
    updater.update_cli.assert_not_called()


@pytest.mark.parametrize("mode", ["invalid", "review", "read-only"])
def test_invalid_mode_never_launches(launch_env, monkeypatch, mode):
    adapter, updater, kwargs, calls, events = launch_env
    writer = Mock()
    monkeypatch.setattr(adapter, "write_codex_uefn_profile", writer)
    result = adapter.CodexAdapter().launch(**kwargs, mode=mode)
    assert not result.ok and result.effective_mode == ""
    assert result.requested_mode == mode
    writer.assert_not_called()
    updater.resolve_bin.assert_not_called()
    assert not calls and not events


@pytest.mark.parametrize("mode", ["ask", "plan", "agent"])
@pytest.mark.parametrize("session_id", ["", "thread-one"])
@pytest.mark.parametrize("failure", ["readiness", "option", "mcp", "cancelled", "timeout"])
def test_failed_launch_does_not_confirm_mode(launch_env, monkeypatch, mode, session_id, failure):
    adapter, updater, kwargs, calls, events = launch_env
    if failure == "readiness":
        monkeypatch.setattr(adapter, "write_codex_uefn_profile", Mock(side_effect=RuntimeError("PRIVATE")))
    else:
        errors = {"option": "unexpected argument '--sandbox' found",
                  "mcp": "Required MCP server uefn failed to initialize"}
        adapter.run_streaming_process.side_effect = None
        adapter.run_streaming_process.return_value = types.SimpleNamespace(
            cancelled=failure == "cancelled", timed_out=failure == "timeout",
            returncode=1, stderr_tail=errors.get(failure, ""), raw_tail="")
    result = adapter.CodexAdapter().launch(**kwargs, mode=mode, session_id=session_id)
    assert not result.ok and result.effective_mode == "" and result.requested_mode == mode
    assert "PRIVATE" not in repr(result) + repr(events)
    if failure == "readiness":
        assert result.error.startswith("Ducky tools unavailable:")
        adapter.run_streaming_process.assert_not_called()
        assert not events
    else:
        # An old CLI rejecting an option must never retry with weaker flags.
        adapter.run_streaming_process.assert_called_once()
    updater.update_cli.assert_not_called()


@pytest.mark.parametrize("mode", ["ask", "plan", "agent"])
@pytest.mark.parametrize("session_id", ["", "thread-one"])
def test_heal_retry_preserves_mode_and_required_server(launch_env, monkeypatch, mode, session_id):
    adapter, updater, kwargs, calls, events = launch_env
    first = types.SimpleNamespace(cancelled=False, timed_out=False, returncode=1,
                                  stderr_tail="outdated executable", raw_tail="")
    run_ok = adapter.run_streaming_process.side_effect
    captured = []
    def run(**call):
        captured.append(list(call["argv"]))
        return first if len(captured) == 1 else run_ok(**call)
    adapter.run_streaming_process.side_effect = run
    monkeypatch.setattr(updater, "should_heal_launch", lambda *args: True)
    updater.update_cli.side_effect = None
    updater.update_cli.return_value = {"ok": True}
    result = adapter.CodexAdapter().launch(**kwargs, mode=mode, session_id=session_id)
    assert result.ok and result.effective_mode == mode
    assert len(captured) == 2 and captured[0] == captured[1]
    for argv in captured:
        assert_mode_argv(argv, mode, session_id)


@pytest.mark.parametrize("mode", ["ask", "plan"])
def test_internal_flags_cannot_restore_writes(launch_env, mode):
    adapter, _, _, _, _ = launch_env
    with pytest.raises(ValueError, match="extra flags"):
        adapter.build_codex_argv(binary="codex", prompt="go", model="auto", extra_args="",
                                 session_id="thread", mode=mode,
                                 extra_flags=adapter.ducky_launch_flags("agent"))


@pytest.mark.parametrize("mode", ["ask", "plan"])
@pytest.mark.parametrize("session_id", ["", "thread-one"])
def test_partial_reply_with_failed_process_cannot_confirm_restriction(launch_env, mode, session_id):
    adapter, updater, kwargs, calls, events = launch_env
    original = adapter.run_streaming_process.side_effect
    def run(**call):
        proc = original(**call)
        proc.returncode = 1
        proc.stderr_tail = "sandbox initialization failed"
        return proc
    adapter.run_streaming_process.side_effect = run
    result = adapter.CodexAdapter().launch(**kwargs, mode=mode, session_id=session_id)
    assert not result.ok
    assert result.status == "error" and result.effective_mode == ""
    assert result.reply_text == "Done"
    updater.update_cli.assert_not_called()


@pytest.mark.parametrize("mode", ["ask", "plan"])
@pytest.mark.parametrize("session_id", ["", "thread-one"])
def test_restricted_ignores_saved_execution_allow_rules(launch_env, tmp_path, mode, session_id):
    adapter, _, kwargs, calls, _ = launch_env
    rules = tmp_path / ".codex" / "rules"
    rules.mkdir(parents=True)
    (rules / "default.rules").write_text(
        'prefix_rule(pattern=["powershell"], decision="allow")', encoding="utf-8")
    result = adapter.CodexAdapter().launch(**kwargs, mode=mode, session_id=session_id)
    assert result.ok
    assert "--ignore-rules" in calls[0]["argv"]


@pytest.mark.parametrize("mode", ["ask", "plan"])
def test_resume_id_cannot_become_a_bypass_option(launch_env, mode):
    adapter, updater, kwargs, calls, events = launch_env
    result = adapter.CodexAdapter().launch(
        **kwargs, mode=mode, session_id="--dangerously-bypass-approvals-and-sandbox")
    assert not result.ok and result.effective_mode == ""
    assert not calls and not events
    updater.resolve_bin.assert_not_called()
