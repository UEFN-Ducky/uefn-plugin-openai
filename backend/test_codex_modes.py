"""Read-only Ask/Plan launches and compatibility with older Agent-only hosts."""
from __future__ import annotations

import importlib
import inspect
import json
import sys
import threading
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("mode", ["ask", "plan", "agent"])
@pytest.mark.parametrize("session", ["", "thread-one"])
@pytest.mark.parametrize("ending", ["cancelled", "timeout"])
@pytest.mark.parametrize("channel", ["stderr_tail", "raw_tail"])
def test_terminal_repair_marker_never_replays(launch_env, monkeypatch, mode, session, ending, channel):
    module, updater, kwargs, calls, events = launch_env
    run = module.run_streaming_process.side_effect
    def terminal(**kw):
        kw["on_line"](json.dumps({"type": "item.completed", "item": {
            "id": "inspect", "type": "command_execution", "command": "read fixture",
            "aggregated_output": "Partial inspection", "exit_code": 0,
        }}))
        proc = run(**kw)  # Preserve partial reply and the upstream session.
        proc.returncode = 1
        proc.cancelled = ending == "cancelled"
        proc.timed_out = ending == "timeout"
        setattr(proc, channel, "no such file or directory")
        return proc
    module.run_streaming_process.side_effect = terminal
    classifier = Mock(wraps=updater.should_heal_launch)
    monkeypatch.setattr(updater, "should_heal_launch", classifier)
    updater.update_cli.side_effect = None
    updater.update_cli.return_value = {"ok": True}
    result = module.CodexAdapter().launch(**kwargs, mode=mode, session_id=session)
    assert (updater.update_cli.call_count, module.run_streaming_process.call_count) == (0, 1)
    classifier.assert_not_called()
    assert not result.ok and result.status == ending
    assert result.reply_text == "Done" and result.blocks
    assert result.upstream_session_id == "thread-one"
    assert result.requested_mode == mode and result.effective_mode == ""
    assert not any("updating automatically" in e.get("text", "") for e in events)


@pytest.mark.parametrize("mode", ["ask", "plan", "agent"])
@pytest.mark.parametrize("session", ["", "thread-one"])
def test_real_nonterminal_repair_classifier_retains_flags(launch_env, mode, session):
    module, updater, kwargs, calls, events = launch_env
    run = module.run_streaming_process.side_effect
    first = types.SimpleNamespace(cancelled=False, timed_out=False, returncode=1,
                                  stderr_tail="no such file or directory", raw_tail="")
    attempts = []
    def process(**kw):
        attempts.append(kw["argv"].copy())
        return first if len(attempts) == 1 else run(**kw)
    module.run_streaming_process.side_effect = process
    updater.update_cli.side_effect = None
    updater.update_cli.return_value = {"ok": True}
    result = module.CodexAdapter().launch(**kwargs, mode=mode, session_id=session)
    assert result.ok and result.effective_mode == mode
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    updater.update_cli.assert_called_once()
    for argv in attempts:
        assert_mode_argv(argv, mode, session)


@pytest.mark.parametrize("missing", [True, False])
@pytest.mark.parametrize("instance", ["skip_env", "core_port", "older_core_port"])
def test_second_instance_heal_never_installs_codex(launch_env, monkeypatch, missing, instance):
    """A heal in a sandboxed build (DUCKY_SKIP_CLI_UPDATE=1 or panel port 4300)
    reports Codex missing here and never runs the installer, which would add the
    sandbox's LOCALAPPDATA bin dir to the person's real user PATH."""
    module, updater, kwargs, _calls, events = launch_env
    install = Mock(side_effect=AssertionError("Codex installer ran"))
    monkeypatch.setattr(updater, "_install_cli", install)
    monkeypatch.setattr(updater, "_run", Mock(side_effect=AssertionError("codex update ran")))
    updater.update_cli.side_effect = None  # the real updater, should anything reach it
    if instance == "skip_env":
        monkeypatch.setenv("DUCKY_SKIP_CLI_UPDATE", "1")
        reason = "DUCKY_SKIP_CLI_UPDATE is set"
    else:
        reason = "panel port 4300 is not the default 4200"
        monkeypatch.setenv("UEFN_DUCKY_PANEL_PORT", "4300")
        if instance == "core_port":
            sys.modules["frontend.ide_apply"].auto_apply_skip_reason = lambda: reason
        else:
            monkeypatch.setitem(sys.modules, "frontend.ide_apply", None)  # core without the check
    if missing:
        monkeypatch.setattr(updater, "resolve_bin", Mock(return_value=""))
    else:
        module.run_streaming_process.side_effect = lambda **kw: types.SimpleNamespace(
            cancelled=False, timed_out=False, returncode=1,
            stderr_tail="no such file or directory", raw_tail="")
    result = module.CodexAdapter().launch(**kwargs, mode="agent", session_id="thread-one")
    install.assert_not_called()
    updater._run.assert_not_called()
    assert module.run_streaming_process.call_count == (0 if missing else 1)
    assert not result.ok and f"in this Ducky instance ({reason})" in result.error
    assert not any("automatically" in e.get("text", "") for e in events)


@pytest.mark.parametrize("boundary", ["process", "classifier", "status", "updater"])
def test_cancellation_arriving_at_retry_boundary(launch_env, monkeypatch, boundary):
    module, updater, kwargs, calls, events = launch_env
    cancel = threading.Event()
    def process(**kw):
        if boundary == "process":
            cancel.set()
        return types.SimpleNamespace(cancelled=False, timed_out=False, returncode=1,
                                     stderr_tail="no such file or directory", raw_tail="")
    module.run_streaming_process.side_effect = process
    real_classifier = updater.should_heal_launch
    def classifier(*chunks):
        if boundary == "classifier":
            cancel.set()
        return real_classifier(*chunks)
    monkeypatch.setattr(updater, "should_heal_launch", classifier)
    def update(*args):
        if boundary == "updater":
            cancel.set()
        return {"ok": True}
    updater.update_cli.side_effect = update
    def push(event):
        events.append(event)
        if boundary == "status" and "updating automatically" in event.get("text", ""):
            cancel.set()
    kwargs["push"] = push
    result = module.CodexAdapter().launch(**kwargs, mode="agent", session_id="thread-one", cancel=cancel)
    assert module.run_streaming_process.call_count == 1
    assert updater.update_cli.call_count == (1 if boundary == "updater" else 0)
    assert not result.ok and result.status == "error"  # Original finalized result.
    assert result.upstream_session_id == "thread-one"
    assert result.requested_mode == "agent" and result.effective_mode == ""


@pytest.mark.parametrize("session", ["", "thread-one"])
@pytest.mark.parametrize("ending", ["cancelled", "timeout"])
def test_historical_host_terminal_result_never_replays(launch_env, monkeypatch, session, ending):
    with registered_profile(launch_env, monkeypatch, "legacy") as (module, agent, kw, calls, events, base):
        run = module.run_streaming_process.side_effect
        def terminal(**kwargs):
            kwargs["on_line"](json.dumps({"type": "item.completed", "item": {
                "id": "inspect", "type": "command_execution", "command": "read fixture",
                "aggregated_output": "Partial inspection", "exit_code": 0,
            }}))
            proc = run(**kwargs)
            proc.cancelled = ending == "cancelled"
            proc.timed_out = ending == "timeout"
            proc.returncode = 1
            proc.raw_tail = "no such file or directory"
            return proc
        module.run_streaming_process.side_effect = terminal
        updater = importlib.import_module(module.__package__ + ".cli_update")
        updater.update_cli.side_effect = None
        updater.update_cli.return_value = {"ok": True}
        result = agent.launch(**kw, session_id=session)
        assert (updater.update_cli.call_count, module.run_streaming_process.call_count) == (0, 1)
        assert isinstance(result, base.CodingAgentLaunchResult)
        assert not result.ok and result.status == ending
        assert result.reply_text == "Done" and result.blocks
        assert result.upstream_session_id == "thread-one"


@contextmanager
def registered_profile(launch_env, monkeypatch, profile):
    """Load the actual package entrypoint through the host, with an old API."""
    adapter, updater, kwargs, calls, events = launch_env
    from backend.agent.coding_agents import base, cli_shared
    from backend.uefn_plugins import host

    before = dict(sys.modules)
    try:
        with monkeypatch.context() as patch:
            legacy = types.ModuleType("_codex_legacy_base")
            patch.setitem(sys.modules, legacy.__name__, legacy)
            exec("from dataclasses import dataclass, field\nfrom typing import Any\n" +
                 _LEGACY_BASE_CLASSES, legacy.__dict__)
            if profile in ("legacy", "missing_helper"):
                patch.delattr(base, "normalize_coding_mode")
            if profile in ("legacy", "old_capabilities"):
                patch.setattr(base, "CodingAgentCapabilities", legacy.CodingAgentCapabilities)
            if profile in ("legacy", "old_result"):
                patch.setattr(base, "CodingAgentLaunchResult", legacy.CodingAgentLaunchResult)
                patch.setattr(cli_shared, "CodingAgentLaunchResult", legacy.CodingAgentLaunchResult)
            if profile == "legacy":
                patch.setattr(base, "CodingAgentInfo", legacy.CodingAgentInfo)
            package = host._import_backend("openai_compat_test", Path(__file__).parents[1], "backend")
            loaded = importlib.import_module(package.__name__ + ".codex_adapter")
            cli = importlib.import_module(package.__name__ + ".cli_update")
            patch.setattr(loaded, "run_streaming_process", adapter.run_streaming_process)
            patch.setattr(loaded, "codex_extra_dirs", lambda _: [])
            patch.setattr(loaded, "heal_codex_approval_policy", Mock())
            patch.setattr(cli, "schedule_cli_update_on_plugin_load", Mock())
            patch.setattr(cli, "resolve_bin", updater.resolve_bin)
            patch.setattr(cli, "update_cli", updater.update_cli)
            patch.setattr(package, "_heal_default_model_if_codex_only", Mock())
            api = Mock()
            package.register(api)
            assert api.register_coding_agent.call_args.args == ("codex",)
            instance = api.register_coding_agent.call_args.kwargs["factory"]()
            yield loaded, instance, kwargs, calls, events, base
    finally:
        # Remove all imports introduced by registration, restore previous entries.
        for name in set(sys.modules) - set(before):
            sys.modules.pop(name, None)
        sys.modules.update(before)


@pytest.mark.parametrize("profile", ["legacy", "missing_helper", "old_capabilities", "old_result", "current"])
@pytest.mark.parametrize("session", ["", "thread-one"])
def test_registered_host_profiles_agent(launch_env, monkeypatch, profile, session):
    with registered_profile(launch_env, monkeypatch, profile) as (module, agent, kw, calls, events, base):
        assert isinstance(agent.capabilities, base.CodingAgentCapabilities)
        assert agent.capabilities.resume and agent.capabilities.mcp_inject
        result = agent.launch(**kw, session_id=session)  # Old runner supplies no mode.
        assert isinstance(result, base.CodingAgentLaunchResult)
        assert result.ok and result.reply_text == "Done"
        assert result.upstream_session_id == "thread-one"
        assert_mode_argv(calls[0]["argv"], "agent", session)
        if profile not in ("legacy", "old_result"):
            assert result.requested_mode == result.effective_mode == "agent"
            assert result.to_dict()["effective_mode"] == "agent"
        else:
            assert "effective_mode" not in result.to_dict()


@pytest.mark.parametrize("profile", ["legacy", "missing_helper", "old_capabilities", "old_result"])
@pytest.mark.parametrize("mode", ["ask", "plan", "invalid"])
def test_incomplete_host_contract_refuses_restriction(launch_env, monkeypatch, profile, mode):
    with registered_profile(launch_env, monkeypatch, profile) as (module, agent, kw, calls, events, base):
        writer = Mock(side_effect=AssertionError("profile write"))
        monkeypatch.setattr(module, "write_codex_uefn_profile", writer)
        result = agent.launch(**kw, mode=mode, session_id="thread-one")
        assert not result.ok and result.status == "error"
        assert result.upstream_session_id == "thread-one"
        if mode in ("ask", "plan"):
            assert "MCP read-only enforcement is not supported" in result.error
        assert getattr(agent.capabilities, "supported_modes", ("agent",)) == ("agent",)
        assert not calls and not events
        writer.assert_not_called()
        launch_env[1].resolve_bin.assert_not_called()
        launch_env[1].update_cli.assert_not_called()


@pytest.mark.parametrize("session", ["", "thread-one"])
def test_legacy_process_failure_preserves_result(launch_env, monkeypatch, session):
    with registered_profile(launch_env, monkeypatch, "legacy") as (module, agent, kw, calls, events, base):
        module.run_streaming_process.side_effect = None
        module.run_streaming_process.return_value = types.SimpleNamespace(
            cancelled=False, timed_out=False, returncode=1, stderr_tail="synthetic failure", raw_tail="")
        result = agent.launch(**kw, session_id=session)
        assert isinstance(result, base.CodingAgentLaunchResult)
        assert not result.ok and result.status == "error"
        assert result.upstream_session_id == session
        module.run_streaming_process.assert_called_once()
        launch_env[1].update_cli.assert_not_called()


def test_unrelated_missing_base_import_is_not_hidden(launch_env, monkeypatch):
    from backend.agent.coding_agents import base
    monkeypatch.delattr(base, "CodingAgentInfo")
    with pytest.raises(ImportError, match="CodingAgentInfo"):
        with registered_profile(launch_env, monkeypatch, "missing_helper"):
            pytest.fail("Broken mandatory import must fail")


@pytest.mark.parametrize("profile", ["legacy", "missing_helper", "old_capabilities", "old_result"])
@pytest.mark.parametrize("failure", ["config", "profile"])
@pytest.mark.parametrize("session", ["", "thread-one"])
def test_older_host_readiness_failure_retains_session(launch_env, monkeypatch, profile, failure, session):
    with registered_profile(launch_env, monkeypatch, profile) as (module, agent, kw, calls, events, base):
        if failure == "config":
            Path(kw["mcp_config_path"]).write_text("not-json", encoding="utf-8")
        else:
            monkeypatch.setattr(module, "write_codex_uefn_profile", Mock(side_effect=OSError("PRIVATE")))
        result = agent.launch(**kw, session_id=session)
        assert isinstance(result, base.CodingAgentLaunchResult)
        assert not result.ok and result.status == "error"
        assert result.error.startswith("Ducky tools unavailable:") and "PRIVATE" not in result.error
        assert result.upstream_session_id == session
        assert not calls and not events
        launch_env[1].resolve_bin.assert_not_called()


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
    # side_effect=None with no return_value runs the real updater.
    monkeypatch.setattr(updater, "update_cli", Mock(
        wraps=updater.update_cli, side_effect=AssertionError("Unexpected install")))
    # conftest's panel port and tmp_path as USERPROFILE/LOCALAPPDATA read as a
    # sandbox to core; these tests run the heal unless they say otherwise.
    core_check = types.ModuleType("frontend.ide_apply")
    core_check.auto_apply_skip_reason = lambda: ""
    monkeypatch.setitem(sys.modules, "frontend.ide_apply", core_check)
    monkeypatch.delenv("DUCKY_SKIP_CLI_UPDATE", raising=False)
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
    # Every Ducky turn, any mode: Codex's own shell, helpers and web are off.
    for flag in ("features.shell_tool=false", "features.unified_exec=false",
                 "features.multi_agent=false", 'web_search="disabled"'):
        assert argv[argv.index(flag) - 1] == "-c"
    assert "never apply_patch" in argv[-1]
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
        assert '--ignore-rules' in argv
        assert 'do not change project files' in argv[-1]
        if mode == 'plan':
            assert 'ducky_create_plan' in argv[-1]
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
    assert instance.capabilities.supported_modes == ("agent", "ask", "plan")
    assert "mode" in inspect.signature(instance.launch).parameters
    for mode in ("agent", "ask", "plan", "agent", "ask"):
        previous_calls = len(calls)
        result = instance.launch(**kwargs, mode=mode, session_id="same-thread")
        assert result.ok and result.effective_mode == mode
        assert len(calls) == previous_calls + 1
        assert_mode_argv(calls[-1]["argv"], mode, "same-thread")


@pytest.mark.parametrize("mode", ["agent", "ask", "plan"])
@pytest.mark.parametrize("session_id", ["", "thread-one"])
def test_a_long_brief_goes_to_codex_on_stdin_never_a_file_it_must_read(launch_env, mode, session_id):
    # Ask/Plan's read-only sandbox blocks the shell read a temp file needed, so Codex
    # never saw its instructions (Oct 10 live test: "the shell policy blocked the read").
    adapter, _, kwargs, calls, _ = launch_env
    long_message = "Pasted log line. " * 2000 + "Inspect the project"
    kwargs.update(prompt=long_message)
    result = adapter.CodexAdapter().launch(**kwargs, mode=mode, session_id=session_id)
    assert result.ok and result.effective_mode == mode
    call = calls[-1]
    assert call["argv"][-1] == "-"
    assert call["stdin_data"].endswith(long_message)
    assert "Get-Content" not in " ".join(call["argv"])


def test_a_short_message_stays_on_the_command_line(launch_env):
    adapter, _, kwargs, calls, _ = launch_env
    result = adapter.CodexAdapter().launch(**kwargs, mode="ask", session_id="thread-one")
    assert result.ok
    assert calls[-1]["stdin_data"] is None
    assert calls[-1]["argv"][-1].endswith("Inspect")


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
    assert "unsupported" in result.error
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
    assert not result.ok and result.effective_mode == ""
    assert "restricted mode was not confirmed" in result.error


@pytest.mark.parametrize("mode", ["ask", "plan"])
@pytest.mark.parametrize("session_id", ["", "thread-one"])
def test_restricted_ignores_saved_execution_allow_rules(launch_env, tmp_path, mode, session_id):
    adapter, _, kwargs, calls, _ = launch_env
    rules = tmp_path / ".codex" / "rules"
    rules.mkdir(parents=True)
    (rules / "default.rules").write_text(
        'prefix_rule(pattern=["powershell"], decision="allow")', encoding="utf-8")
    result = adapter.CodexAdapter().launch(**kwargs, mode=mode, session_id=session_id)
    assert result.ok and result.effective_mode == mode
    assert_mode_argv(calls[0]["argv"], mode, session_id)
    assert "--ignore-rules" in calls[0]["argv"]


@pytest.mark.parametrize("mode", ["ask", "plan"])
def test_resume_id_cannot_become_a_bypass_option(launch_env, mode):
    adapter, updater, kwargs, calls, events = launch_env
    result = adapter.CodexAdapter().launch(
        **kwargs, mode=mode, session_id="--dangerously-bypass-approvals-and-sandbox")
    assert not result.ok and result.effective_mode == ""
    assert not calls and not events
    updater.resolve_bin.assert_not_called()


# Frozen class definitions from app 1.2.361, f2094f09307472a19fdd1edf762065649aa734dc.
# Taken from backend/agent/coding_agents/base.py, not inferred from the new API.
_LEGACY_BASE_CLASSES = r'''
@dataclass(frozen=True)
class CodingAgentCapabilities:
    terminal_agent: bool = False
    chat_api: bool = False
    a2a: bool = True
    mcp_inject: bool = False
    needs_api_key: bool = False
    needs_cli: bool = False
    resume: bool = False
    'Plugin-owned: True if this adapter resumes an upstream session.\n\n    Core only stores the id the plugin returns and passes it back on\n    launch(). Vendor flags (--resume, Agent.resume, exec resume, …)\n    live in the plugin, not here.\n    '

@dataclass
class CodingAgentInfo:
    id: str
    label: str
    enabled: bool
    available: bool
    status: str
    cli_path: str = ''
    default_args: str = ''
    capabilities: CodingAgentCapabilities = field(default_factory=CodingAgentCapabilities)
    models: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        install_help = ''
        plugin_id = ''
        shows_thinking = False
        can_login = False
        can_logout = False
        try:
            from backend.uefn_plugins.host import get_coding_agent_registration
            reg = get_coding_agent_registration(self.id) or {}
            install_help = str(reg.get('install_help') or '')
            plugin_id = str(reg.get('plugin_id') or '')
            shows_thinking = bool(reg.get('shows_thinking_effort') or callable(reg.get('thinking_env')))
            can_login = callable(reg.get('login'))
            can_logout = callable(reg.get('logout'))
        except Exception:
            pass
        return {'id': self.id, 'label': self.label, 'enabled': self.enabled, 'available': self.available, 'status': self.status, 'cli_path': self.cli_path, 'default_args': self.default_args, 'install_help': install_help, 'shows_thinking_effort': shows_thinking, 'plugin_id': plugin_id, 'logged_in': getattr(self, 'logged_in', None), 'can_login': can_login, 'can_logout': can_logout, 'capabilities': {'terminal_agent': self.capabilities.terminal_agent, 'chat_api': self.capabilities.chat_api, 'a2a': self.capabilities.a2a, 'mcp_inject': self.capabilities.mcp_inject, 'needs_api_key': self.capabilities.needs_api_key, 'needs_cli': self.capabilities.needs_cli}, 'models': list(self.models)}

@dataclass
class CodingAgentLaunchResult:
    ok: bool
    terminal_session_id: str = ''
    upstream_session_id: str = ''
    output_tail: str = ''
    reply_text: str = ''
    error: str = ''
    status: str = ''
    streamed: bool = False
    'True when the adapter already pushed text deltas live (no re-stream).'
    usage: dict[str, Any] = field(default_factory=dict)
    "Real usage from the CLI's result event: input/output/cache tokens,\n    cost_usd, num_turns, model, context_tokens (window used this turn)."
    blocks: list[dict[str, Any]] = field(default_factory=list)
    "Ordered thinking/text/tool_call blocks (embedded-agent format) so the\n    turn's steps survive a panel reload, not just the final reply text."

    def to_dict(self) -> dict[str, Any]:
        return {'ok': self.ok, 'terminal_session_id': self.terminal_session_id, 'upstream_session_id': self.upstream_session_id, 'output_tail': self.output_tail, 'reply_text': self.reply_text, 'error': self.error, 'status': self.status, 'usage': dict(self.usage), 'blocks': list(self.blocks)}
'''


@pytest.mark.parametrize("thread,inp,cached,out", [("thread-one", 100, 60, 5), ("thread-two", 500, 400, 30), ("thread-one", 0, 0, 0)])
def test_usage_is_cumulative_and_cache_is_not_double_counted(launch_env, thread, inp, cached, out):
    module, *_ = launch_env
    state = module._CodexStream("test", "run", lambda event: None)
    state.on_line(json.dumps({"type": "thread.started", "thread_id": thread}))
    state.on_line(json.dumps({"type": "turn.completed", "usage": {
        "input_tokens": inp, "cached_input_tokens": cached, "output_tokens": out}}))
    assert state.usage["cumulative_thread"] == thread
    assert state.usage["input_tokens"] == inp - cached
    assert state.usage["cache_read_tokens"] == cached
    assert state.usage["context_tokens"] == inp
    assert state.usage["output_tokens"] == out


@pytest.mark.parametrize("exit_code", [0, 1, 2])
def test_command_exit_code_survives_live_event_and_checkpoint(launch_env, exit_code):
    module, _, _, _, events = launch_env
    stream = module._CodexStream("chat", "run", events.append)
    stream.on_line(json.dumps({"type": "item.completed", "item": {
        "id": "tests", "type": "command_execution", "command": "pytest",
        "status": "completed", "exit_code": exit_code, "aggregated_output": "test output",
    }}))
    event = next(e for e in events if e["type"] == "tool_done")
    assert event["tool"]["status"] == ("error" if exit_code else "success")
    assert event["tool"]["result"].startswith(f"Exit code: {exit_code}\n")
    assert stream.finalize_blocks()[-1]["result"]["data"].startswith(f"Exit code: {exit_code}\n")
    assert stream.error_text == ""


def test_stream_drop_is_recognised():
    from importlib import import_module
    here = Path(__file__).resolve().parent
    name = "_ducky_codex_drop"
    package = types.ModuleType(name)
    package.__path__ = [str(here)]
    sys.modules[name] = package
    try:
        adapter = import_module(name + ".codex_adapter")
        assert adapter.is_stream_drop("stream disconnected before completion: websocket closed by server before response.completed")
        assert adapter.is_stream_drop("", "Reconnecting... 2/2", "")
        assert not adapter.is_stream_drop("Codex turn failed: invalid model", "", None)
        assert not adapter.is_stream_drop()
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop(name + ".codex_adapter", None)


def test_dropped_stream_is_retried_once_on_the_same_thread(launch_env):
    adapter, updater, kwargs, calls, events = launch_env

    def run(**call):
        calls.append(call)
        call["on_line"](json.dumps({"type": "thread.started", "thread_id": "thread-one"}))
        if len(calls) == 1:
            call["on_line"](json.dumps({"type": "error", "message": (
                "stream disconnected before completion: websocket closed by server before response.completed")}))
            return types.SimpleNamespace(cancelled=False, timed_out=False, returncode=1,
                                         stderr_tail="ERROR: Reconnecting... 2/2", raw_tail="")
        call["on_line"](json.dumps({"type": "item.completed", "item": {
            "id": "reply", "type": "agent_message", "text": "Done after retry",
        }}))
        return types.SimpleNamespace(cancelled=False, timed_out=False, returncode=0,
                                     stderr_tail="", raw_tail="")

    adapter.run_streaming_process.side_effect = run
    result = adapter.CodexAdapter().launch(**kwargs, mode="agent")
    assert adapter.run_streaming_process.call_count == 2
    assert result.ok, result.error
    assert "Done after retry" in (result.reply_text or "")
    assert updater.update_cli.call_count == 0
    assert any("retrying once" in str(e.get("text", "")) for e in events if isinstance(e, dict))
    second = " ".join(str(a) for a in calls[1]["argv"])
    assert "resume" in second and "thread-one" in second


def test_a_real_error_is_not_retried(launch_env):
    adapter, updater, kwargs, calls, events = launch_env

    def run(**call):
        calls.append(call)
        call["on_line"](json.dumps({"type": "thread.started", "thread_id": "thread-one"}))
        call["on_line"](json.dumps({"type": "error", "message": "invalid model: nope"}))
        return types.SimpleNamespace(cancelled=False, timed_out=False, returncode=1,
                                     stderr_tail="", raw_tail="")

    adapter.run_streaming_process.side_effect = run
    result = adapter.CodexAdapter().launch(**kwargs, mode="agent")
    assert adapter.run_streaming_process.call_count == 1
    assert not result.ok


def _fake_accounts(monkeypatch, homes):
    """Core's accounts: the running login, then the next in ``homes`` after a report_limit."""
    fake = types.SimpleNamespace(running=0, reports=[])
    fake.current_account = lambda _agent: f"acct{fake.running}"
    fake.report_limit = lambda _agent, account, reset_at: fake.reports.append((account, reset_at))

    def next_env(_agent):
        fake.running = min(fake.running + 1, len(homes) - 1)
        return {"CODEX_HOME": homes[fake.running]} if homes[fake.running] else {}

    fake.coding_agent_account_env = next_env
    import backend.agent

    monkeypatch.setattr(backend.agent, "accounts", fake, raising=False)
    monkeypatch.setitem(sys.modules, "backend.agent.accounts", fake)
    return fake


def _limit_run(calls, *, work_first=False):
    def run(**call):
        calls.append(call)
        call["on_line"](json.dumps({"type": "thread.started", "thread_id": "thread-one"}))
        if len(calls) == 1:
            if work_first:
                call["on_line"](json.dumps({"type": "item.completed", "item": {
                    "id": "half", "type": "agent_message", "text": "Edited two files",
                }}))
            call["on_line"](json.dumps({"type": "error", "message": (
                "You've hit your usage limit. Upgrade to Pro or try again in 2 hours 5 minutes.")}))
            return types.SimpleNamespace(cancelled=False, timed_out=False, returncode=1,
                                         stderr_tail="", raw_tail="")
        call["on_line"](json.dumps({"type": "item.completed", "item": {
            "id": "reply", "type": "agent_message", "text": "Done on Work",
        }}))
        return types.SimpleNamespace(cancelled=False, timed_out=False, returncode=0,
                                     stderr_tail="", raw_tail="")

    return run


def test_a_usage_limit_replays_the_turn_once_on_the_next_account(launch_env, monkeypatch, tmp_path):
    adapter, updater, kwargs, calls, events = launch_env
    personal, work = str(tmp_path / "personal"), str(tmp_path / "work")
    fake = _fake_accounts(monkeypatch, [personal, work])
    adapter.run_streaming_process.side_effect = _limit_run(calls)
    result = adapter.CodexAdapter().launch(**{**kwargs, "env": {"CODEX_HOME": personal}}, mode="agent")
    assert result.ok, result.error
    assert "Done on Work" in (result.reply_text or "")
    assert [c["env_extra"]["CODEX_HOME"] for c in calls] == [personal, work]
    assert fake.reports[0][0] == "acct0" and fake.reports[0][1] > 0
    assert any("usage limit" in str(e.get("text", "")) for e in events if isinstance(e, dict))


def test_a_usage_limit_after_work_is_not_replayed(launch_env, monkeypatch, tmp_path):
    adapter, updater, kwargs, calls, events = launch_env
    fake = _fake_accounts(monkeypatch, ["", str(tmp_path / "work")])
    adapter.run_streaming_process.side_effect = _limit_run(calls, work_first=True)
    result = adapter.CodexAdapter().launch(**kwargs, mode="agent")
    assert adapter.run_streaming_process.call_count == 1
    assert not result.ok and not fake.reports


def test_a_usage_limit_with_no_other_account_is_not_replayed(launch_env, monkeypatch):
    adapter, updater, kwargs, calls, events = launch_env
    _fake_accounts(monkeypatch, [""])  # only the default login
    adapter.run_streaming_process.side_effect = _limit_run(calls)
    result = adapter.CodexAdapter().launch(**kwargs, mode="agent")
    assert adapter.run_streaming_process.call_count == 1
    assert not result.ok


def test_limit_reset_reads_codex_wording(launch_env):
    import time as _time

    adapter = launch_env[0]
    got = adapter.limit_reset_at("try again in 2 hours 5 minutes.")
    assert abs(got - (_time.time() + 7500)) < 5
    assert adapter.limit_reset_at("try again later") is None


def _silent_run(calls, reply=""):
    """A Codex process that prints its thread id and then nothing until it is stopped."""

    def run(**call):
        calls.append(call)
        call["on_line"](json.dumps({"type": "thread.started", "thread_id": "thread-one"}))
        if reply and len(calls) > 1:
            call["on_line"](json.dumps({"type": "item.completed", "item": {
                "id": "reply", "type": "agent_message", "text": reply,
            }}))
            return types.SimpleNamespace(cancelled=False, timed_out=False, returncode=0,
                                         stderr_tail="", raw_tail="")
        assert call["cancel"].wait(5.0), "the run was never stopped"
        return types.SimpleNamespace(cancelled=True, timed_out=False, returncode=-1,
                                     stderr_tail="", raw_tail="")

    return run


def test_a_silent_codex_run_is_restarted_once_on_the_same_thread(launch_env, monkeypatch):
    adapter, updater, kwargs, calls, events = launch_env
    monkeypatch.setattr(adapter, "CODEX_SILENCE_LIMIT_S", 0.3)
    adapter.run_streaming_process.side_effect = _silent_run(calls, reply="Done after restart")
    result = adapter.CodexAdapter().launch(**kwargs, mode="agent")
    assert adapter.run_streaming_process.call_count == 2
    assert result.ok, result.error
    assert "Done after restart" in (result.reply_text or "")
    assert updater.update_cli.call_count == 0
    assert any("printed nothing" in str(e.get("text", "")) for e in events if isinstance(e, dict))
    second = " ".join(str(a) for a in calls[1]["argv"])
    assert "resume" in second and "thread-one" in second


def test_a_run_silent_twice_ends_with_a_clear_error(launch_env, monkeypatch):
    adapter, _, kwargs, calls, _ = launch_env
    monkeypatch.setattr(adapter, "CODEX_SILENCE_LIMIT_S", 0.3)
    adapter.run_streaming_process.side_effect = _silent_run(calls)
    result = adapter.CodexAdapter().launch(**kwargs, mode="agent")
    assert adapter.run_streaming_process.call_count == 2
    assert not result.ok and result.status == "error"
    assert "twice" in result.error and "same thread" in result.error
    assert result.upstream_session_id == "thread-one"


def test_the_persons_stop_is_forwarded_and_never_retried(launch_env):
    adapter, _, kwargs, calls, _ = launch_env
    stop = threading.Event()

    def run(**call):
        calls.append(call)
        call["on_line"](json.dumps({"type": "thread.started", "thread_id": "thread-one"}))
        stop.set()
        assert call["cancel"].wait(5.0), "the person's stop never reached the process"
        return types.SimpleNamespace(cancelled=True, timed_out=False, returncode=-1,
                                     stderr_tail="", raw_tail="")

    adapter.run_streaming_process.side_effect = run
    result = adapter.CodexAdapter().launch(**kwargs, mode="agent", cancel=stop)
    assert adapter.run_streaming_process.call_count == 1
    assert not result.ok and result.status == "cancelled"


# Oct 15 2026: Codex stopped offering gpt-6.1-sol and gpt-6-astra to this ChatGPT account.
# Exactly what `codex exec --json -m gpt-6.1-sol` printed then (exit code 1):
_SOL_REFUSED = json.dumps({"type": "error", "status": 400, "error": {
    "type": "invalid_request_error",
    "message": "The 'gpt-6.1-sol' model is not supported when using Codex with a ChatGPT account.",
}})
_SOL_REFUSED_LINES = [
    json.dumps({"type": "thread.started", "thread_id": "thread-one"}),
    json.dumps({"type": "item.completed", "item": {"id": "item_0", "type": "error", "message": (
        "Model metadata for `gpt-6.1-sol` not found. Defaulting to fallback metadata; "
        "this can degrade performance and cause issues.")}}),
    json.dumps({"type": "turn.started"}),
    json.dumps({"type": "error", "message": _SOL_REFUSED}),
    json.dumps({"type": "turn.failed", "error": {"message": _SOL_REFUSED}}),
]
_OFFERED = ["gpt-6-luna", "gpt-5.6-terra", "gpt-5.6-luna"]
_OFFERED_TEXT = "Codex offers gpt-6-luna, gpt-5.6-terra, gpt-5.6-luna. Pick one of those for this chat."


def _write_models_cache(home: Path, slugs: list[str]) -> None:
    models = [{"slug": slug, "display_name": slug.upper(), "visibility": "list", "priority": i}
              for i, slug in enumerate(slugs)]
    models.append({"slug": "codex-auto-review", "visibility": "hide", "priority": 43})
    (home / "models_cache.json").write_text(json.dumps({"models": models}), encoding="utf-8")


def test_a_model_codex_refuses_is_a_clear_error_never_the_raw_json(launch_env, tmp_path, monkeypatch):
    adapter, updater, kwargs, calls, events = launch_env
    picker = Mock()
    monkeypatch.setattr(adapter, "_refresh_model_picker", picker)
    _write_models_cache(tmp_path, ["gpt-6.1-sol", *_OFFERED])  # the list before the run

    def run(**call):
        calls.append(call)
        _write_models_cache(tmp_path, _OFFERED)  # Codex refetched it at start-up
        for line in _SOL_REFUSED_LINES:
            call["on_line"](line)
        return types.SimpleNamespace(cancelled=False, timed_out=False, returncode=1,
                                     stderr_tail="Reading additional input from stdin...\n",
                                     raw_tail="\n".join(_SOL_REFUSED_LINES))

    adapter.run_streaming_process.side_effect = run
    result = adapter.CodexAdapter().launch(**{**kwargs, "model": "gpt-6.1-sol"}, mode="agent")
    assert adapter.run_streaming_process.call_count == 1  # not a dropped stream: no retry
    assert updater.update_cli.call_count == 0
    assert not result.ok and result.status == "error"
    assert result.error == "This model isn't available for your ChatGPT account: gpt-6.1-sol. " + _OFFERED_TEXT
    # Ducky turns a failed turn's output tail into the reply when there is none.
    assert result.output_tail == "" and result.reply_text == ""
    assert result.upstream_session_id == "thread-one"
    picker.assert_called_once_with()
    shown = [str(e.get("text", "")) for e in events] + [result.error]
    assert not any('"type"' in text or "{" in text for text in shown)


@pytest.mark.parametrize("raw,shown", [
    (_SOL_REFUSED, "This model isn't available for your ChatGPT account: gpt-6.1-sol. Pick another model for this chat."),
    ('unexpected status 400 Bad Request: {"detail":"The \'gpt-6-astra\' model is not supported when using Codex with a ChatGPT account."}',
     "This model isn't available for your ChatGPT account: gpt-6-astra. Pick another model for this chat."),
    (json.dumps({"error": {"message": json.dumps({"error": {"message": "Rate limit reached"}})}}), "Rate limit reached"),
    ('unexpected status 401 Unauthorized: {"error":{"message":"Your session has expired. Please log in again."}}',
     "Your session has expired. Please log in again."),
    ("stream disconnected before completion: websocket closed by server before response.completed",
     "stream disconnected before completion: websocket closed by server before response.completed"),
    ('{"items": [1, 2]}', '{"items": [1, 2]}'),
    ("{not json", "{not json"),
    ("", ""),
])
def test_codex_error_text_is_the_message_inside_the_error_body(launch_env, raw, shown):
    adapter = launch_env[0]  # no models_cache.json in this CODEX_HOME
    assert adapter.codex_error_text(raw) == shown


def test_a_turn_failed_with_a_plain_string_error_still_fails_the_turn(launch_env):
    adapter = launch_env[0]
    stream = adapter._CodexStream("chat", "run", lambda event: None)
    stream.on_line(json.dumps({"type": "turn.failed", "error": "Quota exceeded"}))
    assert stream.error_text == "Quota exceeded"
    stream.on_line(json.dumps({"type": "turn.failed"}))
    assert stream.error_text == "Codex turn failed"


def test_codex_json_is_never_the_error_of_a_turn_without_one(launch_env):
    adapter, _, kwargs, calls, _ = launch_env
    lines = [json.dumps({"type": "thread.started", "thread_id": "thread-one"}), json.dumps({"type": "turn.started"})]

    def run(**call):
        calls.append(call)
        for line in lines:
            call["on_line"](line)
        return types.SimpleNamespace(cancelled=False, timed_out=False, returncode=3,
                                     stderr_tail="", raw_tail="\n".join(lines))

    adapter.run_streaming_process.side_effect = run
    result = adapter.CodexAdapter().launch(**kwargs, mode="agent")
    assert not result.ok and result.status == "error"
    assert result.error == "Codex stopped without a reply (exit code 3)."
    assert result.output_tail == ""


@pytest.mark.parametrize("session", ["", "thread-one"])
def test_a_model_codex_no_longer_lists_is_flagged_before_the_turn(launch_env, tmp_path, monkeypatch, session):
    adapter, _, kwargs, calls, events = launch_env
    _write_models_cache(tmp_path, _OFFERED)
    fresh = Mock(return_value=list(_OFFERED))
    picker = Mock()
    monkeypatch.setattr(adapter, "_fresh_offered_models", fresh)
    monkeypatch.setattr(adapter, "_refresh_model_picker", picker)
    result = adapter.CodexAdapter().launch(**{**kwargs, "model": "gpt-6-astra"}, mode="agent", session_id=session)
    assert not result.ok and result.status == "error"
    assert result.error == "This model isn't available for your ChatGPT account: gpt-6-astra. " + _OFFERED_TEXT
    assert result.upstream_session_id == session  # the thread carries on with another model
    assert result.requested_mode == "agent" and result.effective_mode == ""
    assert not calls and not events  # no Codex run, no "Starting Codex…"
    fresh.assert_called_once_with("codex.exe")
    picker.assert_called_once_with()


@pytest.mark.parametrize("model,cache,fresh_list,checks_cli", [
    ("gpt-6-luna", _OFFERED, None, False),  # listed: no CLI call before the turn
    ("auto", _OFFERED, None, False),  # Codex picks
    ("gpt-6.2-nova", None, None, False),  # no cache yet: nothing to check against
    ("gpt-6.2-nova", _OFFERED, [*_OFFERED, "gpt-6.2-nova"], True),  # stale cache: the fresh list has it
    ("gpt-6.2-nova", _OFFERED, [], True),  # the CLI could not say: Codex decides
])
def test_a_model_the_account_may_have_is_never_blocked(launch_env, tmp_path, monkeypatch, model, cache, fresh_list, checks_cli):
    adapter, _, kwargs, calls, _ = launch_env
    if cache is not None:
        _write_models_cache(tmp_path, cache)
    fresh = Mock(return_value=fresh_list)
    monkeypatch.setattr(adapter, "_fresh_offered_models", fresh)
    monkeypatch.setattr(adapter, "_refresh_model_picker", Mock(side_effect=AssertionError("picker refresh")))
    result = adapter.CodexAdapter().launch(**{**kwargs, "model": model}, mode="agent")
    assert result.ok, result.error
    assert len(calls) == 1
    assert fresh.called is checks_cli


@pytest.mark.parametrize("outcome,offered", [
    ("listed", ["gpt-6-luna", "gpt-5.6-terra"]),
    ("failed", []),
    ("garbage", []),
    ("timeout", []),
    ("missing", []),
])
def test_the_fresh_model_list_comes_from_codex_debug_models(launch_env, monkeypatch, outcome, offered):
    import subprocess

    adapter, updater, *_ = launch_env
    catalog = {"models": [
        {"slug": "gpt-6-luna", "visibility": "list", "priority": 4},
        {"slug": "gpt-reserve", "visibility": "hide", "priority": 4},
        {"slug": "gpt-5.6-terra", "visibility": "list", "priority": 8},
    ]}

    def run_cli(argv, *, timeout_s):
        assert argv == ["codex.exe", "debug", "models"] and timeout_s <= 30
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, timeout_s)
        if outcome == "missing":
            raise FileNotFoundError(argv[0])
        stdout = "not json" if outcome == "garbage" else json.dumps(catalog)
        return types.SimpleNamespace(returncode=1 if outcome == "failed" else 0, stdout=stdout, stderr="")

    monkeypatch.setattr(updater, "_run", run_cli)
    assert adapter._fresh_offered_models("codex.exe") == offered


def test_codex_starts_in_its_own_process_group(launch_env):
    module, _, kwargs, calls, _ = launch_env
    assert module.CodexAdapter().launch(**kwargs).ok
    assert calls and all(call["new_process_group"] is True for call in calls)


def test_stop_sends_ctrl_break_once_to_the_chats_codex(launch_env, monkeypatch):
    import signal

    from backend.agent.coding_agents import proc_exec

    module = launch_env[0]
    proc = Mock()
    asked: list[str] = []
    monkeypatch.setattr(proc_exec, "get_process", lambda cid: asked.append(cid) or proc, raising=False)
    assert module.CodexAdapter().request_stop("conversation") is True
    expected = signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT
    proc.send_signal.assert_called_once_with(expected)
    assert asked == ["conversation"]


def test_stop_without_a_process_or_an_older_core_asks_core_to_kill(launch_env, monkeypatch):
    from backend.agent.coding_agents import proc_exec

    module = launch_env[0]
    monkeypatch.setattr(proc_exec, "get_process", lambda cid: None, raising=False)
    assert module.CodexAdapter().request_stop("conversation") is False
    gone = Mock(**{"send_signal.side_effect": OSError("exited")})
    monkeypatch.setattr(proc_exec, "get_process", lambda cid: gone)
    assert module.CodexAdapter().request_stop("conversation") is False
    monkeypatch.delattr(proc_exec, "get_process")
    assert module.CodexAdapter().request_stop("conversation") is False
    monkeypatch.setattr(module, "_PROCESS_GROUP", {})
    monkeypatch.setattr(proc_exec, "get_process", lambda cid: Mock(), raising=False)
    assert module.CodexAdapter().request_stop("conversation") is False
