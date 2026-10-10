"""Public launch contract: native sandbox alone cannot authorize Ducky MCP.

The shared fixture runs the actual adapter/finalizer with a fake process. No
vendor process, updater, or external endpoint is used by these tests.
"""
import threading
from unittest.mock import Mock

import pytest

from test_codex_modes import launch_env, registered_profile, assert_mode_argv


def forbid_setup(module, updater, monkeypatch):
    guards = []
    for name in ("normalize_codex_model", "write_codex_uefn_profile",
                 "coding_agent_cfg", "codex_extra_dirs", "heal_codex_approval_policy"):
        guard = Mock(side_effect=AssertionError("PRIVATE_SETUP_TRIPWIRE"))
        monkeypatch.setattr(module, name, guard)
        guards.append(guard)
    for name in ("NamedTemporaryFile", "TemporaryDirectory", "mkdtemp", "mkstemp"):
        guard = Mock(side_effect=AssertionError("PRIVATE_TEMP_TRIPWIRE"))
        monkeypatch.setattr(module.tempfile, name, guard)
        guards.append(guard)
    guards.extend((updater.resolve_bin, updater.update_cli, module.run_streaming_process))
    return guards


def assert_refused(result, requested, session, events):
    assert not result.ok and result.status == "error"
    assert result.upstream_session_id == session
    assert getattr(result, "requested_mode", requested) == requested
    assert getattr(result, "effective_mode", "") == ""
    assert result.error and not result.reply_text
    assert "PRIVATE" not in repr(result.to_dict()) + repr(events)
    assert not events  # Refusal occurs before any launch/progress/success stream.


@pytest.mark.parametrize("mode", ["ask", "plan", " Ask ", " PLAN "])
@pytest.mark.parametrize("session", ["", "original-thread"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_restricted_refusal_precedes_all_setup(launch_env, monkeypatch, mode, session, cancelled):
    module, updater, kw, calls, events = launch_env
    guards = forbid_setup(module, updater, monkeypatch)
    cancel = threading.Event()
    if cancelled:
        cancel.set()
    kw.update(model="PRIVATE_MODEL", mcp_config_path="PRIVATE_CONFIG",
              env={"PRIVATE_TOKEN": "PRIVATE_SECRET"})
    result = module.CodexAdapter().launch(**kw, mode=mode, session_id=session, cancel=cancel)
    assert_refused(result, mode.strip().lower(), session, events)
    for guard in guards:
        guard.assert_not_called()
    assert not calls


@pytest.mark.parametrize("profile", ["current", "legacy", "missing_helper", "old_capabilities", "old_result"])
@pytest.mark.parametrize("session", ["", "original-thread"])
def test_registered_capabilities_are_agent_only(launch_env, monkeypatch, profile, session):
    with registered_profile(launch_env, monkeypatch, profile) as (module, agent, kw, calls, events, base):
        assert isinstance(agent.capabilities, base.CodingAgentCapabilities)
        assert getattr(agent.capabilities, "supported_modes", ("agent",)) == ("agent",)
        guards = forbid_setup(module, launch_env[1], monkeypatch)
        for mode in ("ask", "plan"):
            result = agent.launch(**kw, mode=mode, session_id=session)
            assert type(result) is base.CodingAgentLaunchResult
            assert_refused(result, mode, session, events)
            if profile in ("legacy", "old_result"):
                assert "effective_mode" not in result.to_dict()
        for guard in guards:
            guard.assert_not_called()
        assert not calls


@pytest.mark.parametrize("profile", ["current", "legacy", "missing_helper", "old_capabilities", "old_result"])
@pytest.mark.parametrize("session", ["", "thread-one"])
def test_agent_remains_usable_on_registered_hosts(launch_env, monkeypatch, profile, session):
    with registered_profile(launch_env, monkeypatch, profile) as (module, agent, kw, calls, events, base):
        result = agent.launch(**kw, session_id=session)
        assert type(result) is base.CodingAgentLaunchResult
        assert result.ok and result.status == "done" and result.reply_text == "Done"
        assert result.upstream_session_id == "thread-one"
        assert len(calls) == 1
        assert_mode_argv(calls[0]["argv"], "agent", session)
        if profile not in ("legacy", "old_result"):
            assert result.requested_mode == result.effective_mode == "agent"
        else:
            assert "effective_mode" not in result.to_dict()
        launch_env[1].update_cli.assert_not_called()


@pytest.mark.parametrize("mode", [None, "", "agent", " Agent "])
def test_agent_normalization_is_preserved(launch_env, mode):
    module, updater, kw, calls, events = launch_env
    result = module.CodexAdapter().launch(**kw, mode=mode)
    assert result.ok and result.requested_mode == result.effective_mode == "agent"
    assert len(calls) == 1
    updater.update_cli.assert_not_called()


@pytest.mark.parametrize("session", ["", "thread-one"])
def test_mode_switches_never_reuse_agent_authority(launch_env, monkeypatch, session):
    module, updater, kw, calls, events = launch_env
    agent = module.CodexAdapter()
    for mode in ("agent", "ask", "plan", "agent", "plan", "ask"):
        events.clear()
        count = len(calls)
        if mode == "agent":
            result = agent.launch(**kw, mode=mode, session_id=session)
            assert result.ok and result.effective_mode == "agent"
            assert len(calls) == count + 1
            session = result.upstream_session_id
        else:
            with monkeypatch.context() as patch:
                writer = Mock(side_effect=AssertionError("PRIVATE_SETUP_TRIPWIRE"))
                patch.setattr(module, "write_codex_uefn_profile", writer)
                result = agent.launch(**kw, mode=mode, session_id=session)
                assert_refused(result, mode, session, events)
                writer.assert_not_called()
            assert len(calls) == count
    updater.update_cli.assert_not_called()


@pytest.mark.parametrize("cancelled", [False, True])
def test_invalid_mode_refuses_without_setup(launch_env, monkeypatch, cancelled):
    module, updater, kw, calls, events = launch_env
    guards = forbid_setup(module, updater, monkeypatch)
    cancel = threading.Event()
    if cancelled:
        cancel.set()
    result = module.CodexAdapter().launch(**kw, mode="invalid", session_id="original", cancel=cancel)
    # Existing invalid-mode normalization contract; no new raw-value API.
    assert_refused(result, "invalid", "original", events)
    for guard in guards:
        guard.assert_not_called()


def test_agent_setup_failure_stays_sanitized(launch_env, monkeypatch):
    module, updater, kw, calls, events = launch_env
    monkeypatch.setattr(module, "write_codex_uefn_profile",
                        Mock(side_effect=OSError("PRIVATE_CONFIG_SECRET")))
    result = module.CodexAdapter().launch(**kw, mode="agent", session_id="original")
    assert_refused(result, "agent", "original", events)
    assert result.error.startswith("Ducky tools unavailable:")
    assert not calls
    updater.resolve_bin.assert_not_called()
    updater.update_cli.assert_not_called()
