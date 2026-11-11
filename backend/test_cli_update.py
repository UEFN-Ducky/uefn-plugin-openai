from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

import cli_update
from cli_update import (
    hidden_run_kwargs,
    install_skip_reason,
    npm_argv,
    is_cli_broken_binary_error,
    is_cli_missing_error,
    is_cli_too_old_error,
    is_node_codex_wrapper,
    native_codex_candidates,
    parse_version_tuple,
    plugin_package_version,
    resolve_bin,
    sandbox_skip_reason,
    schedule_cli_update_on_plugin_load,
    should_heal_launch,
    update_cli,
)


def test_installer_finds_powershell_without_path(tmp_path, monkeypatch):
    """'[WinError 2] The system cannot find the file specified' when PATH lacks PowerShell."""
    exe = tmp_path / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    assert cli_update.powershell_exe() == str(exe)
    ran = []
    monkeypatch.setattr(cli_update, "_run", lambda argv, **kw: ran.append(argv) or types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(cli_update.os, "name", "nt")
    assert cli_update._ps_install() == {"ok": True}
    assert ran[0][0] == str(exe)


def test_parse_and_heal():
    assert parse_version_tuple("codex-cli 0.42.0") == (0, 42, 0)
    assert is_cli_too_old_error(
        "does not support this model; version 0.50.0 or newer is required. Run update"
    )
    assert is_cli_missing_error("'codex' is not recognized as an internal or external command")
    assert is_cli_broken_binary_error("Error: spawn EFAULT")
    assert is_cli_broken_binary_error("The system cannot execute the specified program.")
    assert should_heal_launch("is not recognized")
    assert should_heal_launch("node:internal/child_process spawn EFAULT")
    assert not should_heal_launch("rate limit exceeded")


def test_skips_npm_shim_for_standalone(tmp_path, monkeypatch):
    exe = (
        tmp_path
        / ".codex"
        / "packages"
        / "standalone"
        / "releases"
        / "0.153.4-x86_64-pc-windows-msvc"
        / "bin"
        / "codex.exe"
    )
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    monkeypatch.setattr("cli_update._codex_home", lambda: tmp_path / ".codex")
    monkeypatch.setattr(
        "cli_update._path_codex",
        lambda override="": r"C:\Users\x\AppData\Roaming\npm\codex.cmd",
    )
    assert is_node_codex_wrapper(r"C:\Users\x\AppData\Roaming\npm\codex.cmd")
    assert is_node_codex_wrapper(r"C:\Users\x\AppData\Roaming\npm\codex")
    assert is_node_codex_wrapper(r"C:\Users\x\AppData\Roaming\npm\codex.exe")
    assert not is_node_codex_wrapper(str(exe))
    assert native_codex_candidates()[0] == str(exe.resolve())
    assert resolve_bin() == str(exe.resolve())


def test_npm_argv_skips_cmd(tmp_path, monkeypatch):
    node = tmp_path / "node.exe"
    node.write_bytes(b"")
    cli = tmp_path / "node_modules" / "npm" / "bin" / "npm-cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("", encoding="utf-8")
    npm = tmp_path / "npm.cmd"
    npm.write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setattr("cli_update.shutil.which", lambda _name: None)
    argv = npm_argv(str(npm), ["install", "-g", "@openai/codex"])
    assert argv[:2] == [str(node), str(cli)]
    if os.name == "nt":
        assert hidden_run_kwargs().get("startupinfo") is not None


def _core_reason(monkeypatch, reason):
    core = types.ModuleType("frontend.ide_apply")
    core.auto_apply_skip_reason = lambda: reason
    monkeypatch.setitem(sys.modules, "frontend.ide_apply", core)


def test_sandbox_reason_comes_from_core(monkeypatch):
    monkeypatch.setenv("UEFN_DUCKY_PANEL_PORT", "4300")
    _core_reason(monkeypatch, "LOCALAPPDATA is not this Windows user's own AppData\\Local")
    assert sandbox_skip_reason() == "LOCALAPPDATA is not this Windows user's own AppData\\Local"
    _core_reason(monkeypatch, "")
    assert sandbox_skip_reason() == ""
    monkeypatch.setitem(sys.modules, "frontend.ide_apply", None)  # older core: no check
    assert sandbox_skip_reason() == "panel port 4300 is not the default 4200"
    monkeypatch.setenv("UEFN_DUCKY_PANEL_PORT", "4200")
    assert sandbox_skip_reason() == ""
    monkeypatch.delenv("UEFN_DUCKY_PANEL_PORT")
    assert sandbox_skip_reason() == ""


@pytest.mark.parametrize("instance", ["skip_env", "core_port", "older_core_port"])
def test_second_instance_update_never_runs_installer(monkeypatch, instance):
    install = Mock(side_effect=AssertionError("Codex installer ran"))
    run = Mock(side_effect=AssertionError("codex update ran"))
    monkeypatch.setattr(cli_update, "_install_cli", install)
    monkeypatch.setattr(cli_update, "_run", run)
    monkeypatch.setattr(cli_update, "resolve_bin", Mock(return_value=""))
    monkeypatch.setattr(cli_update, "write_stamp", Mock(side_effect=AssertionError("stamp written")))
    monkeypatch.setattr(cli_update, "_last", {})
    monkeypatch.setenv("UEFN_DUCKY_PANEL_PORT", "4300")
    if instance == "skip_env":
        monkeypatch.setenv("DUCKY_SKIP_CLI_UPDATE", "1")
        _core_reason(monkeypatch, "")
        reason = "DUCKY_SKIP_CLI_UPDATE is set"
    else:
        monkeypatch.delenv("DUCKY_SKIP_CLI_UPDATE", raising=False)
        reason = "panel port 4300 is not the default 4200"
        if instance == "core_port":
            _core_reason(monkeypatch, reason)
        else:
            monkeypatch.setitem(sys.modules, "frontend.ide_apply", None)
    assert install_skip_reason() == reason
    for override in ("", "codex.exe"):
        result = update_cli(override)
        assert result["ok"] is False and reason in result["error"]
    install.assert_not_called()
    run.assert_not_called()

    monkeypatch.delenv("DUCKY_SKIP_CLI_UPDATE", raising=False)
    _core_reason(monkeypatch, "")
    install.side_effect = None
    install.return_value = {"ok": False, "error": "offline"}
    assert update_cli("") == {"ok": False, "error": "offline"}  # the person's own Ducky still installs
    install.assert_called_once()


def test_sandboxed_plugin_load_never_installs(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("DUCKY_SKIP_CLI_UPDATE", raising=False)
    needs = Mock(return_value=True)
    started = []
    monkeypatch.setattr(cli_update, "needs_plugin_load_update", needs)
    monkeypatch.setattr(cli_update, "update_cli", Mock(side_effect=AssertionError("installer ran")))
    monkeypatch.setattr(cli_update, "threading", types.SimpleNamespace(
        Thread=lambda **kw: types.SimpleNamespace(start=lambda: started.append(kw["name"])),
    ))
    _core_reason(monkeypatch, "LOCALAPPDATA is not the Windows user's own folder")
    schedule_cli_update_on_plugin_load()
    assert started == [] and needs.call_count == 0

    _core_reason(monkeypatch, "")
    schedule_cli_update_on_plugin_load()
    assert started == ["codex-cli-update"]

    monkeypatch.setenv("DUCKY_SKIP_CLI_UPDATE", "1")
    schedule_cli_update_on_plugin_load()
    assert started == ["codex-cli-update"]


def test_plugin_wires_auto_update():
    init = Path(__file__).with_name("__init__.py").read_text(encoding="utf-8")
    adapter = Path(__file__).with_name("codex_adapter.py").read_text(encoding="utf-8")
    assert "schedule_cli_update_on_plugin_load" in init
    assert "heal_codex_approval_policy" in init
    assert "update_cli" in adapter
    assert plugin_package_version()


if __name__ == "__main__":
    test_parse_and_heal()
    test_plugin_wires_auto_update()
    print("ok")
