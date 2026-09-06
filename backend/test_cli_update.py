from __future__ import annotations

from pathlib import Path

from cli_update import (
    is_cli_broken_binary_error,
    is_cli_missing_error,
    is_cli_too_old_error,
    is_node_codex_wrapper,
    native_codex_candidates,
    parse_version_tuple,
    plugin_package_version,
    resolve_bin,
    should_heal_launch,
)


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
