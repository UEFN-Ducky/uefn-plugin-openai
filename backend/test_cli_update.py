from __future__ import annotations

from pathlib import Path

from cli_update import (
    is_cli_missing_error,
    is_cli_too_old_error,
    parse_version_tuple,
    plugin_package_version,
    should_heal_launch,
)


def test_parse_and_heal():
    assert parse_version_tuple("codex-cli 0.42.0") == (0, 42, 0)
    assert is_cli_too_old_error(
        "does not support this model; version 0.50.0 or newer is required. Run update"
    )
    assert is_cli_missing_error("'codex' is not recognized as an internal or external command")
    assert should_heal_launch("is not recognized")
    assert not should_heal_launch("rate limit exceeded")


def test_plugin_wires_auto_update():
    init = Path(__file__).with_name("__init__.py").read_text(encoding="utf-8")
    adapter = Path(__file__).with_name("codex_adapter.py").read_text(encoding="utf-8")
    assert "schedule_cli_update_on_plugin_load" in init
    assert "update_cli" in adapter
    assert plugin_package_version() == "1.0.20"


if __name__ == "__main__":
    test_parse_and_heal()
    test_plugin_wires_auto_update()
    print("ok")
