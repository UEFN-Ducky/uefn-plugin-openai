"""The person's own URL server named "uefn" never stops Ducky's Codex launches.

Codex merges config layers: Ducky's per-run ``mcp_servers.uefn={command=...}`` landed on top
of a ``[mcp_servers.uefn] url = ...`` in ~/.codex/config.toml and every chat failed with
"Error loading config.toml: url is not supported for stdio | in mcp_servers.uefn".
"""
from __future__ import annotations

import sys
import tomllib
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

from codex_adapter import FOREIGN_UEFN_NAME, _TOML_BEGIN, _TOML_END, move_foreign_uefn  # noqa: E402

THEIRS = 'model = "gpt-5"\n\n[mcp_servers.uefn]\nurl = "http://127.0.0.1:8001/mcp"\n\n[mcp_servers.uefn.http_headers]\nX = "y"\n'


def test_their_url_uefn_moves_aside_and_keeps_working(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(THEIRS, encoding="utf-8")
    assert move_foreign_uefn(codex_home=tmp_path) is True
    data = tomllib.loads(cfg.read_text(encoding="utf-8"))
    assert "uefn" not in data["mcp_servers"]
    moved = data["mcp_servers"][FOREIGN_UEFN_NAME]
    assert moved == {"url": "http://127.0.0.1:8001/mcp", "http_headers": {"X": "y"}}
    assert data["model"] == "gpt-5"
    assert move_foreign_uefn(codex_home=tmp_path) is False  # once only


def test_ducky_block_and_command_servers_are_left_alone(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    ducky = f'{_TOML_BEGIN}\n[mcp_servers.uefn]\ncommand = "C:/UEFN-Ducky-Bridge.exe"\n{_TOML_END}\n'
    cfg.write_text(ducky, encoding="utf-8")
    assert move_foreign_uefn(codex_home=tmp_path) is False
    cfg.write_text('[mcp_servers.uefn]\ncommand = "C:/UEFN-Ducky.exe"\n', encoding="utf-8")
    assert move_foreign_uefn(codex_home=tmp_path) is False
    assert move_foreign_uefn(codex_home=tmp_path / "missing") is False


def test_a_url_uefn_beside_duckys_block_moves_and_the_block_stays(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    ducky = f'{_TOML_BEGIN}\n# Added by Ducky Settings > Apply\nmodel_reasoning_effort = "low"\n{_TOML_END}\n'
    cfg.write_text(THEIRS + "\n" + ducky, encoding="utf-8")
    assert move_foreign_uefn(codex_home=tmp_path) is True
    text = cfg.read_text(encoding="utf-8")
    assert _TOML_BEGIN in text and "# Added by Ducky Settings > Apply" in text
    assert "[mcp_servers.uefn]" not in text and f"[mcp_servers.{FOREIGN_UEFN_NAME}]" in text
