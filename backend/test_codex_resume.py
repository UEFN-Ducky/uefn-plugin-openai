from __future__ import annotations

import json
import sys
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

from codex_adapter import (
    CodexAdapter,
    build_codex_argv,
    heal_codex_approval_policy,
    write_codex_uefn_profile,
)


def test_followup_resumes_thread():
    argv = build_codex_argv(
        binary="codex",
        prompt="follow up",
        model="gpt-5",
        extra_args="",
        session_id="th-abc",
    )
    assert argv[1:4] == ["exec", "resume", "th-abc"]
    assert argv[-1] == "follow up"


def test_first_turn_has_no_resume():
    argv = build_codex_argv(
        binary="codex",
        prompt="hello",
        model="gpt-5",
        extra_args="",
        session_id="",
    )
    assert "resume" not in argv


def test_adapter_opts_into_resume():
    assert CodexAdapter.capabilities.resume is True


def test_writes_codex_profile_from_ducky_mcp(tmp_path: Path):
    mcp = tmp_path / "mcp.json"
    mcp.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "uefn": {
                        "command": r"C:\Ducky\UEFN-Ducky.exe",
                        "args": ["bridge", "--port", "4200"],
                        "env": {"DUCKY_CONV_ID": "chat-1"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "config.toml").write_text('[mcp_servers.node_repl]\ncommand = "x"\n', encoding="utf-8")
    name = write_codex_uefn_profile(str(mcp), codex_home=home)
    assert name == "ducky-uefn"
    text = (home / "ducky-uefn.config.toml").read_text(encoding="utf-8")
    assert "mcp_servers.uefn" in text
    assert 'approval_policy = "on-failure"' in text
    assert 'approval_policy = "never"' not in text
    assert 'default_tools_approval_mode = "approve"' in text
    assert 'default_tools_approval_mode = "auto"' not in text
    assert r"C:\\Ducky\\UEFN-Ducky.exe" in text
    assert "DUCKY_CONV_ID" in text
    cfg = (home / "config.toml").read_text(encoding="utf-8")
    assert "node_repl" in cfg
    assert "BEGIN UEFN-DUCKY" in cfg
    assert "mcp_servers.uefn" in cfg
    write_codex_uefn_profile(str(mcp), codex_home=home)
    assert (home / "config.toml").read_text(encoding="utf-8").count("BEGIN UEFN-DUCKY") == 1
    first = build_codex_argv(
        binary="codex",
        prompt="hello",
        model="gpt-5",
        extra_args="",
        session_id="",
        extra_flags=["-c", 'approval_policy="on-failure"', "-c", 'mcp_servers.uefn.default_tools_approval_mode="approve"', "--approve-for-me", "-c", 'sandbox_workspace_write.writable_roots=["C:\\\\tmp"]'],
    )
    assert "-c" in first
    assert 'approval_policy="on-failure"' in first
    assert 'mcp_servers.uefn.default_tools_approval_mode="approve"' in first
    assert "--approve-for-me" in first
    resume = build_codex_argv(
        binary="codex",
        prompt="continue",
        model="gpt-5",
        extra_args="--full-auto --approve-for-me --add-dir C:\\tmp --oss",
        session_id="th-abc",
        extra_flags=["-p", name, "--add-dir", r"C:\tmp", "-c", 'approval_policy="on-failure"', "-c", 'mcp_servers.uefn.default_tools_approval_mode="approve"'],
    )
    assert "-p" not in resume
    assert "--add-dir" not in resume
    assert "--approve-for-me" not in resume
    assert "--oss" not in resume
    assert "-s" not in resume
    assert 'approval_policy="on-failure"' in resume
    assert 'mcp_servers.uefn.default_tools_approval_mode="approve"' in resume
    assert any(t.startswith("sandbox_mode=") or t == 'sandbox_mode="workspace-write"' for t in resume)
    glued = build_codex_argv(
        binary="codex",
        prompt="continue",
        model="gpt-5",
        extra_args="",
        session_id="th-abc",
        extra_flags=["-p=ducky-uefn", "--profile=ducky-uefn"],
    )
    assert "-p=ducky-uefn" not in glued
    assert "--profile=ducky-uefn" not in glued
    assert write_codex_uefn_profile("", codex_home=home) == ""


def test_heal_codex_approval_inserts_without_wiping_mcp(tmp_path: Path):
    home = tmp_path / "codex-home-insert"
    home.mkdir()
    (home / "config.toml").write_text(
        '# BEGIN UEFN-DUCKY\n[mcp_servers.uefn]\ncommand = "x"\n# END UEFN-DUCKY\n',
        encoding="utf-8",
    )
    assert heal_codex_approval_policy(codex_home=home) is True
    text = (home / "config.toml").read_text(encoding="utf-8")
    assert 'approval_policy = "on-failure"' in text
    assert "[mcp_servers.uefn]" in text
    assert 'command = "x"' in text
    assert 'default_tools_approval_mode = "approve"' in text


def test_heal_codex_approval_rewrites_never(tmp_path: Path):
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "config.toml").write_text(
        '# BEGIN UEFN-DUCKY\napproval_policy = "never"\n[mcp_servers.uefn]\n'
        'default_tools_approval_mode = "auto"\n# END UEFN-DUCKY\n',
        encoding="utf-8",
    )
    (home / "ducky-uefn.config.toml").write_text(
        'approval_policy = "never"\n',
        encoding="utf-8",
    )
    assert heal_codex_approval_policy(codex_home=home) is True
    healed = (home / "config.toml").read_text(encoding="utf-8")
    assert 'approval_policy = "on-failure"' in healed
    assert 'approval_policy = "never"' not in healed
    assert 'default_tools_approval_mode = "approve"' in healed
    assert 'default_tools_approval_mode = "auto"' not in healed
    assert 'approval_policy = "on-failure"' in (
        home / "ducky-uefn.config.toml"
    ).read_text(encoding="utf-8")


if __name__ == "__main__":
    import tempfile

    test_followup_resumes_thread()
    test_first_turn_has_no_resume()
    test_adapter_opts_into_resume()
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        write_home = root / "write"
        write_home.mkdir()
        test_writes_codex_profile_from_ducky_mcp(write_home)
        heal_home = root / "heal"
        heal_home.mkdir()
        test_heal_codex_approval_inserts_without_wiping_mcp(heal_home)
        never_home = root / "never"
        never_home.mkdir()
        test_heal_codex_approval_rewrites_never(never_home)
    print("ok")
