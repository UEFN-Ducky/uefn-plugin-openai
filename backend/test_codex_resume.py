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
    _codex_model_rows,
    _writable_roots_flag,
    build_codex_argv,
    codex_extra_dirs,
    ducky_launch_flags,
    heal_codex_approval_policy,
    inline_prompt_limit,
    normalize_codex_model,
    parse_codex_catalog,
    with_ducky_tools_hint,
    write_codex_uefn_profile,
)


def test_recent_project_is_a_writable_root_but_cwd_is_not(tmp_path, monkeypatch):
    active = tmp_path / "ExampleProject1"
    other = tmp_path / "Roguelike"
    active.mkdir()
    other.mkdir()
    import frontend.ui_web.recent_projects as recent

    monkeypatch.setattr(recent, "load_recent_projects", lambda: [str(active), str(other)])
    dirs = codex_extra_dirs(str(active))
    flag = " ".join(_writable_roots_flag(dirs))
    assert "Roguelike" in flag
    assert "ExampleProject1" not in flag


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
    assert "-s" not in argv
    assert "--sandbox" not in argv


def test_resume_compacts_before_the_thread_gets_too_big():
    argv = build_codex_argv(binary="codex", prompt="go", model="gpt-5", extra_args="", session_id="th-abc")
    assert "model_auto_compact_token_limit=150000" in argv
    assert argv[-1] == "go"
    own = build_codex_argv(
        binary="codex",
        prompt="go",
        model="gpt-5",
        extra_args="-c model_auto_compact_token_limit=90000",
        session_id="th-abc",
    )
    assert sum("model_auto_compact_token_limit" in a for a in own) == 1


def test_reasoning_effort_flag():
    argv = build_codex_argv(
        binary="codex",
        prompt="hello",
        model="gpt-5",
        extra_args="",
        session_id="",
        reasoning_effort="high",
    )
    assert "model_reasoning_effort=high" in argv
    off = build_codex_argv(
        binary="codex",
        prompt="hello",
        model="gpt-5",
        extra_args="",
        session_id="",
        reasoning_effort="off",
    )
    assert "model_reasoning_effort=high" not in off
    assert not any(str(a).startswith("model_reasoning_effort=") for a in off)


def test_first_turn_has_no_resume():
    argv = build_codex_argv(
        binary="codex",
        prompt="hello",
        model="gpt-5",
        extra_args="",
        session_id="",
    )
    assert "resume" not in argv
    assert "-s" not in argv
    assert "--sandbox" not in argv
    assert "--approve-for-me" not in argv
    assert "--dangerously-bypass-approvals-and-sandbox" in argv
    assert 'sandbox_mode="workspace-write"' in argv


def test_adapter_opts_into_resume():
    assert CodexAdapter.capabilities.resume is True


def test_parse_codex_catalog_lists_visible_only():
    rows = parse_codex_catalog(
        {
            "models": [
                {
                    "slug": "gpt-6-astra",
                    "display_name": "GPT-6-Astra",
                    "visibility": "list",
                    "priority": 1,
                    "input_modalities": ["text", "image"],
                    "supports_search_tool": True,
                    "context_window": 272000,
                },
                {
                    "slug": "gpt-5.6-terra",
                    "display_name": "GPT-5.6-Terra",
                    "visibility": "list",
                    "priority": 7,
                },
                {
                    "slug": "codex-auto-review",
                    "display_name": "Codex Auto Review",
                    "visibility": "hide",
                    "priority": 43,
                },
                {"slug": "gpt-reserve", "visibility": "hide"},
            ]
        }
    )
    ids = [r["id"] for r in rows]
    assert ids == ["gpt-6-astra", "gpt-5.6-terra"]
    assert rows[0]["name"] == "GPT-6-Astra"
    assert rows[0]["supports_vision"] is True
    assert rows[0]["supports_web_search"] is True
    assert rows[0]["context_limit"] == 272000
    astra_menu = rows[0].get("thinking_menu") or {}
    assert [lvl["id"] for lvl in astra_menu.get("levels") or []][:2] == ["off", "low"]
    terra_menu = rows[1].get("thinking_menu") or {}
    assert any(lvl.get("id") == "xhigh" for lvl in terra_menu.get("levels") or [])


def test_codex_models_reads_cli_cache(tmp_path: Path):
    cache = tmp_path / "models_cache.json"
    cache.write_text(
        json.dumps(
            {
                "models": [
                    {"slug": "gpt-6-astra", "display_name": "GPT-6-Astra", "visibility": "list", "priority": 1},
                    {"slug": "gpt-5.6-terra", "display_name": "GPT-5.6-Terra", "visibility": "list", "priority": 7},
                    {"slug": "codex-auto-review", "visibility": "hide"},
                ]
            }
        ),
        encoding="utf-8",
    )
    import codex_adapter as ca

    orig = ca._codex_catalog_path
    ca._codex_catalog_path = lambda: cache
    try:
        rows = _codex_model_rows()
        ids = [r["id"] for r in rows]
        assert ids[:3] == ["auto", "gpt-6-astra", "gpt-5.6-terra"]
        assert "codex-auto-review" not in ids
        assert all(r.get("supports_tools") for r in rows)
    finally:
        ca._codex_catalog_path = orig


def test_codex_models_without_cli_cache_uses_fallback(tmp_path: Path):
    import codex_adapter as ca

    orig_path = ca._codex_catalog_path
    orig_refresh = ca._refresh_cli_catalog_in_background
    ca._codex_catalog_path = lambda: tmp_path / "missing.json"
    ca._refresh_cli_catalog_in_background = lambda: None
    try:
        rows = _codex_model_rows()
        ids = {r["id"] for r in rows}
        assert "auto" in ids
        assert "gpt-6-astra" in ids
        assert "gpt-5.6-terra" in ids
        assert "gpt-5.1-codex" not in ids
    finally:
        ca._codex_catalog_path = orig_path
        ca._refresh_cli_catalog_in_background = orig_refresh


def test_auto_model_omits_dash_m():
    argv = build_codex_argv(
        binary="codex",
        prompt="hello",
        model="auto",
        extra_args="",
        session_id="",
    )
    assert "-m" not in argv
    assert "--sandbox" not in argv
    assert "-s" not in argv
    assert normalize_codex_model("default") == "auto"
    assert normalize_codex_model("") == "auto"
    assert normalize_codex_model("gpt-5") == "gpt-5"


def test_first_turn_uses_bypass_not_approve_for_me():
    argv = build_codex_argv(
        binary="codex",
        prompt="hello",
        model="auto",
        extra_args="--full-auto --approve-for-me",
        session_id="",
        extra_flags=["--dangerously-bypass-approvals-and-sandbox", "--approve-for-me"],
    )
    assert "--dangerously-bypass-approvals-and-sandbox" in argv
    assert argv.count("--dangerously-bypass-approvals-and-sandbox") == 1
    assert "--approve-for-me" not in argv
    assert "-s" not in argv
    assert "--sandbox" not in argv


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
    assert "tool_timeout_sec = 1000000000000.0" in text
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
        extra_flags=["-c", 'approval_policy="on-failure"', "-c", 'mcp_servers.uefn.default_tools_approval_mode="approve"', "--dangerously-bypass-approvals-and-sandbox", "--approve-for-me", "-c", 'sandbox_workspace_write.writable_roots=["C:\\\\tmp"]'],
    )
    assert "-c" in first
    assert 'approval_policy="on-failure"' in first
    assert 'mcp_servers.uefn.default_tools_approval_mode="approve"' in first
    assert "--dangerously-bypass-approvals-and-sandbox" in first
    assert "--approve-for-me" not in first
    assert "-s" not in first
    assert "--sandbox" not in first
    resume = build_codex_argv(
        binary="codex",
        prompt="continue",
        model="gpt-5",
        extra_args="--full-auto --approve-for-me --add-dir C:\\tmp --oss",
        session_id="th-abc",
        extra_flags=["-p", name, "--add-dir", r"C:\tmp", "-c", 'approval_policy="on-failure"', "-c", 'mcp_servers.uefn.default_tools_approval_mode="approve"', "--dangerously-bypass-approvals-and-sandbox"],
    )
    assert "-p" not in resume
    assert "--add-dir" not in resume
    assert "--approve-for-me" not in resume
    assert "--oss" not in resume
    assert "-s" not in resume
    assert 'approval_policy="on-failure"' in resume
    assert 'mcp_servers.uefn.default_tools_approval_mode="approve"' in resume
    assert "--dangerously-bypass-approvals-and-sandbox" in resume
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


def test_ducky_mcp_is_required_on_new_and_resumed_turns_but_not_in_shared_config(tmp_path: Path):
    # Codex waits only 1 s for optional MCP servers; Ducky's bridge needs longer.
    for session_id in ("", "th-abc"):
        argv = build_codex_argv(
            binary="codex",
            prompt="hi",
            model="gpt-5",
            extra_args="",
            session_id=session_id,
            extra_flags=ducky_launch_flags(),
        )
        assert "mcp_servers.uefn.required=true" in argv, session_id
        assert 'mcp_servers.uefn.default_tools_approval_mode="approve"' in argv
    mcp = tmp_path / "mcp.json"
    mcp.write_text(json.dumps({"mcpServers": {"uefn": {"command": "node", "args": ["b.mjs"]}}}), encoding="utf-8")
    home = tmp_path / "codex"
    write_codex_uefn_profile(str(mcp), codex_home=home)
    # The user's own Codex sessions read config.toml too: they must not fail when Ducky is closed.
    assert "required" not in (home / "config.toml").read_text(encoding="utf-8")


def test_every_turn_says_where_ducky_tools_are():
    text = with_ducky_tools_hint("Continue.")
    assert "tools.mcp__uefn__" in text and text.endswith("Continue.")


def test_a_normal_message_fits_the_command_line_of_each_launcher():
    assert inline_prompt_limit(r"C:\Programs\OpenAI\Codex\bin\codex.exe") >= 8000
    assert inline_prompt_limit(r"C:\Roaming\npm\codex.cmd") <= 2500  # cmd.exe stops at 8,191


def test_write_strips_unmarked_uefn_duplicate(tmp_path: Path):
    mcp = tmp_path / "mcp.json"
    mcp.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "uefn": {
                        "command": r"C:\Ducky\UEFN-Ducky-Bridge.exe",
                        "args": ["bridge"],
                        "env": {"DUCKY_CONV_ID": "chat-1"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "config.toml").write_text(
        "[mcp_servers.node_repl]\ncommand = \"x\"\n\n"
        "[mcp_servers.uefn]\ncommand = \"old.exe\"\n"
        "[mcp_servers.uefn.env]\nFOO = \"1\"\n",
        encoding="utf-8",
    )
    write_codex_uefn_profile(str(mcp), codex_home=home)
    cfg = (home / "config.toml").read_text(encoding="utf-8")
    headers = [ln.strip() for ln in cfg.splitlines() if ln.startswith("[")]
    assert headers.count("[mcp_servers.uefn]") == 1
    assert headers.count("[mcp_servers.uefn.env]") == 1
    assert "node_repl" in cfg
    assert "old.exe" not in cfg
    assert "UEFN-Ducky-Bridge.exe" in cfg


def test_heal_strips_unmarked_uefn_on_plugin_load(tmp_path: Path):
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "config.toml").write_text(
        "[mcp_servers.node_repl]\ncommand = \"x\"\n\n"
        "[mcp_servers.uefn]\ncommand = \"old.exe\"\n\n"
        "# BEGIN UEFN-DUCKY\n"
        'approval_policy = "on-failure"\n'
        "[mcp_servers.uefn]\n"
        'command = "bridge.exe"\n'
        "# END UEFN-DUCKY\n",
        encoding="utf-8",
    )
    assert heal_codex_approval_policy(codex_home=home) is True
    cfg = (home / "config.toml").read_text(encoding="utf-8")
    headers = [ln.strip() for ln in cfg.splitlines() if ln.startswith("[")]
    assert headers.count("[mcp_servers.uefn]") == 1
    assert "old.exe" not in cfg
    assert "bridge.exe" in cfg
    assert "node_repl" in cfg


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
    test_resume_compacts_before_the_thread_gets_too_big()
    test_reasoning_effort_flag()
    test_first_turn_has_no_resume()
    test_adapter_opts_into_resume()
    test_parse_codex_catalog_lists_visible_only()
    with tempfile.TemporaryDirectory() as raw:
        cache_home = Path(raw) / "cache"
        cache_home.mkdir()
        test_codex_models_reads_cli_cache(cache_home)
        miss_home = Path(raw) / "miss"
        miss_home.mkdir()
        test_codex_models_without_cli_cache_uses_fallback(miss_home)
    test_auto_model_omits_dash_m()
    test_first_turn_uses_bypass_not_approve_for_me()
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        write_home = root / "write"
        write_home.mkdir()
        test_writes_codex_profile_from_ducky_mcp(write_home)
        dup_home = root / "dup"
        dup_home.mkdir()
        test_write_strips_unmarked_uefn_duplicate(dup_home)
        load_home = root / "load"
        load_home.mkdir()
        test_heal_strips_unmarked_uefn_on_plugin_load(load_home)
        heal_home = root / "heal"
        heal_home.mkdir()
        test_heal_codex_approval_inserts_without_wiping_mcp(heal_home)
        never_home = root / "never"
        never_home.mkdir()
        test_heal_codex_approval_rewrites_never(never_home)
    print("ok")
