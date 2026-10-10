"""Run-local MCP overrides leave Codex authentication/history at CODEX_HOME."""
import json
import importlib.util
import threading
import tomllib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

# Other legacy tests replace the backend package during collection. Load this
# existing fixture by path without depending on that mutable package identity.
_spec = importlib.util.spec_from_file_location("_config_readiness_fixture", Path(__file__).with_name("test_ducky_readiness_launch.py"))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
launch_env = _fixture.launch_env


def test_concurrent_and_resumed_launches_keep_their_identity(launch_env, tmp_path):
    adapter, process, update, config, events, kwargs, resolve = launch_env
    home = tmp_path / "selected-home"
    home.mkdir()
    shared = home / "config.toml"
    shared.write_text('model = "user-default"\n', encoding="utf-8")
    barrier = threading.Barrier(2)
    seen = []
    def run(**call):
        if call["conv_id"] in ("a", "b"):
            barrier.wait(timeout=5)
        overrides = [x for x in call["argv"] if x.startswith("mcp_servers.uefn={")]
        assert len(overrides) == 1
        server = tomllib.loads(overrides[0])["mcp_servers"]["uefn"]
        assert server["env"]["DUCKY_CONV_ID"] == call["conv_id"]
        assert server["args"][-1] == server["env"]["DUCKY_RUN_ID"]
        assert call["env_extra"]["CODEX_HOME"] == str(home)
        seen.append(server)
        call["on_line"](json.dumps({"type": "item.completed", "item": {"id": "ok", "type": "agent_message", "text": "Done"}}))
        return SimpleNamespace(cancelled=False, timed_out=False, returncode=0, stderr_tail="", raw_tail="")
    process.side_effect = run
    def launch(chat, session=""):
        path = tmp_path / (chat + ".json")
        path.write_text(json.dumps({"mcpServers": {"uefn": {"command": "node", "args": ["bridge", "--ducky-run-id", chat + "-run"],
                         "env": {"DUCKY_CONV_ID": chat, "DUCKY_RUN_ID": chat + "-run"}}}}), encoding="utf-8")
        return adapter.CodexAdapter().launch(**{**kwargs, "conv_id": chat, "run_id": chat + "-run", "mcp_config_path": str(path),
                                              "env": {"CODEX_HOME": str(home)}}, session_id=session)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(launch, chat) for chat in ("a", "b")]
        assert all(f.result(timeout=10).ok for f in futures)
    assert launch("resumed-a", "thread-a").ok
    assert len(seen) == 3
    assert shared.read_text(encoding="utf-8") == 'model = "user-default"\n'
    assert not (home / "ducky-uefn.config.toml").exists()


def test_legacy_writer_respects_codex_home_and_replaces_atomically(launch_env, tmp_path, monkeypatch):
    adapter, _, _, config, _, _, _ = launch_env
    home = tmp_path / "selected"
    monkeypatch.setenv("CODEX_HOME", str(home))
    replacements = []
    replace = adapter.os.replace
    def spy(src, dst):
        replacements.append(dst)
        tomllib.loads(src.read_text(encoding="utf-8"))
        replace(src, dst)
    monkeypatch.setattr(adapter.os, "replace", spy)
    assert adapter.write_codex_uefn_profile(str(config))
    assert (home / "config.toml").is_file()
    assert set(replacements) == {home / "config.toml", home / "ducky-uefn.config.toml"}
    assert not list(home.glob("*.tmp"))


def test_failed_atomic_replace_preserves_old_file(launch_env, tmp_path, monkeypatch):
    adapter, _, _, config, _, _, _ = launch_env
    target = tmp_path / "config.toml"
    target.write_text('model="original"\n', encoding="utf-8")
    def fail(*args): raise OSError("synthetic")
    monkeypatch.setattr(adapter.os, "replace", fail)
    with pytest.raises(OSError):
        adapter._atomic_write_text(target, 'model="new"\n')
    assert target.read_text(encoding="utf-8") == 'model="original"\n'
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("blocked", ["directory", "config"])
def test_legacy_write_failure_does_not_block_run_local_snapshot(launch_env, tmp_path, blocked):
    adapter, _, _, config, _, _, _ = launch_env
    home = tmp_path / "blocked-home"
    if blocked == "directory":
        home.write_text("not a directory", encoding="utf-8")
    else:
        (home / "config.toml").mkdir(parents=True)
    assert adapter.write_codex_uefn_profile(str(config), codex_home=home) == ""
    assert tomllib.loads(adapter.write_codex_uefn_profile(str(config), per_run=True))["mcp_servers"]["uefn"]["required"]


def test_run_snapshot_escapes_values_and_clears_old_identity(launch_env, tmp_path):
    adapter, _, _, config, _, _, _ = launch_env
    original = {"command": "node", "args": ["line\nquote\"slash\\"], "env": {"A.B": "\tquote\"line\n"}}
    config.write_text(json.dumps({"mcpServers": {"uefn": original}}), encoding="utf-8")
    snapshot = adapter.write_codex_uefn_profile(str(config), per_run=True)
    config.write_text("{}", encoding="utf-8")
    parsed = tomllib.loads(snapshot)["mcp_servers"]["uefn"]
    assert parsed["args"] == original["args"]
    assert parsed["env"]["A.B"] == original["env"]["A.B"]
    assert parsed["env"]["DUCKY_CONV_ID"] == ""
    assert parsed["env"]["DUCKY_LEADER_CONV_ID"] == ""


def test_atomic_readers_only_observe_complete_config(launch_env, tmp_path):
    adapter, _, _, _, _, _, _ = launch_env
    target = tmp_path / "shared.toml"
    first = 'identity="a"\npayload="' + "a" * 10000 + '"\n'
    second = 'identity="b"\npayload="' + "b" * 10000 + '"\n'
    target.write_text(first, encoding="utf-8")
    barrier = threading.Barrier(2)
    def writer():
        barrier.wait(timeout=5)
        for index in range(30):
            adapter._atomic_write_text(target, first if index % 2 else second)
    def reader():
        barrier.wait(timeout=5)
        for _ in range(100):
            snapshot = target.read_text(encoding="utf-8")
            assert snapshot in (first, second)
            data = tomllib.loads(snapshot)
            assert data["payload"] == data["identity"] * 10000
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(writer), executor.submit(reader)]
        for future in futures:
            future.result(timeout=10)
    assert not list(tmp_path.glob("*.tmp"))


def test_default_home_and_healer_respect_selected_home(launch_env, tmp_path, monkeypatch):
    adapter, _, _, _, _, _, _ = launch_env
    monkeypatch.delenv("CODEX_HOME")
    assert adapter._codex_home() == tmp_path / ".codex"
    home = tmp_path / "selected"
    home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
    config = home / "config.toml"
    config.write_text('approval_policy = "never"\n', encoding="utf-8")
    assert adapter.heal_codex_approval_policy()
    assert tomllib.loads(config.read_text(encoding="utf-8"))["approval_policy"] == "on-failure"
    assert adapter._codex_catalog_path() == home / "models_cache.json"
