"""codex_image: one `codex exec` turn on Codex's built-in image tool, its picture copied out."""
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path


def _module():
    pkg = types.ModuleType("_codex_images_pkg")
    pkg.__path__ = [str(Path(__file__).parent)]
    sys.modules["_codex_images_pkg"] = pkg
    spec = importlib.util.spec_from_file_location("_codex_images_pkg.codex_images",
                                                  Path(__file__).with_name("codex_images.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_helpers(monkeypatch, logged_in=True, binary="codex.exe"):
    monkeypatch.setitem(sys.modules, "_codex_images_pkg.cli_update",
                        types.SimpleNamespace(resolve_bin=lambda _p: binary))
    monkeypatch.setitem(sys.modules, "_codex_images_pkg.codex_auth",
                        types.SimpleNamespace(codex_logged_in=lambda _p: logged_in))


def test_argv_reads_the_prompt_on_stdin_and_switches_off_mcp(tmp_path):
    mod = _module()
    (tmp_path / "config.toml").write_text('[mcp_servers.uefn]\nurl = "http://127.0.0.1:1/mcp"\n', encoding="utf-8")
    argv = mod.build_argv("codex", tmp_path, "W", "ref.png")
    assert argv[:2] == ["codex", "exec"] and argv[-1] == "-"
    assert "features.shell_tool=false" in argv and "features.image_generation=true" in argv
    assert 'mcp_servers.uefn={url="http://127.0.0.1:1/mcp", enabled=false}' in argv
    assert argv[argv.index("--image") + 1] == "ref.png"
    assert argv[argv.index("--sandbox") + 1] == "read-only"


def test_the_picture_codex_saved_is_copied_out(tmp_path, monkeypatch):
    mod = _module()
    _fake_helpers(monkeypatch)
    home = tmp_path / "home"
    monkeypatch.setenv("CODEX_HOME", str(home))
    old = home / "generated_images" / "th-1" / "old.png"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    os.utime(old, (time.time() - 120, time.time() - 120))
    seen = {}

    def run(argv, input, **_kw):
        seen["argv"], seen["input"] = argv, input
        (home / "generated_images" / "th-1" / "new.png").write_bytes(b"png")
        out = json.dumps({"type": "thread.started", "thread_id": "th-1"})
        return types.SimpleNamespace(stdout=out, stderr="", returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", run)
    data = json.loads(mod.codex_image("a duck", output_dir=str(tmp_path / "out")))
    path = Path(data["downloaded"][0])
    assert path.parent == tmp_path / "out" and path.read_bytes() == b"png"
    assert seen["input"].endswith("a duck") and "image_gen" in seen["input"]


def test_signed_out_and_no_picture_say_why(tmp_path, monkeypatch):
    mod = _module()
    _fake_helpers(monkeypatch, logged_in=False)
    assert "signed out" in mod.codex_image("a duck")
    _fake_helpers(monkeypatch)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    failed = json.dumps({"type": "turn.failed", "error": {"message": "usage limit reached"}})
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda *_a, **_k: types.SimpleNamespace(stdout=failed, stderr="", returncode=1))
    assert mod.codex_image("a duck") == "Error: usage limit reached"
    assert mod.codex_image("").startswith("Error")
