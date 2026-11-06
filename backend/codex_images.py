"""Pictures made by Codex's own image tool, on the person's ChatGPT sign-in (no API key).

Text to Image and ducky_generate_image run ``codex_image``: one ``codex exec`` turn told to
use its built-in image_gen tool. Codex saves the picture under
CODEX_HOME/generated_images/<thread id>/; it is copied into output_dir.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any

TOOL = "codex_image"
_TIMEOUT_S = 600.0
_MAX_PROMPT = 4000
_PICTURES = (".png", ".jpg", ".jpeg", ".webp")
_INSTRUCTION = (
    "Make exactly one picture with your built-in image generation tool (image_gen), "
    "then stop. Do not run commands, write files or use any other tool. "
    "Reply with one short line when it is done.\n\nPicture request:\n"
)
# Codex here only draws: no shell, no web, no MCP servers (Ducky's own included,
# so it cannot call ducky_generate_image back).
_FLAGS = [
    "--sandbox", "read-only",
    "--skip-git-repo-check",
    "--json",
    "-c", 'approval_policy="never"',
    "-c", "features.image_generation=true",
    "-c", "features.shell_tool=false",
    "-c", "features.unified_exec=false",
    "-c", "features.multi_agent=false",
    "-c", "features.browser_use=false",
    "-c", "features.computer_use=false",
    "-c", 'web_search="disabled"',
]


def _error(text: str) -> str:
    return "Error: " + text


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def _mcp_off_flags(home: Path) -> list[str]:
    """Switch off every MCP server Codex's own config.toml declares (each keeps its
    transport so the override stays a valid server)."""
    try:
        data = tomllib.loads((home / "config.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    flags: list[str] = []
    servers = data.get("mcp_servers")
    for name, server in servers.items() if isinstance(servers, dict) else ():
        if not isinstance(server, dict) or not name.replace("_", "").replace("-", "").isalnum():
            continue
        if isinstance(server.get("url"), str):
            transport = "url=" + json.dumps(server["url"])
        elif isinstance(server.get("command"), str):
            transport = "command=" + json.dumps(server["command"])
        else:
            continue
        flags += ["-c", f"mcp_servers.{name}={{{transport}, enabled=false}}"]
    return flags


def build_argv(binary: str, home: Path, workdir: str, image: str = "") -> list[str]:
    """``codex exec`` for one picture; the prompt goes on stdin ("-") so no shell or
    .cmd shim ever re-reads it."""
    argv = [binary, "exec", *_FLAGS, *_mcp_off_flags(home), "-C", workdir]
    if image:
        argv += ["--image", image]
    return argv + ["-"]


def _thread_id(stdout: str) -> str:
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "thread.started":
            return str(event.get("thread_id") or "")
    return ""


def _failure(stdout: str, stderr: str) -> str:
    """Codex's own error words, from its JSON events or its stderr."""
    for line in reversed(stdout.splitlines()):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "error" and event.get("message"):
            return str(event["message"])
        failed = event.get("error")
        if event.get("type") == "turn.failed" and isinstance(failed, dict) and failed.get("message"):
            return str(failed["message"])
    tail = [line for line in stderr.splitlines() if line.strip()]
    return tail[-1][:300] if tail else ""


def new_pictures(home: Path, thread_id: str, since: float) -> list[Path]:
    """Pictures Codex saved since ``since``: in this thread's folder when known, else anywhere
    under generated_images/ (an older Codex that names its folders otherwise)."""
    root = home / "generated_images"
    folder = root / thread_id if thread_id and (root / thread_id).is_dir() else root
    if not folder.is_dir():
        return []
    found: list[Path] = []
    for path in folder.rglob("*"):
        try:
            if path.suffix.lower() in _PICTURES and path.is_file() and path.stat().st_mtime >= since - 1:
                found.append(path)
        except OSError:
            continue
    return sorted(found, key=lambda item: item.stat().st_mtime)


def _out_dir(output_dir: str) -> Path:
    if str(output_dir or "").strip():
        return Path(output_dir)
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "UEFN-Ducky" / "codex" / "images"


def _no_window() -> dict[str, Any]:
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}


def codex_image(
    prompt: str,
    image: str = "",
    wait: bool = True,
    output_dir: str = "",
    confirm_spend: bool = False,
) -> str:
    """Make ONE picture with Codex's built-in image tool, on your ChatGPT sign-in (no API
    key; it counts against your ChatGPT plan's Codex limits). image (a picture's path) is a
    reference the new picture starts from. Takes up to a few minutes. Saves the picture in
    output_dir and returns JSON {downloaded:[path]}."""
    _ = wait, confirm_spend  # one call that returns the picture; no money is spent
    text = str(prompt or "").strip()
    if not text:
        return _error("prompt is empty")
    if len(text) > _MAX_PROMPT:
        return _error(f"prompt is longer than {_MAX_PROMPT} characters")
    ref = str(image or "").strip()
    if ref and not Path(ref).is_file():
        return _error(f"the reference picture was not found: {ref}")
    from .cli_update import resolve_bin
    from .codex_auth import codex_logged_in

    binary = resolve_bin("")
    if not binary:
        return _error("Codex CLI not found. Settings > LLMs > OpenAI > Codex: press Detect.")
    if codex_logged_in("") is False:
        return _error("Codex is signed out. Settings > LLMs > OpenAI: press Log in.")
    home = _codex_home()
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="ducky-codex-image-") as workdir:
        try:
            proc = subprocess.run(
                build_argv(binary, home, workdir, ref), input=_INSTRUCTION + text, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=_TIMEOUT_S, cwd=workdir, **_no_window(),
            )
        except subprocess.TimeoutExpired:
            return _error(f"Codex made no picture in {int(_TIMEOUT_S // 60)} minutes")
        except OSError as exc:
            return _error(f"Codex could not start: {exc}")
    made = new_pictures(home, _thread_id(proc.stdout), started)
    if not made:
        why = _failure(proc.stdout, proc.stderr)
        if "401" in why or "unauthorized" in why.lower():
            return _error("Codex is signed out. Settings > LLMs > OpenAI: press Log in.")
        return _error(why or f"Codex finished without a picture (exit code {proc.returncode})")
    folder = _out_dir(output_dir)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        src = made[-1]
        dest = folder / f"codex-{stamp}{src.suffix.lower()}"
        n = 1
        while dest.exists():
            n += 1
            dest = folder / f"codex-{stamp}-{n}{src.suffix.lower()}"
        shutil.copyfile(src, dest)
    except OSError as exc:
        return _error(f"the picture could not be saved: {exc}")
    return json.dumps({"ok": True, "downloaded": [str(dest)], "model": "codex image_gen"})
