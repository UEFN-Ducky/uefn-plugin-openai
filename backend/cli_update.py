"""Keep the Codex CLI current for Ducky-only users."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

_BIN = "codex"
_NPM_PKG = "@openai/codex"
_INSTALL_PS = "irm https://chatgpt.com/codex/install.ps1 | iex"
_VERSION_RE = re.compile(r"(\d+(?:\.\d+)+)")
_REQUIRED_RE = re.compile(
    r"does not support this model;\s*version\s+(\d+(?:\.\d+)+)\s+or newer",
    re.I,
)
_MISSING_MARKERS = (
    "is not recognized",
    "no such file or directory",
    "cannot find the path",
    "not found in path",
)
# npm's Windows vendor exe can exist and still refuse to load (openai/codex#24752).
_BROKEN_BIN_MARKERS = (
    "spawn efault",
    "cannot execute the specified program",
    "the system cannot execute",
    "bad address",
)
_update_lock = threading.Lock()
_busy = False
_last: dict[str, Any] = {}


def plugin_package_version() -> str:
    path = Path(__file__).resolve().parent.parent / "plugin.json"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("version") or "")
    except (OSError, json.JSONDecodeError, TypeError):
        return ""


def parse_version_tuple(text: str) -> tuple[int, ...] | None:
    m = _VERSION_RE.search(text or "")
    if not m:
        return None
    try:
        return tuple(int(p) for p in m.group(1).split("."))
    except ValueError:
        return None


def is_cli_too_old_error(*chunks: str) -> bool:
    text = "\n".join(c or "" for c in chunks)
    if _REQUIRED_RE.search(text):
        return True
    low = text.lower()
    return "does not support this model" in low and "update" in low


def is_cli_missing_error(*chunks: str) -> bool:
    low = "\n".join(c or "" for c in chunks).lower()
    return any(m in low for m in _MISSING_MARKERS)


def is_cli_broken_binary_error(*chunks: str) -> bool:
    low = "\n".join(c or "" for c in chunks).lower()
    return any(m in low for m in _BROKEN_BIN_MARKERS)


def should_heal_launch(*chunks: str) -> bool:
    return (
        is_cli_too_old_error(*chunks)
        or is_cli_missing_error(*chunks)
        or is_cli_broken_binary_error(*chunks)
    )


def is_node_codex_wrapper(path: str) -> bool:
    """npm's `codex` / `codex.cmd` is a Node spawn of a vendor exe that often EFAULTs."""
    low = (path or "").replace("\\", "/").lower()
    if not low:
        return False
    name = low.rsplit("/", 1)[-1]
    if name in {"codex.cmd", "codex.ps1", "codex.js"}:
        return True
    if "/node_modules/@openai/codex" in low:
        return True
    if "/roaming/npm/" in low and name in {"codex", "codex.exe"}:
        return True
    return False


def _codex_home() -> Path:
    return Path.home() / ".codex"


def native_codex_candidates() -> list[str]:
    home = _codex_home()
    found: list[Path] = []
    releases = home / "packages" / "standalone" / "releases"
    if releases.is_dir():
        for release in releases.iterdir():
            for name in ("codex.exe", "codex"):
                cand = release / "bin" / name
                if cand.is_file():
                    found.append(cand)
    extra = home / "plugins" / ".plugin-appserver" / ("codex.exe" if os.name == "nt" else "codex")
    if extra.is_file():
        found.append(extra)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for p in found:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(str(p.resolve()))
    return out


def _path_codex(override: str = "") -> str:
    from backend.agent.coding_agents.base import which_cli

    return which_cli(_BIN, override)


def resolve_bin(override: str = "") -> str:
    """Prefer a native `codex.exe` that Windows will load over the npm Node shim."""
    found = _path_codex(override)
    if found and not is_node_codex_wrapper(found):
        return found
    natives = native_codex_candidates()
    if natives:
        return natives[0]
    return "" if is_node_codex_wrapper(found) else (found or "")


def stamp_path() -> Path:
    from frontend.settings import default_app_data_dir

    return default_app_data_dir() / "coding_agents" / "codex_cli.json"


def read_stamp() -> dict[str, Any]:
    try:
        data = json.loads(stamp_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_stamp(data: dict[str, Any]) -> None:
    path = stamp_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def status_text() -> str:
    if _busy:
        return "Updating Codex CLI…"
    return str(_last.get("message") or "")


def _env() -> dict[str, str]:
    try:
        from frontend.ui_web.terminal.path_env import env_with_fresh_path, refresh_process_path

        refresh_process_path()
        return env_with_fresh_path()
    except Exception:
        return dict(os.environ)


def _run(argv: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": timeout_s,
        "encoding": "utf-8",
        "errors": "replace",
        "env": _env(),
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(argv, **kwargs)


def read_cli_version(binary: str) -> str:
    if not (binary or "").strip():
        return ""
    try:
        proc = _run([binary, "--version"], timeout_s=20)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    tup = parse_version_tuple((proc.stdout or "") + "\n" + (proc.stderr or ""))
    return ".".join(str(p) for p in tup) if tup else ""


def _npm_install() -> dict[str, Any]:
    from backend.agent.coding_agents.base import which_cli

    npm = which_cli("npm") or which_cli("npm.cmd")
    if not npm:
        return {"ok": False, "error": "npm not found"}
    try:
        proc = _run([npm, "install", "-g", _NPM_PKG], timeout_s=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout or "npm install failed").strip()}
    return {"ok": True}


def _ps_install() -> dict[str, Any]:
    if os.name != "nt":
        return {"ok": False, "error": "no PowerShell installer"}
    ps = f"Set-ExecutionPolicy -Scope Process Bypass -Force; {_INSTALL_PS}"
    try:
        proc = _run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            timeout_s=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout or "install failed").strip()}
    return {"ok": True}


def _install_cli() -> dict[str, Any]:
    first = _ps_install() if os.name == "nt" else {"ok": False}
    if first.get("ok") and resolve_bin():
        return first
    # npm's win32 vendor exe is what spawn EFAULT'd — don't reinstall it if
    # the official standalone package already has a loadable binary.
    if native_codex_candidates():
        return {"ok": True}
    return _npm_install()


def update_cli(override: str = "") -> dict[str, Any]:
    global _busy, _last
    with _update_lock:
        _busy = True
        try:
            before_bin = resolve_bin(override)
            before = read_cli_version(before_bin) if before_bin else ""
            if before_bin:
                try:
                    proc = _run([before_bin, "update"], timeout_s=300)
                    if proc.returncode != 0:
                        installed = _install_cli()
                        if not installed.get("ok"):
                            _last = installed
                            return installed
                except (OSError, subprocess.TimeoutExpired):
                    installed = _install_cli()
                    if not installed.get("ok"):
                        _last = installed
                        return installed
            else:
                installed = _install_cli()
                if not installed.get("ok"):
                    _last = installed
                    return installed
            after_bin = resolve_bin(override) or before_bin
            after = read_cli_version(after_bin) if after_bin else ""
            result = {
                "ok": bool(after_bin),
                "cli_path": after_bin,
                "version_before": before,
                "version_after": after,
                "message": f"Codex CLI {after}" if after else "Codex CLI updated",
                "error": "" if after_bin else "Codex CLI still not found after install/update",
            }
            if result["ok"]:
                write_stamp(
                    {
                        "plugin_version": plugin_package_version(),
                        "cli_version": after,
                        "updated_at": time.time(),
                        "ok": True,
                    }
                )
            _last = result
            return result
        finally:
            _busy = False


def needs_plugin_load_update() -> bool:
    if not resolve_bin():
        return True
    stamp = read_stamp()
    if not stamp.get("ok"):
        return True
    return str(stamp.get("plugin_version") or "") != plugin_package_version()


def schedule_cli_update_on_plugin_load() -> None:
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("DUCKY_SKIP_CLI_UPDATE"):
        return
    try:
        if not needs_plugin_load_update():
            return
    except Exception:
        return

    def _worker() -> None:
        try:
            update_cli("")
        except Exception:
            pass

    threading.Thread(target=_worker, name="codex-cli-update", daemon=True).start()
