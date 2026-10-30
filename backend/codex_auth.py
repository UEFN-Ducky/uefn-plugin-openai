"""Codex sign-in: Settings → LLMs → OpenAI → Log in, and a chat whose Codex is signed out.

Device-code sign-in first: its link and one-time code work from any device, so a phone or
another PC showing this Ducky can sign this PC's Codex in. When OpenAI hands out no device
code, Codex's own browser sign-in, which finishes only in a browser on this PC.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any

SETTINGS_LOGIN_HREF = "ducky://settings.llms/openai#login"
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x07")
_URL_RE = re.compile(r"https://auth\.openai\.com/[^\s\"'<>]+")
_CODE_RE = re.compile(r"one-time code[^\n]*\n\s*([A-Z0-9]{3,}(?:-[A-Z0-9]{3,})+)")
_LINK_WAIT_S = 12.0
_TAB_CLOSED = "The Codex Login tab closed. Press Log in again."
# ponytail: one sign-in at a time (the Settings modal is the only place that starts one).
_login: dict[str, Any] = {}


def _no_window() -> dict[str, Any]:
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}


def _codex(cli_path: str, *args: str, home: str = "") -> subprocess.CompletedProcess[str] | None:
    from .cli_update import resolve_bin

    binary = resolve_bin(cli_path)
    if not binary:
        return None
    env = {**os.environ, **account_env(home)} if home else None
    try:
        return subprocess.run([binary, *args], capture_output=True, text=True, timeout=20,
                              encoding="utf-8", errors="replace", env=env, **_no_window())
    except (OSError, subprocess.SubprocessError):
        return None


def codex_logged_in(cli_path: str = "", home: str = "") -> bool | None:
    """`codex login status`: True / False, or None when Codex could not say."""
    proc = _codex(cli_path, "login", "status", home=home)
    if proc is None:
        return None
    text = f"{proc.stdout}\n{proc.stderr}".lower()
    if "not logged in" in text:
        return False
    return True if proc.returncode == 0 and "logged in" in text else None


def parse_login_output(text: str) -> dict[str, str]:
    """The sign-in link and one-time code Codex printed so far ('' until they show)."""
    clean = _ANSI_RE.sub("", text or "").replace("\r", "")
    urls = _URL_RE.findall(clean)
    code = _CODE_RE.search(clean)
    return {"auth_url": urls[-1].rstrip(").,]") if urls else "", "user_code": code.group(1) if code else ""}


def _session() -> Any:
    sid = str(_login.get("session_id") or "")
    if not sid:
        return None
    try:
        from frontend.ui_web.terminal import get_terminal_manager

        session = get_terminal_manager().get_session(sid)
    except Exception:
        return None
    return session if session is not None and session.is_alive() else None


def cancel_login(**_kw: Any) -> dict[str, Any]:
    """Close the Codex Login tab (modal cancelled, finished, or a new sign-in)."""
    sid = str(_login.get("session_id") or "")
    _login.clear()
    if sid:
        try:
            from frontend.ui_web.terminal import get_terminal_manager

            get_terminal_manager().kill(sid, push_close=True)
        except Exception:
            pass
    return {"ok": True, "cancelled": True}


def _wait_for_link(ui: str) -> dict[str, str]:
    """Output of the running login once its link (and, for a device code, the code) shows."""
    deadline = time.time() + _LINK_WAIT_S
    while time.time() < deadline:
        time.sleep(0.25)
        session = _session()
        tail = session.read_output_tail(16000) if session is not None else ""
        got = parse_login_output(tail)
        if got["auth_url"] and (ui == "browser" or got["user_code"]):
            return got
        if session is None or "error" in tail.lower():
            break
    return {}


def start_login(*, cwd: str = "", cli_path: str = "", **_kw: Any) -> dict[str, Any]:
    from frontend.ui_web.terminal import get_terminal_manager

    from .cli_update import resolve_bin

    cancel_login()
    binary = resolve_bin(cli_path)
    if not binary:
        return {"ok": False, "error": "Codex CLI not found. Press Detect on the Codex row, then Log in again."}
    if codex_logged_in(cli_path):
        return {"ok": True, "logged_in": True}
    workdir = cwd if cwd and os.path.isdir(cwd) else os.getcwd()
    for args, ui in ((["login", "--device-auth"], "device_code"), (["login"], "browser")):
        spawn = get_terminal_manager().spawn(
            shell="powershell" if os.name == "nt" else "bash", cwd=workdir, title="Codex Login",
            push_open=True, conv_id="__settings__", command=[binary, *args], activate=False,
        )
        sid = str(spawn.get("session_id") or spawn.get("id") or "")
        if not spawn.get("ok") or not sid:
            return {"ok": False, "error": str(spawn.get("error") or "Could not start the Codex Login tab.")}
        _login.update(session_id=sid, login_ui=ui)
        got = _wait_for_link(ui)
        if got:
            _login.update(got)
            return {"ok": True, "logged_in": False, "needs_login": True, "login_ui": ui,
                    "terminal_session_id": sid, **got}
        cancel_login()
    return {"ok": False, "error": "Codex never printed a sign-in link. Check the Codex Login tab, then press Log in again."}


def login_status(*, cli_path: str = "", **_kw: Any) -> dict[str, Any]:
    """The modal's poll: the link and code, or logged_in once Codex is signed in."""
    if not _login:
        return {"ok": True, "logged_in": False}
    session = _session()
    tail = session.read_output_tail(16000).lower() if session is not None else ""
    # Ask Codex only once its tab says so or has closed: not a process every poll.
    if (session is None or "logged in" in tail) and codex_logged_in(cli_path):
        cancel_login()
        return {"ok": True, "logged_in": True}
    row = {"ok": True, "logged_in": False, "needs_login": True, "login_ui": _login.get("login_ui", ""),
           "auth_url": _login.get("auth_url", ""), "user_code": _login.get("user_code", ""),
           "terminal_session_id": _login.get("session_id", ""), "session_alive": session is not None}
    return row if session is not None else {**row, "auth_url": "", "error": _TAB_CLOSED}


def logout(*, cli_path: str = "", **_kw: Any) -> dict[str, Any]:
    proc = _codex(cli_path, "logout")
    if proc is None:
        return {"ok": False, "error": "Codex CLI not found"}
    if codex_logged_in(cli_path):
        return {"ok": False, "logged_in": True, "error": (proc.stderr or proc.stdout or "Logout did not take effect").strip()[:300]}
    return {"ok": True, "logged_in": False, "message": "Logged out of Codex."}


# Extra accounts: core keeps the list and the folder; each Codex login lives in its own CODEX_HOME.
# Shared with ~/.codex so every account keeps the person's settings, prompts and skills; auth.json,
# sessions and history stay per account. Ducky's MCP server rides each launch as a `-c` override,
# so a shared config.toml never outranks it.
ACCOUNT_SHARED = {"dirs": ["prompts", "skills"], "files": ["config.toml", "AGENTS.md"]}
_LIMIT_RE = re.compile(r"usage[ _]limit|usage_limit_reached|rate_limit_exceeded|hit your (usage )?limit"
                       r"|too many requests|\b429\b", re.I)
_QUOTA_RE = re.compile(r"\b429\b|rate[ _]limit|insufficient_quota|quota|too many requests", re.I)


def account_env(config_dir: str) -> dict[str, str]:
    return {"CODEX_HOME": str(config_dir)} if config_dir else {}


def account_login(config_dir: str, *, cli_path: str = "", **_kw: Any) -> dict[str, Any]:
    """Core's login terminal: device-code sign-in, which works from any device."""
    from .cli_update import resolve_bin

    binary = resolve_bin(cli_path)
    if not binary:
        return {"ok": False, "error": "Codex CLI not found. Press Detect on the Codex row, then Log in again."}
    os.makedirs(config_dir, exist_ok=True)
    return {"argv": [binary, "login", "--device-auth"], "env": account_env(config_dir)}


def account_status(config_dir: str, *, cli_path: str = "", **_kw: Any) -> dict[str, Any]:
    return {"loggedIn": bool(codex_logged_in(cli_path, home=config_dir))}


def account_logout(config_dir: str, *, cli_path: str = "", **_kw: Any) -> dict[str, Any]:
    proc = _codex(cli_path, "logout", home=config_dir)
    if proc is None:
        return {"ok": False, "error": "Codex CLI not found"}
    return {"ok": not codex_logged_in(cli_path, home=config_dir)}


def is_limit_error(text: str) -> bool:
    """A Codex turn that stopped on the ChatGPT plan's usage limit."""
    return bool(_LIMIT_RE.search(text or ""))


def is_quota_error(text: str) -> bool:
    """An OpenAI API key out of quota or rate limited."""
    return bool(_QUOTA_RE.search(text or ""))


def account_usage(config_dir: str, **kw: Any) -> list[dict[str, Any]]:
    from .usage import account_usage as usage

    return usage(config_dir, **kw)


# register_coding_agent("codex", **ACCOUNT_HOOKS): core's per-account login, status and limits.
ACCOUNT_HOOKS: dict[str, Any] = {
    "account_env": account_env, "account_login": account_login, "account_status": account_status,
    "account_logout": account_logout, "account_shared": ACCOUNT_SHARED, "account_usage": account_usage,
    "is_limit_error": is_limit_error,
}


def login_message(push: Any) -> str:
    """A chat turn whose Codex is signed out: point to Settings and spotlight Log in."""
    if push:
        try:
            push({
                "type": "open_coding_agent_login",
                "provider_id": "openai",
                "title": "Sign in to Codex",
                "text": "Press Log in. Open the link on any device and enter the code it shows.",
            })
        except Exception:
            pass
    return "\n".join([
        "## Codex is not signed in",
        "",
        f"[Open Settings → OpenAI and log in]({SETTINGS_LOGIN_HREF})",
        "",
        ("Press **Log in** there, open the link on any device and enter the code it shows. "
         "Then send your message again."),
    ])
