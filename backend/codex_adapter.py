"""Codex CLI adapter — `codex exec --json` events + thread resume.

This plugin owns ``codex exec resume <id>``. Core only stores the thread
id this adapter returns and passes it back on the next turn.
"""

from __future__ import annotations

import json
import shlex
import threading
import time
from pathlib import Path
from typing import Any, Callable

from backend.agent.coding_agents.base import (
    CodingAgentCapabilities,
    CodingAgentInfo,
    CodingAgentLaunchResult,
)
from backend.agent.coding_agents.cli_shared import finalize_cli_turn, truncate_tool_result
from backend.agent.coding_agents.proc_exec import run_streaming_process
from backend.agent.coding_agents.settings_helpers import coding_agent_cfg

_CODEX_INSTALL_PS = r"irm https://chatgpt.com/codex/install.ps1 | iex"
_CODEX_INSTALL_NPM = "npm install -g @openai/codex"

# ChatGPT login (no API key) still runs these via `codex exec`. Live /v1/models
# is merged on top when a key exists.
_CODEX_FALLBACK_MODELS: tuple[tuple[str, str], ...] = (
    ("auto", "Auto"),
    ("gpt-5.1-codex", "GPT-5.1 Codex"),
    ("gpt-5.1-codex-mini", "GPT-5.1 Codex Mini"),
    ("gpt-5-codex", "GPT-5 Codex"),
    ("gpt-5", "GPT-5"),
    ("gpt-5-mini", "GPT-5 Mini"),
    ("o3", "o3"),
    ("o4-mini", "o4-mini"),
)


def _is_codex_auto_model(model: str) -> bool:
    return (model or "").strip().lower() in ("", "auto", "default")


def normalize_codex_model(model: str) -> str:
    mid = (model or "").strip()
    return "auto" if _is_codex_auto_model(mid) else mid


def _codex_row(model_id: str, name: str, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": model_id,
        "name": name or model_id,
        "provider": "Codex",
        "supports_tools": True,
    }
    row.update(extra)
    return row


def _codex_fallback_rows() -> list[dict[str, Any]]:
    return [_codex_row(mid, name) for mid, name in _CODEX_FALLBACK_MODELS]


def _codex_model_rows() -> list[dict[str, Any]]:
    """Picker rows. ChatGPT-login users get the fallback list; an API key adds live ids."""
    fallback = _codex_fallback_rows()
    try:
        from backend.agent.secrets import get_key

        key = (get_key("openai") or "").strip()
    except Exception:
        return fallback
    if not key:
        return fallback
    try:
        from .model_fetch import fetch_models

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for info in fetch_models(key):
            mid = (info.id or "").strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            extra: dict[str, Any] = {
                "supports_vision": bool(info.supports_vision),
                "supports_tools": True,
                "supports_web_search": bool(info.supports_web_search),
            }
            if info.context_limit:
                extra["context_limit"] = int(info.context_limit)
            if info.price_in is not None:
                extra["price_in"] = info.price_in
            if info.price_out is not None:
                extra["price_out"] = info.price_out
            rows.append(_codex_row(mid, info.display_name or mid, **extra))
        if not rows:
            return fallback
        if "auto" not in seen:
            rows.insert(0, _codex_row("auto", "Auto"))
        return rows
    except Exception:
        return fallback


def _codex_missing_status() -> str:
    return (
        "Codex CLI not found — Ducky is installing it automatically "
        "(not the ChatGPT desktop app). Send another message in a moment."
    )


def _normalize_codex_extra_args(extra: str) -> list[str]:
    """Map legacy flags (e.g. --full-auto) onto current `codex exec` options."""
    raw = (extra or "").strip()
    if not raw:
        return ["-s", "workspace-write"]
    parts = shlex.split(raw, posix=False)
    out: list[str] = []
    i = 0
    saw_sandbox = False
    while i < len(parts):
        tok = parts[i]
        if tok in ("--full-auto", "-a", "--ask-for-approval"):
            # Old interactive flag — for exec, prefer workspace-write sandbox.
            if not saw_sandbox:
                out.extend(["-s", "workspace-write"])
                saw_sandbox = True
            i += 1
            # Skip optional value after -a / --ask-for-approval
            if tok in ("-a", "--ask-for-approval") and i < len(parts) and not parts[i].startswith("-"):
                i += 1
            continue
        if tok in ("-s", "--sandbox"):
            saw_sandbox = True
            out.append(tok)
            if i + 1 < len(parts):
                out.append(parts[i + 1])
                i += 2
            else:
                i += 1
            continue
        out.append(tok)
        i += 1
    if not saw_sandbox:
        out.extend(["-s", "workspace-write"])
    return out


_CODEX_PROFILE = "ducky-uefn"
_TOML_BEGIN = "# BEGIN UEFN-DUCKY"
_TOML_END = "# END UEFN-DUCKY"
# `codex exec` hard-sets session approval to Never. MCP tools then die with
# "approval policy is never" unless the server is pre-approved (`approve`,
# not `auto`). `-c` on every turn because resume cannot take --approve-for-me.
_APPROVAL_TOML = "approval_policy = \"on-failure\""
_APPROVAL_FLAG = ["-c", 'approval_policy="on-failure"']
_MCP_APPROVE_TOML = 'default_tools_approval_mode = "approve"'
_MCP_APPROVE_FLAG = ["-c", 'mcp_servers.uefn.default_tools_approval_mode="approve"']
# openai/codex#24135: exec ignores default_tools_approval_mode and rejects MCP
# with "approval policy is never". This flag is on `exec` and `exec resume`.
_BYPASS_APPROVALS = ["--dangerously-bypass-approvals-and-sandbox"]
# `codex exec resume` is a smaller clap parser than `codex exec`.
_EXEC_ONLY_VALUE = {
    "-p",
    "--profile",
    "--add-dir",
    "-s",
    "--sandbox",
    "-C",
    "--cd",
    "--local-provider",
    "--color",
}
_EXEC_ONLY_BARE = {"--approve-for-me", "--oss"}


def _toml_quote(value: str) -> str:
    return '"' + (value or "").replace("\\", "\\\\").replace('"', '\\"') + '"'


def _strip_exec_only(parts: list[str]) -> list[str]:
    """Drop flags `codex exec resume` does not accept."""
    out: list[str] = []
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok in _EXEC_ONLY_BARE:
            i += 1
            continue
        if tok in _EXEC_ONLY_VALUE or tok.startswith("-p=") or tok.startswith("--profile="):
            i += 2 if tok in _EXEC_ONLY_VALUE and i + 1 < len(parts) and not str(parts[i + 1]).startswith("-") else 1
            continue
        out.append(tok)
        i += 1
    return out


def _writable_roots_flag(dirs: list[Path]) -> list[str]:
    if not dirs:
        return []
    roots = ", ".join(_toml_quote(str(p)) for p in dirs)
    return ["-c", f"sandbox_workspace_write.writable_roots=[{roots}]"]


def _merge_marked_toml(path: Path, body: str) -> None:
    block = f"{_TOML_BEGIN}\n{body.rstrip()}\n{_TOML_END}\n"
    existing = ""
    if path.is_file():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError:
            existing = ""
    start = existing.find(_TOML_BEGIN)
    end = existing.find(_TOML_END)
    if start >= 0 and end > start:
        new = existing[:start] + block + existing[end + len(_TOML_END) :].lstrip("\n")
    elif start >= 0:
        new = existing[:start] + block
    else:
        new = existing.rstrip() + ("\n\n" if existing.strip() else "") + block
    path.write_text(new, encoding="utf-8")


def heal_codex_approval_policy(*, codex_home: Path | None = None) -> bool:
    """Rewrite stale `never` so Ducky MCP tools run without a user click."""
    home = Path(codex_home) if codex_home else Path.home() / ".codex"
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    changed = False
    cfg = home / "config.toml"
    try:
        text = cfg.read_text(encoding="utf-8") if cfg.is_file() else ""
    except OSError:
        return False
    new = (
        text.replace('approval_policy = "never"', _APPROVAL_TOML)
        .replace("approval_policy = 'never'", _APPROVAL_TOML)
        .replace('default_tools_approval_mode = "auto"', _MCP_APPROVE_TOML)
        .replace("default_tools_approval_mode = 'auto'", _MCP_APPROVE_TOML)
    )
    if "approval_policy" not in new:
        if _TOML_BEGIN in new:
            new = new.replace(_TOML_BEGIN, f"{_TOML_BEGIN}\n{_APPROVAL_TOML}", 1)
        else:
            _merge_marked_toml(cfg, _APPROVAL_TOML + "\n")
            return True
        changed = True
    if "[mcp_servers.uefn]" in new and "default_tools_approval_mode" not in new:
        new = new.replace(
            "[mcp_servers.uefn]", f"[mcp_servers.uefn]\n{_MCP_APPROVE_TOML}", 1
        )
        changed = True
    if new != text:
        try:
            cfg.write_text(new, encoding="utf-8")
        except OSError:
            return False
        changed = True
    extra = home / f"{_CODEX_PROFILE}.config.toml"
    if extra.is_file():
        try:
            et = extra.read_text(encoding="utf-8")
        except OSError:
            et = ""
        en = (
            et.replace('approval_policy = "never"', _APPROVAL_TOML)
            .replace("approval_policy = 'never'", _APPROVAL_TOML)
            .replace('default_tools_approval_mode = "auto"', _MCP_APPROVE_TOML)
            .replace("default_tools_approval_mode = 'auto'", _MCP_APPROVE_TOML)
        )
        if en != et:
            try:
                extra.write_text(en, encoding="utf-8")
                changed = True
            except OSError:
                pass
    return changed


def write_codex_uefn_profile(
    mcp_config_path: str,
    *,
    codex_home: Path | None = None,
) -> str:
    """Write [mcp_servers.uefn] into config.toml (resume has no -p/--profile)."""
    if not mcp_config_path:
        return ""
    src = Path(mcp_config_path)
    if not src.is_file():
        return ""
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    uefn = servers.get("uefn") if isinstance(servers, dict) else None
    if not isinstance(uefn, dict):
        return ""
    command = str(uefn.get("command") or "").strip()
    if not command:
        return ""
    args = [str(a) for a in (uefn.get("args") or [])]
    env = uefn.get("env") if isinstance(uefn.get("env"), dict) else {}
    home = Path(codex_home) if codex_home else Path.home() / ".codex"
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError:
        return ""
    lines = [
        _APPROVAL_TOML,
        "",
        "[mcp_servers.uefn]",
        f"command = {_toml_quote(command)}",
        f"args = [{', '.join(_toml_quote(a) for a in args)}]",
        "enabled = true",
        "startup_timeout_sec = 60.0",
        "tool_timeout_sec = 180.0",
        _MCP_APPROVE_TOML,
    ]
    if env:
        lines.append("")
        lines.append("[mcp_servers.uefn.env]")
        for key in sorted(str(k) for k in env):
            lines.append(f"{key} = {_toml_quote(str(env[key]))}")
    body = "\n".join(lines) + "\n"
    dest = home / f"{_CODEX_PROFILE}.config.toml"
    try:
        dest.write_text("# Generated by UEFN-Ducky — rewritten each Codex launch.\n" + body, encoding="utf-8")
        _merge_marked_toml(home / "config.toml", body)
    except OSError:
        return ""
    return _CODEX_PROFILE


def build_codex_argv(
    *,
    binary: str,
    prompt: str,
    model: str,
    extra_args: str,
    session_id: str,
    image_paths: list[str] | None = None,
    extra_flags: list[str] | None = None,
) -> list[str]:
    """Argv for one `codex exec --json` turn; resumes ``session_id`` when set."""
    argv = [binary, "exec"]
    if session_id:
        argv.extend(["resume", session_id])
    argv.extend(["--skip-git-repo-check", "--json"])
    for path in image_paths or []:
        # Codex exec natively attaches images to the initial prompt.
        argv.extend(["--image", path])
    extra = _normalize_codex_extra_args(extra_args)
    flags = list(extra_flags or [])
    if session_id:
        rewritten: list[str] = []
        i = 0
        while i < len(extra):
            if extra[i] in ("-s", "--sandbox") and i + 1 < len(extra):
                rewritten.extend(["-c", f'sandbox_mode="{extra[i + 1]}"'])
                i += 2
                continue
            rewritten.append(extra[i])
            i += 1
        extra = _strip_exec_only(rewritten)
        flags = _strip_exec_only(flags)
    argv.extend(extra)
    argv.extend(flags)
    if not _is_codex_auto_model(model):
        argv.extend(["-m", model])
    argv.append(prompt)
    return argv


def _mcp_result_text(item: dict[str, Any]) -> str:
    """Flatten mcp_tool_call.result into a short UI string."""
    err = item.get("error")
    if isinstance(err, dict) and err.get("message"):
        return str(err.get("message") or "")
    result = item.get("result")
    if not isinstance(result, dict):
        return truncate_tool_result(str(item.get("aggregated_output") or item.get("output") or ""))
    parts: list[str] = []
    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
    if parts:
        return truncate_tool_result("\n".join(parts))
    structured = result.get("structured_content")
    if structured is not None:
        try:
            return truncate_tool_result(json.dumps(structured, ensure_ascii=False))
        except (TypeError, ValueError):
            return truncate_tool_result(str(structured))
    return truncate_tool_result(str(item.get("aggregated_output") or item.get("output") or ""))


def _tool_args_from_item(itype: str, item: dict[str, Any]) -> dict[str, Any]:
    """Best-effort arguments for the tool card (path/query/command/etc.)."""
    if itype == "mcp_tool_call":
        args = item.get("arguments")
        out: dict[str, Any] = dict(args) if isinstance(args, dict) else {}
        server = str(item.get("server") or "").strip()
        if server:
            out.setdefault("server", server)
        return out
    if itype == "command_execution":
        cmd = str(item.get("command") or "").strip()
        return {"command": cmd} if cmd else {}
    if itype == "web_search":
        query = str(item.get("query") or "").strip()
        return {"query": query} if query else {}
    if itype == "file_change":
        changes = item.get("changes")
        if isinstance(changes, list):
            paths = [
                str(c.get("path") or "")
                for c in changes
                if isinstance(c, dict) and str(c.get("path") or "").strip()
            ]
            if paths:
                return {"paths": paths, "path": paths[0]}
        return {}
    return {}


def _tool_name_from_item(itype: str, item: dict[str, Any]) -> str:
    if itype == "mcp_tool_call":
        tool = str(item.get("tool") or "").strip()
        server = str(item.get("server") or "").strip()
        if tool and server:
            return f"{server}/{tool}"
        return tool or server or itype
    if itype == "command_execution":
        return str(item.get("command") or itype)
    if itype == "web_search":
        return "web_search"
    if itype == "file_change":
        return "file_change"
    return str(item.get("title") or item.get("tool") or item.get("command") or itype)


def _todo_status_text(item: dict[str, Any]) -> str:
    rows = item.get("items")
    if not isinstance(rows, list) or not rows:
        return "Updating plan…"
    done = 0
    total = 0
    current = ""
    for row in rows:
        if not isinstance(row, dict):
            continue
        total += 1
        if row.get("completed"):
            done += 1
        elif not current:
            current = str(row.get("text") or "").strip()
    if total == 0:
        return "Updating plan…"
    if done >= total:
        return f"Plan complete ({done}/{total})"
    if current:
        return f"Plan {done}/{total} · {current}"
    return f"Plan {done}/{total}"


class _CodexStream:
    """Parses codex exec --json events (new `thread.*/item.*` and legacy `msg` shapes)."""

    def __init__(self, conv_id: str, run_id: str, push: Callable[[dict[str, Any]], None]) -> None:
        self.conv_id = conv_id
        self.run_id = run_id
        self.push = push
        self.session_id = ""
        self.text_parts: list[str] = []
        self.error_text = ""
        self.usage: dict[str, Any] = {}
        # Ordered thinking/text/tool_call blocks in the embedded agent's persisted format,
        # so the turn's steps survive a panel reload (not just the reply).
        self.blocks: list[dict[str, Any]] = []
        self._seg_text: list[str] = []
        self._seg_thinking: list[str] = []
        # item id → {name, arguments, started_at} for tool/tool_done pairing in the panel UI.
        self._items: dict[str, dict[str, Any]] = {}

    def _emit(self, event: dict[str, Any]) -> None:
        event.setdefault("conv_id", self.conv_id)
        if self.run_id:
            event.setdefault("run_id", self.run_id)
        self.push(event)

    def _emit_status(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._emit({"type": "status", "text": text})

    def _emit_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if self.text_parts:
            self.text_parts.append("\n\n")
            self._emit({"type": "text_delta", "text": "\n\n"})
        self.text_parts.append(text)
        if self._seg_text:
            self._seg_text.append("\n\n")
        self._seg_text.append(text)
        self._emit({"type": "text_delta", "text": text})

    def _emit_thinking(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if self._seg_thinking:
            self._seg_thinking.append("\n\n")
            self._emit({"type": "thinking", "text": "\n\n"})
        self._seg_thinking.append(text)
        self._emit({"type": "thinking", "text": text})

    def _flush_segments(self) -> None:
        """Move buffered reasoning/narration into blocks (a tool call follows)."""
        thinking = "".join(self._seg_thinking).strip()
        if thinking:
            self.blocks.append({"type": "thinking", "text": thinking})
        self._seg_thinking = []
        text = "".join(self._seg_text).strip()
        if text:
            self.blocks.append({"type": "text", "text": text})
        self._seg_text = []

    def trailing_text(self) -> str:
        """Narration after the last tool call — the turn's final answer text."""
        return "".join(self._seg_text).strip()

    def finalize_blocks(self) -> list[dict[str, Any]]:
        """Persist trailing thinking; trailing text stays out as message content."""
        thinking = "".join(self._seg_thinking).strip()
        if thinking:
            self.blocks.append({"type": "thinking", "text": thinking})
            self._seg_thinking = []
        return self.blocks

    def on_line(self, line: str) -> None:
        if not line.startswith("{"):
            return
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        kind = str(data.get("type") or "")
        if kind == "thread.started":
            self.session_id = str(data.get("thread_id") or "") or self.session_id
            return
        if kind == "turn.started":
            self._emit_status("Codex is working…")
            return
        if kind in ("item.started", "item.completed", "item.updated"):
            self._on_item(kind, data.get("item") or {})
            return
        if kind == "turn.completed":
            self._on_usage(data.get("usage") or {})
            return
        if kind == "turn.failed":
            err = data.get("error") or {}
            self.error_text = str(err.get("message") or "") or "Codex turn failed"
            return
        if kind == "error":
            msg = str(data.get("message") or "") or "Codex error"
            # Transient reconnect notices are progress, not fatal turn failures.
            if "reconnect" in msg.lower():
                self._emit_status(msg)
            else:
                self.error_text = msg
            return
        # Legacy protocol shape: {"id":..,"msg":{"type":"agent_message","message":"..."}}
        msg = data.get("msg")
        if isinstance(msg, dict):
            mtype = str(msg.get("type") or "")
            if mtype == "agent_message":
                self._emit_text(str(msg.get("message") or ""))
            elif mtype in ("agent_reasoning", "agent_reasoning_delta", "reasoning"):
                self._emit_thinking(str(msg.get("text") or msg.get("message") or ""))
            elif mtype == "session_configured":
                self.session_id = str(msg.get("session_id") or "") or self.session_id
            elif mtype == "token_count":
                self._on_usage(msg.get("info") or msg.get("usage") or {})
            elif mtype == "error":
                self.error_text = str(msg.get("message") or "") or "Codex error"

    def _on_usage(self, usage: dict[str, Any]) -> None:
        if not isinstance(usage, dict) or not usage:
            return
        inp = int(usage.get("input_tokens") or 0)
        cached = int(usage.get("cached_input_tokens") or usage.get("cache_read_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
        self.usage = {
            "input_tokens": inp,
            "output_tokens": out,
            "cache_read_tokens": cached,
            "cache_write_tokens": 0,
            "context_tokens": inp + cached,
            "cost_usd": None,
            "num_turns": 0,
            "model": "",
        }

    def _on_item(self, kind: str, item: dict[str, Any]) -> None:
        if not isinstance(item, dict):
            return
        itype = str(item.get("type") or item.get("item_type") or "")
        if itype == "agent_message":
            if kind == "item.completed":
                self._emit_text(str(item.get("text") or ""))
            return
        if itype == "reasoning":
            if kind == "item.completed":
                self._emit_thinking(str(item.get("text") or ""))
            return
        if itype == "todo_list":
            self._emit_status(_todo_status_text(item))
            return
        if itype == "error" and kind == "item.completed":
            note = str(item.get("message") or "").strip()
            if note:
                self._emit_status(note)
            return
        if itype in ("command_execution", "mcp_tool_call", "file_change", "web_search"):
            name = _tool_name_from_item(itype, item)
            args = _tool_args_from_item(itype, item)
            item_id = str(item.get("id") or "") or f"{itype}-{len(self._items)}"
            if kind == "item.started":
                self._flush_segments()
                self._items[item_id] = {
                    "name": name,
                    "arguments": args,
                    "started_at": time.monotonic(),
                    "started_wall": time.time(),
                }
                self._emit(
                    {
                        "type": "tool",
                        "text": f"⚙ {name}",
                        "tool": {"name": name, "arguments": args, "status": "pending"},
                    }
                )
            elif kind == "item.completed":
                info = self._items.pop(item_id, None)
                # A completed item we never saw start still needs an intent row
                # so the UI's positional tool/tool_done pairing lines up.
                if info is None:
                    self._flush_segments()
                    self._emit(
                        {
                            "type": "tool",
                            "text": f"⚙ {name}",
                            "tool": {"name": name, "arguments": args, "status": "pending"},
                        }
                    )
                    info = {"name": name, "arguments": args, "started_at": 0.0, "started_wall": 0.0}
                started = float(info.get("started_at") or 0.0)
                ms = int((time.monotonic() - started) * 1000) if started else 0
                failed = str(item.get("status") or "").lower() in ("failed", "error")
                if itype == "mcp_tool_call":
                    result_text = _mcp_result_text(item)
                elif itype == "file_change":
                    changes = item.get("changes") if isinstance(item.get("changes"), list) else []
                    bits = []
                    for c in changes:
                        if isinstance(c, dict) and c.get("path"):
                            bits.append(f"{c.get('kind') or 'update'}: {c.get('path')}")
                    result_text = truncate_tool_result("\n".join(bits) if bits else "")
                elif itype == "web_search":
                    result_text = truncate_tool_result(str(item.get("query") or ""))
                else:
                    result_text = truncate_tool_result(
                        str(item.get("aggregated_output") or item.get("output") or "")
                    )
                status = "error" if failed else "success"
                out_name = str(info.get("name") or name)
                out_args = info.get("arguments") if isinstance(info.get("arguments"), dict) else args
                tool_payload: dict[str, Any] = {
                    "name": out_name,
                    "arguments": out_args,
                    "status": status,
                    "durationMs": ms,
                    "result": result_text,
                    "hint": "",
                }
                if not failed:
                    try:
                        from frontend.ui_web.verse_editor.agent_sync import file_edit_meta_for_stream

                        file_edit = file_edit_meta_for_stream(out_name, out_args, result_text)
                        if file_edit:
                            tool_payload["fileEdit"] = file_edit
                    except Exception:
                        pass
                block: dict[str, Any] = {
                    "type": "tool_call",
                    "id": item_id,
                    "name": out_name,
                    "arguments": out_args,
                    "started": float(info.get("started_wall") or 0.0),
                    "duration_ms": ms,
                    "result": {"ok": not failed, "data": result_text, "hint": ""},
                    "status": status,
                }
                if "fileEdit" in tool_payload:
                    block["file_edit"] = tool_payload["fileEdit"]
                self.blocks.append(block)
                self._emit(
                    {
                        "type": "tool_done",
                        "text": f"⚙ {name} · {status}" + (f" · {ms}ms" if ms else ""),
                        "success": not failed,
                        "tool": tool_payload,
                    }
                )

    def finish_unresolved_tools(self, *, cancelled: bool) -> None:
        """Close any still-open chips so nothing spins forever after the turn ends."""
        note = "Cancelled before the tool finished." if cancelled else "Turn ended before the tool reported a result."
        for item_id, info in self._items.items():
            name = str(info.get("name") or "tool")
            args = info.get("arguments") if isinstance(info.get("arguments"), dict) else {}
            self.blocks.append(
                {
                    "type": "tool_call",
                    "id": item_id,
                    "name": name,
                    "arguments": args,
                    "started": float(info.get("started_wall") or 0.0),
                    "duration_ms": 0,
                    "result": {"ok": False, "data": note, "hint": ""},
                    "status": "error",
                }
            )
            self._emit(
                {
                    "type": "tool_done",
                    "text": f"⚙ {name} · error",
                    "success": False,
                    "tool": {
                        "name": name,
                        "arguments": args,
                        "status": "error",
                        "durationMs": 0,
                        "result": note,
                        "hint": "",
                    },
                }
            )
        self._items.clear()


class CodexAdapter:
    id = "codex"
    label = "Codex"
    capabilities = CodingAgentCapabilities(
        terminal_agent=True,
        chat_api=False,
        a2a=True,
        mcp_inject=True,
        needs_api_key=False,
        needs_cli=True,
        resume=True,
    )

    def detect(self, settings: Any) -> CodingAgentInfo:
        cfg = coding_agent_cfg(settings, self.id)
        enabled = bool(cfg.get("enabled", True))
        override = str(cfg.get("cli_path") or "")
        from .cli_update import read_cli_version, resolve_bin, status_text

        path = resolve_bin(override)
        default_args = str(cfg.get("default_args") or "--full-auto")
        if path:
            ver = read_cli_version(path)
            extra = status_text()
            status = f"Found: {path}"
            if ver:
                status += f" · v{ver}"
            if extra and extra not in status:
                status += f" · {extra}"
            available = enabled
        else:
            status = _codex_missing_status()
            available = False
        return CodingAgentInfo(
            id=self.id,
            label=self.label,
            enabled=enabled,
            available=available,
            status=status,
            cli_path=path or override,
            default_args=default_args,
            capabilities=self.capabilities,
            models=_codex_model_rows(),
        )

    def launch(
        self,
        *,
        prompt: str,
        system_prompt: str,
        cwd: str,
        conv_id: str,
        model: str,
        mcp_config_path: str,
        extra_args: str,
        cli_path: str,
        env: dict[str, str],
        push: Any,
        session_id: str = "",
        run_id: str = "",
        cancel: threading.Event | None = None,
        timeout_s: float = 0.0,
        image_paths: list[str] | None = None,
    ) -> CodingAgentLaunchResult:
        model_id = normalize_codex_model(model)
        from .cli_update import resolve_bin, should_heal_launch, update_cli

        binary = resolve_bin(cli_path)
        if not binary:
            push(
                {
                    "type": "status",
                    "text": "Codex CLI missing — installing automatically…",
                    "conv_id": conv_id,
                    "run_id": run_id,
                }
            )
            upd = update_cli(cli_path)
            binary = resolve_bin(cli_path) or str(upd.get("cli_path") or "")
            if not binary:
                return CodingAgentLaunchResult(
                    ok=False,
                    error=str(upd.get("error") or "Codex CLI not found and auto-install failed"),
                    status="error",
                )
        full_prompt = prompt
        # System context only on the first turn; the resumed thread keeps it.
        if system_prompt.strip() and not session_id:
            full_prompt = system_prompt.strip() + "\n\n" + prompt

        # Windows CreateProcess dies with WinError 206 when argv is huge.
        # Keep the CLI arg short; put the real brief in a temp file.
        from backend.agent.coding_agents.mcp_inject import write_prompt_file

        prompt_file = None
        launch_prompt = full_prompt
        if len(full_prompt) > 2500:
            prompt_file = write_prompt_file(full_prompt, conv_id=conv_id)
            launch_prompt = (
                "Open this UTF-8 file and follow every instruction in it exactly "
                f"(do not summarize first): {prompt_file}"
            )
        extra_flags: list[str] = (
            list(_APPROVAL_FLAG) + list(_MCP_APPROVE_FLAG) + list(_BYPASS_APPROVALS)
        )
        if not session_id:
            extra_flags.append("--approve-for-me")
        write_codex_uefn_profile(mcp_config_path)
        extra_dirs: list[Path] = []
        if prompt_file is not None:
            extra_dirs.append(Path(prompt_file).parent)
        skills = Path.home() / ".claude" / "skills"
        if skills.is_dir():
            extra_dirs.append(skills)
        extra_flags.extend(_writable_roots_flag(extra_dirs))
        try:
            argv = build_codex_argv(
                binary=binary,
                prompt=launch_prompt,
                model=model_id,
                extra_args=extra_args,
                session_id=session_id,
                image_paths=list(image_paths or []),
                extra_flags=extra_flags,
            )
            state = _CodexStream(conv_id, run_id, push)
            if session_id:
                push({"type": "status", "text": "Resumed Codex session…", "conv_id": conv_id, "run_id": run_id})
            else:
                push({"type": "status", "text": "Starting Codex…", "conv_id": conv_id, "run_id": run_id})
            proc = run_streaming_process(
                argv=argv,
                cwd=cwd,
                env_extra=env,
                conv_id=conv_id,
                on_line=state.on_line,
                timeout_s=timeout_s,
                cancel=cancel,
            )

            # Never leave a chip spinning: whatever ended this turn, resolve leftovers.
            state.finish_unresolved_tools(cancelled=proc.cancelled)
            blocks = state.finalize_blocks()

            # Text before tool calls lives in blocks; the trailing segment is the
            # final answer (all of it when no tools ran).
            streamed_all = "".join(state.text_parts).strip()
            reply = state.trailing_text() or ("" if blocks else streamed_all)
            new_session = state.session_id or session_id

            result = finalize_cli_turn(
                proc=proc,
                reply=reply,
                streamed=bool(streamed_all) or bool(blocks),
                blocks=blocks,
                session_id=session_id,
                new_session=new_session,
                usage=state.usage,
                agent_label="Codex",
                timeout_s=timeout_s,
                error_text=state.error_text,
                stale_session_markers=("resume", "not found"),
            )
            if result.ok or not should_heal_launch(
                result.error or "",
                result.reply_text or "",
                proc.stderr_tail,
                proc.raw_tail,
                state.error_text,
            ):
                return result
            push(
                {
                    "type": "status",
                    "text": "Codex CLI is stale or missing — updating automatically…",
                    "conv_id": conv_id,
                    "run_id": run_id,
                }
            )
            upd = update_cli(binary)
            if not upd.get("ok"):
                result.error = (
                    (result.error or "")
                    + "\n\nDucky tried to update Codex automatically and failed: "
                    + str(upd.get("error") or "unknown")
                )
                return result
            binary = resolve_bin(cli_path) or str(upd.get("cli_path") or binary)
            argv[0] = binary
            state = _CodexStream(conv_id, run_id, push)
            proc = run_streaming_process(
                argv=argv,
                cwd=cwd,
                env_extra=env,
                conv_id=conv_id,
                on_line=state.on_line,
                timeout_s=timeout_s,
                cancel=cancel,
            )
            state.finish_unresolved_tools(cancelled=proc.cancelled)
            blocks = state.finalize_blocks()
            streamed_all = "".join(state.text_parts).strip()
            reply = state.trailing_text() or ("" if blocks else streamed_all)
            new_session = state.session_id or session_id
            return finalize_cli_turn(
                proc=proc,
                reply=reply,
                streamed=bool(streamed_all) or bool(blocks),
                blocks=blocks,
                session_id=session_id,
                new_session=new_session,
                usage=state.usage,
                agent_label="Codex",
                timeout_s=timeout_s,
                error_text=state.error_text,
                stale_session_markers=("resume", "not found"),
            )
        finally:
            if prompt_file is not None:
                try:
                    prompt_file.unlink(missing_ok=True)
                except OSError:
                    pass
