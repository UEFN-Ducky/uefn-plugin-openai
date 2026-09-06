"""Codex CLI adapter — `codex exec --json` events + thread resume.

This plugin owns ``codex exec resume <id>``. Core only stores the thread
id this adapter returns and passes it back on the next turn.
"""

from __future__ import annotations

import json
import shlex
import threading
import time
from typing import Any, Callable

from backend.agent.coding_agents.base import (
    CodingAgentCapabilities,
    CodingAgentInfo,
    CodingAgentLaunchResult,
    which_cli,
)
from backend.agent.coding_agents.cli_shared import finalize_cli_turn, truncate_tool_result
from backend.agent.coding_agents.proc_exec import run_streaming_process
from backend.agent.coding_agents.settings_helpers import coding_agent_cfg

_CODEX_INSTALL_PS = r"irm https://chatgpt.com/codex/install.ps1 | iex"
_CODEX_INSTALL_NPM = "npm install -g @openai/codex"


def _codex_model_rows() -> list[dict[str, str]]:
    """Live OpenAI /v1/models — empty if the key is missing or the catalog call fails."""
    try:
        from backend.agent.secrets import get_key

        key = (get_key("openai") or "").strip()
    except Exception:
        return []
    if not key:
        return []
    try:
        from .model_fetch import fetch_models

        return [
            {"id": info.id, "name": info.display_name or info.id, "provider": "Codex"}
            for info in fetch_models(key)
            if info.id
        ]
    except Exception:
        return []


def _codex_missing_status() -> str:
    return (
        "Codex CLI not found — Ducky will install it automatically "
        "(not the ChatGPT desktop app). If this stays, send another message or "
        f"Settings → Store → Update OpenAI. Manual fallback: {_CODEX_INSTALL_PS} or {_CODEX_INSTALL_NPM}"
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


def build_codex_argv(
    *,
    binary: str,
    prompt: str,
    model: str,
    extra_args: str,
    session_id: str,
    image_paths: list[str] | None = None,
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
    if session_id:
        # `codex exec resume` has no -s/--sandbox flag; use the -c config form.
        rewritten: list[str] = []
        i = 0
        while i < len(extra):
            if extra[i] in ("-s", "--sandbox") and i + 1 < len(extra):
                rewritten.extend(["-c", f'sandbox_mode="{extra[i + 1]}"'])
                i += 2
                continue
            rewritten.append(extra[i])
            i += 1
        extra = rewritten
    argv.extend(extra)
    if model and model not in ("", "default"):
        argv.extend(["-m", model])
    else:
        raise ValueError("Codex requires an exact model id")
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
        path = which_cli("codex", override)
        default_args = str(cfg.get("default_args") or "--full-auto")
        if path:
            from .cli_update import read_cli_version, status_text

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

    def _ensure_project_mcp(self, cwd: str, mcp_config_path: str) -> None:
        """Codex often reads project .mcp.json; merge uefn server when possible."""
        if not cwd or not mcp_config_path:
            return
        from pathlib import Path

        try:
            src = json.loads(Path(mcp_config_path).read_text(encoding="utf-8"))
            servers = src.get("mcpServers") if isinstance(src, dict) else None
            if not isinstance(servers, dict) or "uefn" not in servers:
                return
            dest = Path(cwd) / ".mcp.json"
            existing: dict[str, Any] = {}
            if dest.is_file():
                try:
                    existing = json.loads(dest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
            if not isinstance(existing, dict):
                existing = {}
            mcp = existing.get("mcpServers")
            if not isinstance(mcp, dict):
                mcp = {}
            mcp["uefn"] = servers["uefn"]
            existing["mcpServers"] = mcp
            dest.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except OSError:
            pass

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
        model_id = (model or "").strip()
        if not model_id or model_id.lower() == "default":
            return CodingAgentLaunchResult(
                ok=False,
                error=(
                    "No exact Codex model selected. Pick a concrete model "
                    "for this chat or Ducky profile."
                ),
                status="error",
            )
        self._ensure_project_mcp(cwd, mcp_config_path)
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
        try:
            argv = build_codex_argv(
                binary=binary,
                prompt=launch_prompt,
                model=model,
                extra_args=extra_args,
                session_id=session_id,
                image_paths=list(image_paths or []),
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
