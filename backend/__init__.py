"""OpenAI gateway — LLM provider + Codex coding agent via host registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_INSTALL_HELP = (
    "Needs the OpenAI Codex CLI (`codex` in PowerShell) — not the ChatGPT desktop app. "
    "Install in Windows PowerShell: irm https://chatgpt.com/codex/install.ps1 | iex "
    "(or npm install -g @openai/codex). Ducky runs `codex exec` (non-interactive) so replies return to chat. "
    "Ensure %APPDATA%\\npm is on PATH, run codex --version, restart Ducky, and click Detect."
)


def _fetch_models(api_key: str, **kw: Any) -> Any:
    from .model_fetch import fetch_models

    return fetch_models(api_key, verify=bool(kw.get("verify", False)))


def _resolve_api_fallback(model_id: str) -> tuple[str, str] | None:
    from backend.agent.secrets import has_key

    if not has_key("openai"):
        return None
    mid = (model_id or "").strip() or "gpt-4o"
    return "openai", mid


def _skills_dir() -> str:
    # Codex reads SKILL.md from the Claude skills tree when present.
    return str(Path.home() / ".claude" / "skills")


def register(api) -> None:
    from .codex_adapter import CodexAdapter
    from .openai_provider import OpenAIProvider

    from .model_fetch import clear_model_cache

    api.register_llm_provider(
        "openai",
        factory=lambda api_key, model, **kw: OpenAIProvider(api_key, model, **kw),
        fetch_models=_fetch_models,
        test_key_model="gpt-4o-mini",
        tool_schema="openai",
        clear_model_cache=clear_model_cache,
        cache_mode="cached",
    )
    api.register_coding_agent(
        "codex",
        factory=lambda: CodexAdapter(),
        resolve_api_fallback=_resolve_api_fallback,
        aliases=["openai_codex"],
        skills_dir=_skills_dir,
        settings_defaults={
            "enabled": True,
            "cli_path": "",
            "default_args": "--full-auto",
        },
        install_help=_INSTALL_HELP,
        token_provider="openai",
    )
    api.log("OpenAI gateway contribution active (Providers + Codex)")
