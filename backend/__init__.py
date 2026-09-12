"""OpenAI gateway — LLM provider + Codex coding agent via host registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_INSTALL_HELP = (
    "Ducky installs and updates the Codex CLI when this plugin is installed or "
    "updated — you should not run the Codex installer yourself. Needs the `codex` "
    "CLI (not the ChatGPT desktop app)."
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


def _normalize_codex_model(model: str) -> str:
    from .codex_adapter import normalize_codex_model

    return normalize_codex_model(model)


def _heal_default_model_if_codex_only() -> None:
    """ChatGPT-login installs have no API catalog — pin Default Model to Codex Auto."""
    try:
        from backend.agent.secrets import has_key
        from frontend.settings import PanelSettings

        settings = PanelSettings.load()
        if (getattr(settings, "default_model", "") or "").strip():
            return
        if has_key("openai"):
            return
        from .cli_update import resolve_bin

        if not resolve_bin(""):
            return
        settings.default_model = "codex:auto"
        settings.save()
    except Exception:
        pass


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
        shows_thinking_effort=True,
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
        normalize_model=_normalize_codex_model,
        shows_thinking_effort=True,
    )
    try:
        from .codex_adapter import heal_codex_approval_policy

        heal_codex_approval_policy()
    except Exception:
        pass
    try:
        from .cli_update import schedule_cli_update_on_plugin_load

        schedule_cli_update_on_plugin_load()
    except Exception:
        pass
    try:
        _heal_default_model_if_codex_only()
    except Exception:
        pass
    api.log("OpenAI gateway contribution active (Providers + Codex)")
