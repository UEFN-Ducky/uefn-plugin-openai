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


def _fetch_usage(api_key: str, **kw: Any) -> Any:
    from .usage import fetch_usage

    return fetch_usage(api_key, model=str(kw.get("model") or ""))


def _resolve_api_fallback(model_id: str) -> tuple[str, str] | None:
    from backend.agent.secrets import has_key

    if not has_key("openai"):
        return None
    mid = (model_id or "").strip() or "gpt-4o"
    return "openai", mid


def _skills_dir() -> str:
    # Same folder Apply writes beside ~/.codex/config.toml.
    return str(Path.home() / ".codex" / "skills")


def _normalize_codex_model(model: str) -> str:
    from .codex_adapter import normalize_codex_model

    return normalize_codex_model(model)


def _before_launch(*, conv: Any, cli_path: str, push: Any, run_id: str, agent_id: str,
                   emit_assistant: Any, **_kw: Any) -> dict[str, Any] | None:
    """A signed-out Codex says so and opens its Log in, instead of failing with a 401."""
    from .codex_auth import codex_logged_in, login_message

    if codex_logged_in(cli_path) is not False:
        return None
    return emit_assistant(conv, agent_id=agent_id, reply=login_message(push), push=push,
                          run_id=run_id, ok=True, status="needs_login")


def _on_needs_login(*, conv: Any, push: Any, run_id: str, agent_id: str, result: Any,
                    emit_assistant: Any, **_kw: Any) -> dict[str, Any]:
    from .codex_auth import login_message

    return emit_assistant(conv, agent_id=agent_id, reply=login_message(push), push=push,
                          run_id=run_id, ok=True, status="needs_login",
                          blocks=getattr(result, "blocks", None))


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
    from . import codex_auth
    from .codex_adapter import CodexAdapter
    from .openai_provider import OpenAIProvider

    from .model_fetch import clear_model_cache

    api.register_llm_provider(
        "openai",
        factory=lambda api_key, model, **kw: OpenAIProvider(api_key, model, **kw),
        fetch_models=_fetch_models,
        fetch_usage=_fetch_usage,
        test_key_model="gpt-4o-mini",
        tool_schema="openai",
        clear_model_cache=clear_model_cache,
        cache_mode="cached",
        cache_ttl_s=300,
        shows_thinking_effort=True,
        is_quota_error=codex_auth.is_quota_error,
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
        before_launch=_before_launch,
        on_needs_login=_on_needs_login,
        login=codex_auth.start_login,
        login_status=codex_auth.login_status,
        login_cancel=codex_auth.cancel_login,
        logout=codex_auth.logout,
        login_status_ok="logged in",
        **codex_auth.ACCOUNT_HOOKS,
        shows_thinking_effort=True,
        # Codex works through Ducky's tools only, so Ducky's approval modes apply to it:
        # Ducky asks before its file changes (Ask before changes) and its terminal commands.
        chat_permission_modes=("ask", "edits", "all"),
    )
    api.register_ide_hookup("codex", label="Codex")
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
    from . import graph_nodes

    graph_nodes.register_nodes(api)
    api.log("OpenAI gateway contribution active (Providers + Codex + IDE)")
