"""Codex sign-in: what the login tab prints, and what `codex login status` says."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest import mock

import codex_auth
from codex_auth import codex_logged_in, parse_login_output

# `codex login --device-auth` (0.163): bold/colour codes around the link and the code.
_DEVICE = (
    "Follow these steps to sign in with ChatGPT using device code authorization:\r\n\r\n"
    "1. Open this link in your browser and sign in to your account\r\n"
    "   \x1b[94mhttps://auth.openai.com/codex/device\x1b[0m\r\n\r\n"
    "2. Enter this one-time code \x1b[90m(expires in 15 minutes)\x1b[0m\r\n"
    "   \x1b[94mABCD-EFG12\x1b[0m\r\n"
)
_BROWSER = (
    "Starting local login server on http://localhost:1455.\r\n"
    "If your browser did not open, navigate to this URL to authenticate:\r\n\r\n"
    "https://auth.openai.com/oauth/authorize?response_type=code&client_id=app_X&state=abc\r\n"
)


def test_device_login_gives_the_link_and_the_code():
    assert parse_login_output(_DEVICE) == {
        "auth_url": "https://auth.openai.com/codex/device",
        "user_code": "ABCD-EFG12",
    }


def test_browser_login_gives_the_link_only():
    got = parse_login_output(_BROWSER)
    assert got["auth_url"].startswith("https://auth.openai.com/oauth/authorize?")
    assert got["auth_url"].endswith("state=abc")
    assert got["user_code"] == ""
    assert parse_login_output("Requesting a one-time code...") == {"auth_url": "", "user_code": ""}


def test_login_status_reads_signed_in_signed_out_and_unknown(monkeypatch):
    def status(out: str, code: int):
        monkeypatch.setattr(codex_auth, "_codex",
                            lambda *_a, **_k: subprocess.CompletedProcess([], code, stdout=out, stderr=""))

    status("Logged in using ChatGPT\n", 0)
    assert codex_logged_in() is True
    status("Not logged in\n", 1)
    assert codex_logged_in() is False
    # A crash or garbled answer is not a logout: the turn runs and a real 401 opens Log in.
    status("", 1)
    assert codex_logged_in() is None
    monkeypatch.setattr(codex_auth, "_codex", lambda *_a, **_k: None)
    assert codex_logged_in() is None


def test_extra_account_runs_codex_in_its_own_codex_home(monkeypatch, tmp_path):
    home = str(tmp_path / "accounts" / "codex" / "team")
    assert codex_auth.account_env(home) == {"CODEX_HOME": home}
    assert codex_auth.account_env("") == {}
    seen = []
    monkeypatch.setattr(codex_auth.subprocess, "run", lambda argv, **kw: (
        seen.append(kw["env"]), subprocess.CompletedProcess(argv, 0, stdout="Logged in using ChatGPT", stderr=""))[1])
    fake_bin = mock.Mock(resolve_bin=lambda _p: "codex")
    with mock.patch.dict(sys.modules, {"fakecodexpkg": mock.Mock(), "fakecodexpkg.cli_update": fake_bin}):
        monkeypatch.setattr(codex_auth, "__package__", "fakecodexpkg")
        assert codex_auth.account_status(home) == {"loggedIn": True}
        assert codex_auth.codex_logged_in() is True
        login = codex_auth.account_login(home)
    assert seen[0]["CODEX_HOME"] == home
    assert seen[1] is None  # the default account keeps Codex's own folder
    assert login == {"argv": ["codex", "login", "--device-auth"], "env": {"CODEX_HOME": home}}


def test_limit_and_quota_errors():
    assert codex_auth.is_limit_error("You've hit your usage limit. Try again in 3 hours.")
    assert codex_auth.is_limit_error('{"type":"usage_limit_reached"}')
    assert not codex_auth.is_limit_error("file not found")
    assert codex_auth.is_quota_error("Error code: 429 - insufficient_quota")
    assert not codex_auth.is_quota_error("invalid api key")


def test_registration_carries_account_hooks():
    backend = Path(__file__).parent
    spec = importlib.util.spec_from_file_location("ducky_openai_reg", backend / "__init__.py",
                                                  submodule_search_locations=[str(backend)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pkg
    try:
        spec.loader.exec_module(pkg)
        api = mock.MagicMock()
        with mock.patch.object(pkg, "_heal_default_model_if_codex_only"), \
             mock.patch(f"{spec.name}.codex_adapter.heal_codex_approval_policy"), \
             mock.patch(f"{spec.name}.cli_update.schedule_cli_update_on_plugin_load"):
            pkg.register(api)
    finally:
        for name in [m for m in sys.modules if m == spec.name or m.startswith(spec.name + ".")]:
            del sys.modules[name]
    agent = api.register_coding_agent.call_args.kwargs
    assert agent["account_env"]("X") == {"CODEX_HOME": "X"}
    for hook in ("account_login", "account_status", "account_logout", "account_usage", "is_limit_error"):
        assert callable(agent[hook])
    assert "config.toml" in agent["account_shared"]["files"]
    assert "auth.json" not in agent["account_shared"]["files"]
    assert callable(api.register_llm_provider.call_args.kwargs["is_quota_error"])
