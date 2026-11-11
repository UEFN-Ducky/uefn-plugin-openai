"""Codex sign-in: what the login tab prints, and what `codex login status` says."""

from __future__ import annotations

import subprocess
import sys
import time
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


class _FakeProc:
    def __init__(self, auth=None, code=0, out=""):
        self._auth, self._code, self.returncode = auth, code, None
        self.stdout = iter([out]) if out else None
        self.started = time.monotonic()

    def poll(self):
        if self.returncode is None and time.monotonic() - self.started > 0.2:
            if self._auth is not None:
                self._auth.write_text("{}", encoding="utf-8")
            self.returncode = self._code
        return self.returncode

    def kill(self):
        self.returncode = -9


def _fake_pkg(binary):
    return mock.patch.dict(sys.modules, {"fakecodexpkg": mock.Mock(), "fakecodexpkg.cli_update": mock.Mock(
        resolve_bin=lambda _p: binary, hidden_run_kwargs=lambda: {}, update_cli=lambda _o: {"ok": True})})


def _wait_done(home):
    for _ in range(100):
        if not codex_auth.account_status(home).get("pending"):
            return
        time.sleep(0.05)


def test_one_click_login_runs_hidden_codex_login_and_reports_signed_in(monkeypatch, tmp_path):
    home = tmp_path / "acct"
    seen = []
    monkeypatch.setattr(codex_auth, "__package__", "fakecodexpkg")
    monkeypatch.setattr(codex_auth, "codex_logged_in", lambda _c="", home="": (Path(home) / "auth.json").is_file())
    with _fake_pkg("codex"):
        out = codex_auth.account_browser_login(
            str(home), poll_s=0.02,
            popen=lambda argv, **kw: seen.append((argv, kw["env"].get("CODEX_HOME"))) or _FakeProc(home / "auth.json"))
        assert out == {"message": codex_auth.BROWSER_WAITING}
        assert codex_auth.account_status(str(home)) == {"loggedIn": False, "pending": True}
        _wait_done(str(home))
        assert codex_auth.account_status(str(home)) == {"loggedIn": True}
    assert seen == [(["codex", "login"], str(home))]  # browser login, not --device-auth


def test_one_click_login_cancelled_says_why(monkeypatch, tmp_path):
    home = tmp_path / "acct2"
    monkeypatch.setattr(codex_auth, "__package__", "fakecodexpkg")
    monkeypatch.setattr(codex_auth, "codex_logged_in", lambda *_a, **_k: False)
    with _fake_pkg("codex"):
        codex_auth.account_browser_login(str(home), poll_s=0.02,
                                         popen=lambda *_a, **_k: _FakeProc(None, 1, "Error: login cancelled\n"))
        _wait_done(str(home))
        st = codex_auth.account_status(str(home))
    assert st["loggedIn"] is False and st["error"].startswith(codex_auth.BROWSER_NOT_FINISHED)
    assert "login cancelled" in st["error"]


def test_one_click_login_installs_missing_codex_first(monkeypatch):
    monkeypatch.setattr(codex_auth, "__package__", "fakecodexpkg")
    monkeypatch.setitem(codex_auth._INSTALL, "busy", False)
    started = []
    monkeypatch.setattr(codex_auth.threading, "Thread", lambda target, **_k: mock.Mock(start=lambda: started.append(target)))
    with _fake_pkg(""):
        assert codex_auth.account_browser_login("X") == {"error": codex_auth.INSTALLING_CODEX}
        assert codex_auth.account_status("X") == {"loggedIn": False, "pending": True, "installing": True}
        started[0]()  # the install finishes
    assert codex_auth._INSTALL == {"busy": False, "error": ""}


def test_registration_carries_account_hooks():
    # register() passes **ACCOUNT_HOOKS to register_coding_agent("codex", ...).
    agent = codex_auth.ACCOUNT_HOOKS
    assert agent["account_env"]("X") == {"CODEX_HOME": "X"}
    for hook in ("account_login", "account_status", "account_logout", "account_usage", "is_limit_error"):
        assert callable(agent[hook])
    assert "config.toml" in agent["account_shared"]["files"]
    assert "auth.json" not in agent["account_shared"]["files"]
