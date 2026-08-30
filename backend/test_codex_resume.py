from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for k in list(sys.modules):
    if k == "backend" or k.startswith("backend."):
        del sys.modules[k]
for p in _here.parents:
    cand = p / "UEFN-Ducky-Release" / "ducky_app"
    if (cand / "backend" / "agent").is_dir():
        sys.path.insert(0, str(_here))
        sys.path.insert(0, str(cand))
        break

from codex_adapter import CodexAdapter, build_codex_argv


def test_followup_resumes_thread():
    argv = build_codex_argv(
        binary="codex",
        prompt="follow up",
        model="gpt-5",
        extra_args="",
        session_id="th-abc",
    )
    assert argv[1:4] == ["exec", "resume", "th-abc"]
    assert argv[-1] == "follow up"


def test_first_turn_has_no_resume():
    argv = build_codex_argv(
        binary="codex",
        prompt="hello",
        model="gpt-5",
        extra_args="",
        session_id="",
    )
    assert "resume" not in argv


def test_adapter_opts_into_resume():
    assert CodexAdapter.capabilities.resume is True
