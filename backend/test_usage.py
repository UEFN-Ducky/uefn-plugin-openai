from __future__ import annotations

import json

import usage
from usage import windows_from_headers, windows_from_wham


def test_account_usage_reads_that_accounts_login(tmp_path, monkeypatch) -> None:
    (tmp_path / "auth.json").write_text(json.dumps({"tokens": {"access_token": "work-tok", "account_id": "a1"}}),
                                        encoding="utf-8")
    monkeypatch.setattr(usage, "_wham", lambda token, account: [{"id": "hourly", "token": token, "account": account}])
    assert usage.account_usage(str(tmp_path)) == [{"id": "hourly", "token": "work-tok", "account": "a1"}]
    assert usage.account_usage(str(tmp_path / "signed-out")) == []


def test_wham_five_hour_and_weekly() -> None:
    rows = windows_from_wham(
        {
            "rate_limit": {
                "primary_window": {
                    "used_percent": 99,
                    "reset_after_seconds": 356400,
                },
                "secondary_window": {
                    "used_percent": 99,
                    "reset_after_seconds": 356400,
                },
            }
        }
    )
    assert [w["id"] for w in rows] == ["hourly", "weekly"]
    assert rows[0]["label"] == "5-hour limit"
    assert rows[1]["label"] == "Weekly limit"
    assert rows[0]["readout"] == "1% left"
    assert "4d" in rows[1]["reset"]


def test_hourly_requests_from_openai_headers() -> None:
    rows = windows_from_headers(
        {
            "x-ratelimit-limit-requests": "40",
            "x-ratelimit-remaining-requests": "28",
            "x-ratelimit-reset-requests": "3600",
        }
    )
    assert len(rows) == 1
    assert rows[0]["used"] == 12
    assert rows[0]["limit"] == 40


def test_empty_without_limits() -> None:
    assert windows_from_headers({}) == []
    assert windows_from_wham({}) == []
    assert windows_from_wham(None) == []
