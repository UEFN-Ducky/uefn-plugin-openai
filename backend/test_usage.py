from __future__ import annotations

from usage import windows_from_headers, windows_from_wham


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
