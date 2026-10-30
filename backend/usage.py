"""Live Codex/ChatGPT plan windows. Not API-key RPM tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PAIRS = (
    ("x-ratelimit-remaining-requests", "x-ratelimit-limit-requests", "x-ratelimit-reset-requests", "requests"),
    ("x-ratelimit-remaining-tokens", "x-ratelimit-limit-tokens", "x-ratelimit-reset-tokens", "tokens"),
)


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        n = float(str(v).strip().rstrip("s"))
    except (TypeError, ValueError):
        return None
    if n != n or n < 0:
        return None
    return n


def reset_after_text(seconds: Any) -> str:
    s = _num(seconds)
    if s is None:
        return ""
    n = int(s)
    if n <= 0:
        return ""
    d, rem = divmod(n, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    parts: list[str] = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h" if d else f"{h} hr")
    if m and d == 0:
        parts.append(f"{m} min")
    return f"Resets in {' '.join(parts)}" if parts else "Resets soon"


def _pct_row(wid: str, label: str, used_pct: float, *, reset: str = "", left: bool = False) -> dict[str, Any]:
    used = max(0.0, min(100.0, used_pct))
    left_pct = max(0.0, 100.0 - used)
    readout = f"{int(round(left_pct))}% left" if left else f"{int(round(used))}%"
    row: dict[str, Any] = {"id": wid, "label": label, "used": used, "limit": 100.0, "readout": readout}
    if reset:
        row["reset"] = reset
    return row


def windows_from_wham(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    rl = data.get("rate_limit") if isinstance(data.get("rate_limit"), dict) else {}
    if not rl:
        rl = data.get("rate_limits") if isinstance(data.get("rate_limits"), dict) else {}
    primary = rl.get("primary_window") or data.get("five_hour") or {}
    secondary = rl.get("secondary_window") or data.get("weekly") or {}
    out: list[dict[str, Any]] = []
    for wid, label, blob in (
        ("hourly", "5-hour limit", primary),
        ("weekly", "Weekly limit", secondary),
    ):
        if not isinstance(blob, dict):
            continue
        pct = blob.get("used_percent")
        if pct is None:
            pct = blob.get("used_percentage")
        n = _num(pct)
        if n is None:
            continue
        reset = reset_after_text(blob.get("reset_after_seconds"))
        if not reset:
            reset = reset_after_text(blob.get("reset_after"))
        out.append(_pct_row(wid, label, n, reset=reset, left=True))
    return out


def windows_from_headers(headers: Any) -> list[dict[str, Any]]:
    raw: dict[str, str] = {}
    items = headers.items() if headers is not None and hasattr(headers, "items") else []
    for k, v in items:
        if isinstance(v, (list, tuple)):
            v = v[0] if v else ""
        raw[str(k or "").lower()] = str(v or "").strip()
    out: list[dict[str, Any]] = []
    for rem_k, lim_k, reset_k, unit in _PAIRS:
        rem = _num(raw.get(rem_k))
        lim = _num(raw.get(lim_k))
        if rem is None or lim is None or lim <= 0:
            continue
        used = max(0.0, lim - rem)
        out.append(
            {
                "id": unit,
                "label": unit[:1].upper() + unit[1:],
                "used": used,
                "limit": lim,
                "unit": unit,
                "reset": reset_after_text(raw.get(reset_k)),
            }
        )
    return out


def _codex_auth(home: str = "") -> tuple[str, str]:
    path = Path(home or Path.home() / ".codex") / "auth.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "", ""
    tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else data
    if not isinstance(tokens, dict):
        return "", ""
    token = str(tokens.get("access_token") or data.get("access_token") or "").strip()
    account = str(tokens.get("account_id") or data.get("account_id") or "").strip()
    return token, account


def _wham(token: str, account: str) -> list[dict[str, Any]]:
    try:
        import httpx

        headers = {"Authorization": f"Bearer {token}"}
        if account:
            headers["ChatGPT-Account-Id"] = account
        r = httpx.get(
            "https://chatgpt.com/backend-api/wham/usage",
            headers=headers,
            timeout=8.0,
            follow_redirects=True,
        )
        return windows_from_wham(r.json()) if r.status_code < 400 else []
    except Exception:
        return []


def account_usage(config_dir: str, **_kw: Any) -> list[dict[str, Any]]:
    """Plan windows of one Codex login (its CODEX_HOME)."""
    token, account = _codex_auth(config_dir)
    return _wham(token, account) if token else []


def fetch_usage(api_key: str, *, model: str = "") -> dict[str, Any]:
    token, account = _codex_auth()
    if token:
        rows = _wham(token, account)
        if rows:
            return {"windows": rows}
    key = (api_key or "").strip()
    if not key:
        return {"windows": []}
    try:
        import httpx

        r = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=8.0,
            follow_redirects=True,
        )
        return {"windows": windows_from_headers(r.headers)}
    except Exception:
        return {"windows": []}
