"""Vendor model list / pricing fetch for this gateway plugin."""

from __future__ import annotations

import hashlib
import html
import logging
import re
import time
from typing import Any

from backend.agent.model_fetch import (
    ModelInfo,
    _PricingRow,
    _cache_put,
    _float_from_record,
    _int_from_record,
    _merge_prices,
    _parse_price_cell,
    _per_million_from_token_rate,
)

_log = logging.getLogger(__name__)
_CACHE_MAX = 512
_CACHE_TTL_S = 6 * 3600.0


_OPENAI_PRICING_URL = "https://developers.openai.com/api/docs/pricing"
_OPENAI_DOCS_MODEL_URL = "https://developers.openai.com/api/docs/models/{id}.md"
_OPENAI_DASHBOARD_CACHE: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}
_OPENAI_LIST_CACHE: dict[str, tuple[float, list[ModelInfo]]] = {}
_PROVIDER_PRICING_CACHE: dict[str, tuple[float, dict[str, _PricingRow]]] = {}
_DOCS_SPEC_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_LIST_TTL_S = 120.0

# Dashboard `max_tokens` is max output (Astra: 128k), not the context window (1.05M).
_CONTEXT_KEYS = (
    "context_window",
    "max_context_tokens",
    "max_input_tokens",
    "context_length",
    "n_ctx",
)
_DOCS_CTX_RE = re.compile(r"([\d,]+)\s+context window", re.I)
_DOCS_PRICE_RE = re.compile(
    r"\|\s*(Input|Cached input|Cache writes|Output)\s*\|\s*\$([\d.]+)",
    re.I,
)


def _key_hash(api_key: str) -> str:
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()


def _pricing_catalog() -> dict[str, _PricingRow]:
    hit = _PROVIDER_PRICING_CACHE.get("openai")
    if hit is not None and (time.time() - hit[0]) < _CACHE_TTL_S:
        return hit[1]
    catalog = _fetch_openai_pricing_catalog()
    _cache_put(_PROVIDER_PRICING_CACHE, "openai", (time.time(), catalog))
    return catalog


def clear_model_cache() -> None:
    _OPENAI_DASHBOARD_CACHE.clear()
    _OPENAI_LIST_CACHE.clear()
    _PROVIDER_PRICING_CACHE.clear()
    _DOCS_SPEC_CACHE.clear()


def fetch_models(api_key: str, *, verify: bool = False) -> list[ModelInfo]:
    return _fetch_openai(api_key, verify=verify)

def _openai_model_price_keys(model_id: str, alias: str | None = None) -> list[str]:
    keys: list[str] = []
    for raw in (model_id, alias or ""):
        mid = (raw or "").strip()
        if not mid or mid in keys:
            continue
        keys.append(mid)
        base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", mid)
        if base and base not in keys:
            keys.append(base)
    return keys


def _fetch_openai_pricing_catalog() -> dict[str, _PricingRow]:
    catalog: dict[str, _PricingRow] = {}
    try:
        import httpx

        r = httpx.get(_OPENAI_PRICING_URL, follow_redirects=True, timeout=30.0)
        r.raise_for_status()
        text = html.unescape(r.text)
        row_re = re.compile(
            r'\[\[0,"([^"]+)"\],'
            r"\[(?:0,([^,\]]+)|\"([^\"]*)\")\],"
            r"\[(?:0,([^,\]]+)|\"([^\"]*)\")\],"
            r"\[(?:0,([^,\]]+)|\"([^\"]*)\")\]\]"
        )
        for m in row_re.finditer(text):
            name = m.group(1).split(" (")[0].strip()
            if name in catalog:
                continue
            price_in = _parse_price_cell(m.group(2) if m.group(2) is not None else m.group(3))
            cached = _parse_price_cell(m.group(4) if m.group(4) is not None else m.group(5))
            price_out = _parse_price_cell(m.group(6) if m.group(6) is not None else m.group(7))
            if price_in is None or price_out is None:
                continue
            catalog[name] = (price_in, price_out, cached, None)
    except Exception as exc:
        _log.warning("OpenAI pricing page unavailable: %s", exc)
    return catalog


def _feature_list(record: dict[str, Any]) -> list[str]:
    raw = record.get("features")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        return [str(k) for k, v in raw.items() if v]
    return []


def _openai_prices_from_record(record: dict[str, Any]) -> tuple[float | None, float | None, float | None, float | None]:
    pricing = record.get("pricing")
    if isinstance(pricing, dict):
        price_in = _float_from_record(
            pricing,
            "input",
            "input_per_million",
            "standard_input",
            "prompt",
            "text_input",
            "input_price_per_million",
            "input_cost_per_million",
        )
        price_out = _float_from_record(
            pricing,
            "output",
            "output_per_million",
            "standard_output",
            "completion",
            "text_output",
            "output_price_per_million",
            "output_cost_per_million",
        )
        cached = _float_from_record(
            pricing,
            "cached_input",
            "cache_read",
            "cached",
            "cached_input_price_per_million",
        )
        cache_write = _float_from_record(pricing, "cache_write", "cache_creation")
        if price_in is not None:
            price_in = _per_million_from_token_rate(price_in)
        if price_out is not None:
            price_out = _per_million_from_token_rate(price_out)
        if cached is not None:
            cached = _per_million_from_token_rate(cached)
        if cache_write is not None:
            cache_write = _per_million_from_token_rate(cache_write)
        if any(v is not None for v in (price_in, price_out, cached, cache_write)):
            return price_in, price_out, cached, cache_write
    price_in = _float_from_record(
        record,
        "input_price_per_million",
        "input_cost_per_million",
        "price_per_million_input",
        "standard_input_price",
        "input_cost_per_token",
        "standard_input_cost_per_token",
    )
    price_out = _float_from_record(
        record,
        "output_price_per_million",
        "output_cost_per_million",
        "price_per_million_output",
        "standard_output_price",
        "output_cost_per_token",
        "standard_output_cost_per_token",
    )
    cached = _float_from_record(
        record,
        "cached_input_price_per_million",
        "cached_price_per_million",
        "cached_input_cost_per_token",
    )
    cache_write = _float_from_record(record, "cache_write_price_per_million", "cache_write_cost_per_token")
    if price_in is not None:
        price_in = _per_million_from_token_rate(price_in)
    if price_out is not None:
        price_out = _per_million_from_token_rate(price_out)
    if cached is not None:
        cached = _per_million_from_token_rate(cached)
    if cache_write is not None:
        cache_write = _per_million_from_token_rate(cache_write)
    return price_in, price_out, cached, cache_write


def _lookup_openai_price(
    catalog: dict[str, _PricingRow],
    model_id: str,
    alias: str | None,
) -> _PricingRow | None:
    for key in _openai_model_price_keys(model_id, alias):
        row = catalog.get(key)
        if row:
            return row
    return None


def _context_from_record(record: dict[str, Any]) -> int | None:
    ctx = _int_from_record(record, *_CONTEXT_KEYS)
    if ctx:
        return ctx
    for key in ("limits", "capabilities", "specs", "config"):
        nested = record.get(key)
        if isinstance(nested, dict):
            ctx = _int_from_record(nested, *_CONTEXT_KEYS)
            if ctx:
                return ctx
    return None


def parse_openai_docs_md(text: str) -> dict[str, Any]:
    """Context + prices from an official developers.openai.com model .md page."""
    out: dict[str, Any] = {}
    m = _DOCS_CTX_RE.search(text or "")
    if m:
        out["context_limit"] = int(m.group(1).replace(",", ""))
    prices: dict[str, float] = {}
    for kind, raw in _DOCS_PRICE_RE.findall(text or ""):
        prices[kind.strip().lower()] = float(raw)
    if "input" in prices:
        out["price_in"] = prices["input"]
    if "output" in prices:
        out["price_out"] = prices["output"]
    if "cached input" in prices:
        out["price_cached_in"] = prices["cached input"]
    if "cache writes" in prices:
        out["price_cache_write"] = prices["cache writes"]
    return out


def _docs_spec(model_id: str) -> dict[str, Any]:
    mid = (model_id or "").strip()
    if not mid:
        return {}
    hit = _DOCS_SPEC_CACHE.get(mid)
    if hit is not None and (time.time() - hit[0]) < _CACHE_TTL_S:
        return dict(hit[1])
    spec: dict[str, Any] = {}
    try:
        import httpx

        r = httpx.get(_OPENAI_DOCS_MODEL_URL.format(id=mid), follow_redirects=True, timeout=20.0)
        if r.status_code == 200 and r.text:
            spec = parse_openai_docs_md(r.text)
    except Exception as exc:
        _log.warning("OpenAI model docs unavailable for %s: %s", mid, exc)
    _cache_put(_DOCS_SPEC_CACHE, mid, (time.time(), spec))
    return dict(spec)


def _looks_like_chat_model(model_id: str) -> bool:
    m = (model_id or "").strip().lower()
    return m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt"))


def _openai_info_from_dashboard(
    record: dict[str, Any],
    model_id: str,
    pricing_catalog: dict[str, _PricingRow] | None = None,
    docs: dict[str, Any] | None = None,
) -> ModelInfo:
    features = _feature_list(record)
    alias = str(record.get("alias") or "") or None
    price_in, price_out, cached, cache_write = _merge_prices(
        _openai_prices_from_record(record),
        _lookup_openai_price(pricing_catalog or {}, model_id, alias),
    )
    docs = docs or {}
    if price_in is None:
        price_in = docs.get("price_in")
    if price_out is None:
        price_out = docs.get("price_out")
    if cached is None:
        cached = docs.get("price_cached_in")
    if cache_write is None:
        cache_write = docs.get("price_cache_write")
    return ModelInfo(
        id=model_id,
        display_name=str(alias or record.get("display_name") or model_id),
        supports_vision="image_content" in features,
        supports_tools="function_calling" in features,
        supports_web_search="web_search" in features,
        context_limit=_context_from_record(record) or docs.get("context_limit"),
        price_in=price_in,
        price_out=price_out,
        price_cached_in=cached,
        price_cache_write=cache_write,
    )


def _normalize_openai_dashboard(data: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if isinstance(data, dict):
        if data and all(isinstance(v, dict) for v in data.values()):
            for key, val in data.items():
                if not isinstance(val, dict):
                    continue
                mid = str(val.get("id") or key)
                out[mid] = val
                alias = val.get("alias")
                if alias:
                    out[str(alias)] = val
        elif isinstance(data.get("data"), list):
            for item in data["data"]:
                if isinstance(item, dict):
                    mid = str(item.get("id") or "")
                    if mid:
                        out[mid] = item
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                mid = str(item.get("id") or "")
                if mid:
                    out[mid] = item
    return out


def _fetch_openai_dashboard(api_key: str) -> dict[str, dict[str, Any]]:
    cache_key = _key_hash(api_key)
    hit = _OPENAI_DASHBOARD_CACHE.get(cache_key)
    if hit is not None and (time.time() - hit[0]) < _CACHE_TTL_S:
        return hit[1]
    catalog: dict[str, dict[str, Any]] = {}
    try:
        import httpx

        r = httpx.get(
            "https://api.openai.com/dashboard/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        r.raise_for_status()
        catalog = _normalize_openai_dashboard(r.json())
    except Exception as exc:
        _log.warning("OpenAI dashboard models unavailable: %s", exc)
    _cache_put(_OPENAI_DASHBOARD_CACHE, cache_key, (time.time(), catalog))
    return catalog


def _openai_chat_completions_ok(client, model_id: str) -> bool:
    """True if model accepts chat.completions (not Responses-API-only)."""
    try:
        client.chat.completions.create(
            model=model_id,
            max_tokens=1,
            messages=[{"role": "user", "content": "x"}],
        )
        return True
    except Exception as e:
        err = str(e).lower()
        if "v1/responses" in err and "v1/chat/completions" in err:
            return False
        if "only supported in v1/responses" in err:
            return False
        if "429" in err or "rate" in err or "quota" in err:
            return True
        if "404" in err or "does not exist" in err or "not found" in err:
            return False
        return True


def _fetch_openai(api_key: str, *, verify: bool = False) -> list[ModelInfo]:
    from openai import OpenAI

    cache_key = _key_hash(api_key or "")
    if not verify:
        hit = _OPENAI_LIST_CACHE.get(cache_key)
        if hit is not None and (time.time() - hit[0]) < _LIST_TTL_S:
            return list(hit[1])

    client = OpenAI(api_key=api_key)
    listed_ids: list[str] = []
    for m in client.models.list():
        mid = (m.id or "").strip()
        if mid:
            listed_ids.append(mid)

    dashboard = _fetch_openai_dashboard(api_key)
    pricing_catalog = _pricing_catalog()
    models: list[ModelInfo] = []
    seen: set[str] = set()
    for mid in listed_ids:
        if mid in seen:
            continue
        rec = dashboard.get(mid)
        docs = (
            _docs_spec(mid)
            if rec is not None and _looks_like_chat_model(mid) and _context_from_record(rec) is None
            else {}
        )
        if rec:
            info = _openai_info_from_dashboard(rec, mid, pricing_catalog, docs)
        else:
            info = ModelInfo(
                id=mid,
                display_name=mid,
                context_limit=docs.get("context_limit"),
                price_in=docs.get("price_in"),
                price_out=docs.get("price_out"),
                price_cached_in=docs.get("price_cached_in"),
                price_cache_write=docs.get("price_cache_write"),
            )
        models.append(info)
        seen.add(mid)

    models.sort(key=lambda m: m.id, reverse=True)
    if verify and models:
        models = [info for info in models if _openai_chat_completions_ok(client, info.id)]
    _cache_put(_OPENAI_LIST_CACHE, cache_key, (time.time(), models))
    return models

