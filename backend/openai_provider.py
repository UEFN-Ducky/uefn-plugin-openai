"""OpenAI Chat Completions + Responses (Astra / gpt-6) provider."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from backend.agent.multimodal_content import build_openai_user_content
from backend.agent.prompt_cache import PromptCachePayload
from backend.agent.providers.base import (
    ProviderMessage,
    StreamEvent,
    StreamEventKind,
    ToolCallRequest,
)
from backend.agent.providers.cache_utils import openai_system_messages, parse_openai_usage
from backend.agent.providers.thinking import ThinkSplitter, reasoning_from_delta
from backend.agent.thinking_effort import normalize_thinking_effort


def uses_responses_api(model: str) -> bool:
    mid = (model or "").strip().lower()
    return "astra" in mid or mid.startswith("gpt-6")


def responses_effort(thinking_effort: str) -> str:
    """Astra rejects `none`; Off in the UI maps to the lowest supported effort."""
    v = normalize_thinking_effort(thinking_effort)
    if v in ("low", "medium", "high"):
        return v
    return "low"


def to_responses_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict):
            out.append(
                {
                    "type": "function",
                    "name": fn.get("name") or "",
                    "description": fn.get("description") or "",
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
            continue
        if isinstance(t, dict):
            out.append(t)
    return out


def _as_dict(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            data = dump(exclude_none=True)
        except TypeError:
            data = dump()
        return data if isinstance(data, dict) else None
    return None


def _responses_user_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    parts: list[dict[str, Any]] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "image_url":
            url = ((p.get("image_url") or {}) if isinstance(p.get("image_url"), dict) else {}).get("url") or ""
            parts.append({"type": "input_image", "image_url": url})
        elif p.get("type") == "text":
            parts.append({"type": "input_text", "text": p.get("text") or ""})
        else:
            parts.append(p)
    return parts


def to_responses_input(messages: list[ProviderMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "type": "function_call_output",
                    "call_id": m.tool_call_id,
                    "output": m.content or "",
                }
            )
            continue
        for block in m.thinking_blocks or []:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                out.append(block)
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                out.append(
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments or {}),
                    }
                )
            continue
        if m.role == "user" and m.attachments:
            out.append(
                {
                    "role": "user",
                    "content": _responses_user_content(build_openai_user_content(m.content, m.attachments)),
                }
            )
            continue
        if m.content:
            out.append({"role": m.role, "content": m.content})
    return out


def _event_delta(event: Any) -> str:
    d = getattr(event, "delta", None)
    if isinstance(d, str) and d:
        return d
    if d is None:
        return ""
    text = getattr(d, "text", None)
    return text if isinstance(text, str) else ""


def _responses_usage(usage: Any) -> dict[str, int]:
    parsed = parse_openai_usage(usage)
    if parsed.get("input_tokens") or parsed.get("output_tokens"):
        return parsed
    if usage is None:
        return {}
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_read_tokens": int(
            getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
        ),
        "cache_write_tokens": 0,
    }


class OpenAIProvider:
    def __init__(self, api_key: str, model: str, *, thinking_effort: str = "off", **_kw: Any) -> None:
        self._api_key = api_key
        self._model = model
        self._thinking_effort = normalize_thinking_effort(thinking_effort)

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=self._api_key)

    def _to_openai_messages(
        self,
        system: str,
        messages: list[ProviderMessage],
        *,
        cache: PromptCachePayload | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = list(openai_system_messages(cache, fallback_system=system))
        for m in messages:
            if m.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id,
                        "content": m.content,
                    }
                )
                continue
            if m.role == "assistant" and m.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": m.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in m.tool_calls
                        ],
                    }
                )
                continue
            if m.role == "user" and m.attachments:
                out.append({"role": "user", "content": build_openai_user_content(m.content, m.attachments)})
                continue
            out.append({"role": m.role, "content": m.content})
        return out

    def _instructions(self, system: str, cache: PromptCachePayload | None) -> str:
        return "\n\n".join(
            str(m.get("content") or "")
            for m in openai_system_messages(cache, fallback_system=system)
            if m.get("content")
        )

    async def stream_turn(
        self,
        *,
        system: str,
        messages: list[ProviderMessage],
        tools: list[dict[str, Any]],
        cancel_event: Any | None = None,
        cache: PromptCachePayload | None = None,
    ) -> AsyncIterator[StreamEvent]:
        if uses_responses_api(self._model):
            async for event in self._stream_responses(
                system=system,
                messages=messages,
                tools=tools,
                cancel_event=cancel_event,
                cache=cache,
            ):
                yield event
            return
        async for event in self._stream_chat(
            system=system,
            messages=messages,
            tools=tools,
            cancel_event=cancel_event,
            cache=cache,
        ):
            yield event

    async def _stream_responses(
        self,
        *,
        system: str,
        messages: list[ProviderMessage],
        tools: list[dict[str, Any]],
        cancel_event: Any | None = None,
        cache: PromptCachePayload | None = None,
    ) -> AsyncIterator[StreamEvent]:
        client = self._client()
        collected_text = ""
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        thinking_blocks: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        cancelled = False
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "input": to_responses_input(messages),
            "instructions": self._instructions(system, cache),
            "tools": to_responses_tools(tools) or None,
            "stream": True,
            "store": False,
            "reasoning": {"effort": responses_effort(self._thinking_effort), "summary": "auto"},
        }
        cache_key = (cache.prompt_cache_key if cache else "") or ""
        if cache_key:
            create_kwargs["prompt_cache_key"] = cache_key

        stream = client.responses.create(**create_kwargs)
        for event in stream:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                cancelled = True
                break
            etype = getattr(event, "type", "") or ""
            if etype == "response.output_text.delta":
                chunk = _event_delta(event)
                if chunk:
                    collected_text += chunk
                    yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=chunk)
                continue
            if etype in (
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            ):
                chunk = _event_delta(event)
                if chunk:
                    yield StreamEvent(kind=StreamEventKind.THINKING, text=chunk)
                continue
            if etype == "response.output_item.added":
                item = getattr(event, "item", None)
                if getattr(item, "type", "") == "function_call":
                    idx = int(getattr(event, "output_index", 0) or 0)
                    tool_calls_acc[idx] = {
                        "id": getattr(item, "call_id", None) or getattr(item, "id", "") or "",
                        "name": getattr(item, "name", "") or "",
                        "arguments": getattr(item, "arguments", "") or "",
                    }
                continue
            if etype == "response.function_call_arguments.delta":
                idx = int(getattr(event, "output_index", 0) or 0)
                acc = tool_calls_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                acc["arguments"] += _event_delta(event)
                continue
            if etype == "response.output_item.done":
                item = getattr(event, "item", None)
                dumped = _as_dict(item)
                if dumped and dumped.get("type") == "reasoning":
                    thinking_blocks.append(dumped)
                elif getattr(item, "type", "") == "function_call":
                    idx = int(getattr(event, "output_index", 0) or 0)
                    acc = tool_calls_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    acc["id"] = getattr(item, "call_id", None) or acc["id"] or getattr(item, "id", "") or ""
                    acc["name"] = getattr(item, "name", None) or acc["name"]
                    if getattr(item, "arguments", None):
                        acc["arguments"] = item.arguments
                continue
            if etype == "response.completed":
                resp = getattr(event, "response", None)
                usage = _responses_usage(getattr(resp, "usage", None) if resp is not None else None)

        if cancelled:
            return

        rebuilt: list[ToolCallRequest] = []
        for acc in tool_calls_acc.values():
            try:
                args = json.loads(acc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            rebuilt.append(
                ToolCallRequest(id=acc["id"] or "call", name=acc["name"], arguments=args)
            )
        if rebuilt:
            yield StreamEvent(
                kind=StreamEventKind.TOOL_CALLS,
                tool_calls=rebuilt,
                usage=usage,
                thinking_blocks=thinking_blocks,
            )
        yield StreamEvent(
            kind=StreamEventKind.DONE,
            text=collected_text,
            stop_reason="tool_calls" if rebuilt else "stop",
            usage=usage,
            thinking_blocks=thinking_blocks,
        )

    async def _stream_chat(
        self,
        *,
        system: str,
        messages: list[ProviderMessage],
        tools: list[dict[str, Any]],
        cancel_event: Any | None = None,
        cache: PromptCachePayload | None = None,
    ) -> AsyncIterator[StreamEvent]:
        client = self._client()
        collected_text = ""
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage: dict[str, int] = {}
        cancelled = False
        splitter = ThinkSplitter()

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_openai_messages(system, messages, cache=cache),
            "tools": tools if tools else None,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        cache_key = (cache.prompt_cache_key if cache else "") or ""
        if cache_key:
            create_kwargs["prompt_cache_key"] = cache_key

        stream = client.chat.completions.create(**create_kwargs)
        for chunk in stream:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                cancelled = True
                break
            if chunk.usage:
                usage = parse_openai_usage(chunk.usage)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            reasoning = reasoning_from_delta(delta)
            if reasoning:
                yield StreamEvent(kind=StreamEventKind.THINKING, text=reasoning)
            if delta.content:
                for kind, seg in splitter.feed(delta.content):
                    if kind == "think":
                        yield StreamEvent(kind=StreamEventKind.THINKING, text=seg)
                    else:
                        collected_text += seg
                        yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=seg)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    acc = tool_calls_acc.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function and tc.function.name:
                        acc["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        acc["arguments"] += tc.function.arguments

        if cancelled:
            return

        for kind, seg in splitter.flush():
            if kind == "think":
                yield StreamEvent(kind=StreamEventKind.THINKING, text=seg)
            else:
                collected_text += seg
                yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=seg)

        rebuilt: list[ToolCallRequest] = []
        for acc in tool_calls_acc.values():
            try:
                args = json.loads(acc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            rebuilt.append(
                ToolCallRequest(id=acc["id"] or "call", name=acc["name"], arguments=args)
            )
        if rebuilt:
            yield StreamEvent(kind=StreamEventKind.TOOL_CALLS, tool_calls=rebuilt, usage=usage)
        yield StreamEvent(
            kind=StreamEventKind.DONE,
            text=collected_text,
            stop_reason="tool_calls" if rebuilt else "stop",
            usage=usage,
        )

    async def test_connection(self) -> tuple[bool, str]:
        try:
            client = self._client()
            if uses_responses_api(self._model):
                r = client.responses.create(
                    model=self._model,
                    input="ping",
                    reasoning={"effort": "low"},
                )
                _ = getattr(r, "output_text", None) or True
                return True, "OpenAI OK"
            r = client.chat.completions.create(
                model=self._model,
                max_tokens=8,
                messages=[{"role": "user", "content": "ping"}],
            )
            _ = r.choices[0].message.content
            return True, "OpenAI OK"
        except Exception as e:
            return False, str(e)
