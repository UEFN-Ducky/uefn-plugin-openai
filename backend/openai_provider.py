"""OpenAI Chat Completions provider."""

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


class OpenAIProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

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

    async def stream_turn(
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
            r = client.chat.completions.create(
                model=self._model,
                max_tokens=8,
                messages=[{"role": "user", "content": "ping"}],
            )
            _ = r.choices[0].message.content
            return True, "OpenAI OK"
        except Exception as e:
            err = str(e)
            if "v1/responses" in err or "only supported in v1/responses" in err:
                return (
                    False,
                    "Model requires OpenAI Responses API (not supported in chat yet). "
                    "Pick a chat-completions model such as gpt-4o or gpt-4o-mini.",
                )
            return False, err
