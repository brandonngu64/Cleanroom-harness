"""Translate the model-independent agent interface to Anthropic's Messages API."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import math
from typing import Any

from .agent import Message, ModelError, ModelReply, ToolCall

_STOP_REASONS = {
    "end_turn": "completed",
    "tool_use": "tool_calls",
    "max_tokens": "output_limit",
    "model_context_window_exceeded": "output_limit",
    "refusal": "refused",
}


class ClaudeClient:
    """Use an explicitly selected model; credentials may come from the SDK's env.

    The optional SDK is imported only when this adapter is constructed. Injected
    SDK clients remain owned by their caller. Automatic retries are disabled on
    every request so the agent's request count remains meaningful. SDK timeouts
    are transport timeouts, not a hard wall-clock cancellation guarantee.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: Any = None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty model ID")
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                'Install Claude support with: pip install -e ".[claude]"'
            ) from exc

        self.model = model
        self._api_error = anthropic.APIError
        self._owns_client = client is None
        if client is None:
            options = {"max_retries": 0}
            if api_key is not None:
                options["api_key"] = api_key
            client = anthropic.Anthropic(**options)
        elif api_key is not None:
            raise ValueError("Pass either an SDK client or an API key, not both")
        self._client = client

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
        *,
        max_output_tokens: int,
        timeout: float,
    ) -> ModelReply:
        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive, finite number")

        system, conversation = self._translate_messages(messages)
        api_tools = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": deepcopy(tool["parameters"]),
            }
            for tool in tools
        ]
        request = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "messages": conversation,
            "tools": api_tools,
        }
        if system:
            request["system"] = system
        try:
            response = self._client.with_options(
                max_retries=0, timeout=timeout
            ).messages.create(**request)
        except self._api_error as exc:
            raise ModelError(str(exc)) from exc

        return self._normalize_reply(response)

    def _translate_messages(
        self, messages: Sequence[Message]
    ) -> tuple[str, list[dict[str, object]]]:
        system = []
        conversation = []
        pending = set()
        for message in messages:
            if message.role == "system":
                if conversation:
                    raise ValueError("System messages must precede the conversation")
                system.append(message.text)
                continue

            if message.role == "tool":
                if message.tool_call_id not in pending:
                    raise ValueError("Tool results must match pending tool call IDs")
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.text,
                    "is_error": message.is_error,
                }
                # Claude expects the entire batch in one immediately following
                # user message, rather than separate messages with role='tool'.
                if conversation[-1]["role"] == "user":
                    conversation[-1]["content"].append(block)
                else:
                    conversation.append({"role": "user", "content": [block]})
                pending.remove(message.tool_call_id)
                continue

            if pending:
                raise ValueError("Every tool call needs a result before continuing")
            if message.role == "user":
                conversation.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": message.text}],
                    }
                )
                continue
            if message.role != "assistant":
                raise ValueError(f"Unsupported message role: {message.role}")

            ids = [call.id for call in message.tool_calls]
            if len(ids) != len(set(ids)):
                raise ValueError("Assistant tool call IDs must be unique")
            if message.provider_data is not None:
                if message.provider_data.get("provider") != "anthropic":
                    raise ValueError("Assistant provider data is not from Anthropic")
                content = deepcopy(message.provider_data["response"]["content"])
            else:
                content = []
                if message.text:
                    content.append({"type": "text", "text": message.text})
                content.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": deepcopy(call.arguments),
                    }
                    for call in message.tool_calls
                )
            conversation.append({"role": "assistant", "content": content})
            pending.update(ids)

        if pending:
            raise ValueError("Conversation ends with unresolved tool calls")
        if not conversation or conversation[0]["role"] != "user":
            raise ValueError("Conversation must start with a user message")
        return "\n\n".join(system), conversation

    def _normalize_reply(self, response: Any) -> ModelReply:
        raw = response.model_dump(mode="json", exclude_none=True)
        reason = raw.get("stop_reason")
        if reason not in _STOP_REASONS:
            raise ModelError(f"Unsupported Claude stop reason: {reason}")
        text = []
        calls = []
        for block in raw["content"]:
            kind = block["type"]
            if kind == "text":
                text.append(block["text"])
            elif kind == "tool_use":
                calls.append(ToolCall(block["id"], block["name"], block["input"]))
            elif kind not in ("thinking", "redacted_thinking"):
                raise ModelError(f"Unsupported Claude content block: {kind}")

        usage = raw["usage"]
        input_tokens = sum(
            self._usage_count(usage, key)
            for key in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        )
        output_tokens = self._usage_count(usage, "output_tokens")
        return ModelReply(
            text="\n".join(text),
            tool_calls=tuple(calls),
            stop_reason=_STOP_REASONS[reason],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_data={"provider": "anthropic", "response": raw},
        )

    def _usage_count(self, usage: dict[str, object], key: str) -> int:
        if key in ("input_tokens", "output_tokens") and key not in usage:
            raise ModelError(f"Claude response is missing {key}")
        value = usage.get(key)
        if value is None and key.startswith("cache_"):
            return 0
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ModelError(f"Invalid Claude token usage: {key}")
        return value

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ClaudeClient:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
