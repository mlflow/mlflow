"""Generic OpenAI-compatible chat-completions provider for MLflow Assistant.

Drives any server that exposes `POST /v1/chat/completions` in OpenAI SSE form:
MLflow AI Gateway, Ollama (via its `/v1` shim), vLLM, LM Studio, etc.

The wire-level differences between these servers (model-listing endpoint, auth
header, error messages) are passed to the constructor as data, so a single
class can be registered multiple times with different presets in
`providers/__init__.py`.
"""

import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, AsyncGenerator

import aiohttp

from mlflow.assistant.providers.base import (
    AssistantProvider,
    NotAuthenticatedError,
    ProviderNotConfiguredError,
    load_config,
)
from mlflow.assistant.providers.prompts import ASSISTANT_SYSTEM_PROMPT
from mlflow.assistant.providers.tool_executor import build_tools_schema, execute_tool
from mlflow.assistant.types import Event, Message, ToolResultBlock, ToolUseBlock

_logger = logging.getLogger(__name__)

# OpenAI-compatible servers have no server-side session state, so we encode
# the full message history as JSON in the session_id field. 500 KB stays
# well below typical LLM context windows (gpt-5.4-mini handles ~200K tokens
# / roughly 800 KB of UTF-8 text) and gives tool-heavy multi-turn
# conversations enough headroom to avoid frequent trimming. Older turns are
# dropped first; the system message at index 0 is always kept.
_MAX_SESSION_BYTES = 500 * 1024
_JSON_LIST_OVERHEAD_BYTES = 2
_JSON_LIST_SEPARATOR_BYTES = 2

# Callable signature for the per-preset model-listing strategy.
# Takes (base_url, api_key) and returns a list of model/endpoint names.
# May be None for presets where the frontend handles listing directly
# (e.g. the in-server MLflow AI Gateway, which exposes its own ajax API).
ListModelsFn = Callable[[str, str | None], list[str]]

# Builds the chat-completions URL for a turn. Receives the configured
# `base_url` (may be empty when the preset routes through the MLflow server
# itself) and the `tracking_uri` (the MLflow server URL passed to astream).
# Returning None means the URL cannot be resolved and the turn should fail.
ChatUrlBuilder = Callable[[str | None, str], str | None]


def _default_chat_url_builder(base_url: str | None, _tracking_uri: str) -> str | None:
    """Default URL builder: appends `/v1/chat/completions` to base_url."""
    if not base_url:
        return None
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def _message_size_bytes(message: dict[str, Any]) -> int:
    return len(json.dumps(message).encode())


def _total_session_bytes(messages: list[dict[str, Any]]) -> int:
    if not messages:
        return _JSON_LIST_OVERHEAD_BYTES
    sizes = [_message_size_bytes(m) for m in messages]
    separators = max(0, len(sizes) - 1) * _JSON_LIST_SEPARATOR_BYTES
    return _JSON_LIST_OVERHEAD_BYTES + sum(sizes) + separators


def _trim_session(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim oldest conversation turns until the JSON-encoded size fits.

    Drops whole user-rooted turn groups (user message + the assistant/tool
    messages that follow it up to the next user message). Popping single
    messages would leave orphaned `tool` messages whose `tool_call_id`
    points at an assistant message that was already removed; OpenAI rejects
    those silently with an empty completion.
    """
    while _total_session_bytes(messages) > _MAX_SESSION_BYTES and len(messages) > 2:
        # End of the oldest turn = index of the next `user` message after
        # the first non-system message.
        end = 2
        while end < len(messages) and messages[end].get("role") != "user":
            end += 1
        if end >= len(messages):
            # Only one turn exists after the system message; we cannot drop
            # anything without losing the active turn. Stop and let the
            # gateway return a clear "context too long" error.
            break
        del messages[1:end]
    return messages


def _strip_think_blocks(buf: str, in_think: bool) -> tuple[str, str, bool]:
    """Strip <think>...</think> spans that some reasoning models emit inline.

    Returns (emit_text, remaining_buf, new_in_think_flag). The remaining_buf
    holds a partial open/close tag that should be re-fed next chunk.
    """
    emit = ""
    while buf:
        if in_think:
            end = buf.find("</think>")
            if end == -1:
                return emit, "", in_think
            buf = buf[end + len("</think>") :]
            in_think = False
        else:
            start = buf.find("<think>")
            if start == -1:
                emit += buf
                return emit, "", in_think
            emit += buf[:start]
            buf = buf[start + len("<think>") :]
            in_think = True
    return emit, "", in_think


def _merge_tool_call_chunk(accumulator: list[dict[str, Any]], chunk: dict[str, Any]) -> None:
    """Merge a streamed tool-call delta into the accumulator.

    OpenAI streams tool calls in pieces keyed by `index`: the first chunk
    typically carries `id` and `function.name`, subsequent chunks append to
    `function.arguments`.
    """
    idx = chunk.get("index", 0)
    while len(accumulator) <= idx:
        accumulator.append({"id": "", "function": {"name": "", "arguments": ""}})
    entry = accumulator[idx]
    if call_id := chunk.get("id"):
        entry["id"] = call_id
    fn = chunk.get("function") or {}
    if name := fn.get("name"):
        entry["function"]["name"] = name
    if args := fn.get("arguments"):
        entry["function"]["arguments"] += args


class OpenAICompatibleProvider(AssistantProvider):
    """Provider for any server exposing `POST /v1/chat/completions` in OpenAI form."""

    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        connection_hint: str,
        list_models_fn: ListModelsFn | None = None,
        chat_url_builder: ChatUrlBuilder = _default_chat_url_builder,
        default_base_url: str | None = None,
        skills_dirname: str | None = None,
    ):
        self._name = name
        self._display_name = display_name
        self._description = description
        self._list_models_fn = list_models_fn
        self._connection_hint = connection_hint
        self._chat_url_builder = chat_url_builder
        self._default_base_url = default_base_url
        # `.agent/skills` is the cross-tool convention for agent-skill discovery.
        # OAI-compat providers don't actually load skills at runtime, but the
        # path is preserved so users can opt-in later via skill_installer.
        self._skills_dirname = skills_dirname or ".agent"

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def description(self) -> str:
        return self._description

    def is_available(self) -> bool:
        return True

    def _load_config(self):
        try:
            return load_config(self.name)
        except RuntimeError:
            return None

    def _resolve_base_url(self, override: str | None = None) -> str | None:
        if override:
            return override.rstrip("/")
        config = self._load_config()
        if config and config.base_url:
            return config.base_url.rstrip("/")
        if self._default_base_url:
            return self._default_base_url.rstrip("/")
        return None

    def _auth_headers(self, api_key: str | None) -> dict[str, str]:
        if api_key:
            return {"Authorization": f"Bearer {api_key}"}
        return {}

    def check_connection(self, echo: Callable[[str], None] | None = None) -> None:
        if self._list_models_fn is None:
            # Presets without a backend listing strategy (e.g. the in-server
            # MLflow Gateway) have their connection verified by the frontend.
            return
        base_url = self._resolve_base_url()
        if not base_url:
            raise NotAuthenticatedError(
                f"{self._display_name} is not configured. {self._connection_hint}"
            )
        if echo:
            echo(f"Connecting to {self._display_name} at {base_url}...")
        config = self._load_config()
        api_key = getattr(config, "api_key", None) if config else None
        try:
            self._list_models_fn(base_url, api_key)
        except Exception as e:
            if echo:
                echo(f"Cannot connect: {e}")
            raise NotAuthenticatedError(
                f"Cannot connect to {self._display_name} at {base_url}. {self._connection_hint}"
            ) from e
        if echo:
            echo("Connection verified")

    def list_models(self, base_url: str | None = None, api_key: str | None = None) -> list[str]:
        if self._list_models_fn is None:
            raise NotImplementedError(
                f"Model listing is not supported for provider '{self.name}'"
            )
        resolved = self._resolve_base_url(base_url)
        if not resolved:
            raise ProviderNotConfiguredError(f"{self._display_name} base URL is not configured.")
        if api_key is None:
            config = self._load_config()
            api_key = getattr(config, "api_key", None) if config else None
        try:
            return self._list_models_fn(resolved, api_key)
        except Exception as e:
            raise ProviderNotConfiguredError(
                f"Cannot connect to {self._display_name} at {resolved}: {e}"
            ) from e

    def resolve_skills_path(self, base_directory: Path) -> Path:
        return base_directory / self._skills_dirname / "skills"

    async def astream(
        self,
        prompt: str,
        tracking_uri: str,
        session_id: str | None = None,
        mlflow_session_id: str | None = None,
        cwd: Path | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Event, None]:
        config = load_config(self.name)
        base_url = (config.base_url or self._default_base_url or "").rstrip("/") or None
        chat_url = self._chat_url_builder(base_url, tracking_uri)
        if not chat_url:
            yield Event.from_error(
                f"{self._display_name} chat URL could not be resolved. {self._connection_hint}"
            )
            return

        model = config.model if config.model and config.model != "default" else None
        api_key = getattr(config, "api_key", None)

        if model is None:
            if self._list_models_fn is None or not base_url:
                yield Event.from_error(
                    f"No model selected for {self._display_name}. {self._connection_hint}"
                )
                return
            try:
                available = self._list_models_fn(base_url, api_key)
            except Exception as e:
                yield Event.from_error(
                    f"Cannot connect to {self._display_name} at {base_url}: {e}. "
                    f"{self._connection_hint}"
                )
                return
            if not available:
                yield Event.from_error(
                    f"No models available from {self._display_name} at {base_url}."
                )
                return
            model = available[0]

        if context:
            user_text = f"<context>\n{json.dumps(context)}\n</context>\n\n{prompt}"
        else:
            user_text = prompt

        messages: list[dict[str, Any]] = []
        if session_id:
            try:
                messages = json.loads(session_id)
            except (json.JSONDecodeError, TypeError):
                _logger.warning("Failed to decode session history; starting a new session")
                messages = []

        if not messages:
            sys_content = ASSISTANT_SYSTEM_PROMPT.format(tracking_uri=tracking_uri)
            messages.append({"role": "system", "content": sys_content})

        messages.append({"role": "user", "content": user_text})
        tools = build_tools_schema()

        headers = self._auth_headers(api_key)

        try:
            async with aiohttp.ClientSession() as session:
                while True:
                    accumulated_text = ""
                    tool_calls_acc: list[dict[str, Any]] = []
                    in_think = False
                    think_buf = ""

                    payload = {
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "stream": True,
                    }
                    async with session.post(
                        chat_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=300),
                    ) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            yield Event.from_error(
                                f"{self._display_name} error {resp.status}: {body}"
                            )
                            return

                        async for raw_line in resp.content:
                            line = raw_line.strip()
                            if not line:
                                continue
                            # SSE frames start with `data: `. Skip event-name lines
                            # and comments, tolerate vanilla JSONL too.
                            if line.startswith(b"data:"):
                                line = line[len(b"data:") :].strip()
                            if line == b"[DONE]":
                                continue
                            if not line or line.startswith(b":"):
                                continue
                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                _logger.debug("Skipping non-JSON stream line: %r", line)
                                continue

                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}

                            if text := delta.get("content") or "":
                                accumulated_text += text
                                think_buf += text
                                emit, think_buf, in_think = _strip_think_blocks(think_buf, in_think)
                                if emit:
                                    yield Event.from_stream_event({
                                        "type": "content_delta",
                                        "delta": {"text": emit},
                                    })

                            if tcs := delta.get("tool_calls"):
                                for tc in tcs:
                                    _merge_tool_call_chunk(tool_calls_acc, tc)

                    if not tool_calls_acc:
                        if accumulated_text:
                            messages.append({"role": "assistant", "content": accumulated_text})
                        break

                    # Normalize accumulated tool calls into the OpenAI assistant
                    # message format expected on the next turn.
                    assistant_tool_calls = [
                        {
                            "id": tc["id"] or str(uuid.uuid4()),
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for tc in tool_calls_acc
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": accumulated_text,
                        "tool_calls": assistant_tool_calls,
                    })

                    for tc in assistant_tool_calls:
                        fn = tc["function"]
                        tool_name = fn["name"]
                        raw_args = fn["arguments"] or "{}"
                        try:
                            tool_input = (
                                json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                            )
                        except json.JSONDecodeError:
                            tool_input = {}

                        yield Event.from_message(
                            Message(
                                role="assistant",
                                content=[
                                    ToolUseBlock(id=tc["id"], name=tool_name, input=tool_input)
                                ],
                            )
                        )

                        result_str, is_error = await execute_tool(
                            tool_name,
                            tool_input,
                            cwd=cwd,
                            tracking_uri=tracking_uri,
                            permissions=config.permissions,
                        )

                        yield Event.from_message(
                            Message(
                                role="user",
                                content=[
                                    ToolResultBlock(
                                        tool_use_id=tc["id"],
                                        content=result_str,
                                        is_error=is_error,
                                    )
                                ],
                            )
                        )

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result_str,
                        })

            new_session_id = json.dumps(_trim_session(messages))
            yield Event.from_result(result=None, session_id=new_session_id)

        except Exception as e:
            _logger.exception("Error communicating with %s", self._display_name)
            yield Event.from_error(str(e))
