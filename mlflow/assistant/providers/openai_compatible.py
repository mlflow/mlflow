"""Generic OpenAI-compatible chat-completions provider for MLflow Assistant.

Drives any server that exposes `POST /v1/chat/completions` in OpenAI SSE form:
MLflow AI Gateway, Ollama (via its `/v1` shim), vLLM, LM Studio, etc.

:class:`OpenAICompatibleProvider` owns the shared streaming/tool-loop
machinery. Concrete presets (``MlflowGatewayProvider``, ``OllamaProvider``,
etc.) subclass it and forward their per-preset constants through
``super().__init__()``.
"""

import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, AsyncGenerator

import aiohttp

from mlflow.assistant.config import PermissionsConfig
from mlflow.assistant.providers.base import (
    AssistantProvider,
    NotAuthenticatedError,
    ProviderNotConfiguredError,
    load_config,
)
from mlflow.assistant.providers.prompts import ASSISTANT_SYSTEM_PROMPT
from mlflow.assistant.providers.tool_executor import (
    build_tools_schema,
    execute_tool,
    static_permission_error,
)
from mlflow.assistant.types import Event, Message, ToolResultBlock, ToolUseBlock

_logger = logging.getLogger(__name__)

# OpenAI-compatible servers have no server-side session state, so the client
# carries the full message history as JSON in conversation_history. 500 KB stays
# well below typical LLM context windows and gives tool-heavy multi-turn
# conversations enough headroom to avoid frequent trimming. Older turns
# are dropped first; the system message at index 0 is always kept.
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
    if _total_session_bytes(messages) > _MAX_SESSION_BYTES:
        _logger.warning(
            "Session payload still exceeds %d bytes after trimming; the active "
            "turn is too large to drop. The gateway will likely return a "
            "context-length error.",
            _MAX_SESSION_BYTES,
        )
    return messages


def _tool_result_ids(messages: list[dict[str, Any]]) -> set[str]:
    """IDs of tool_calls that already have a `tool` result message."""
    return {
        m["tool_call_id"] for m in messages if m.get("role") == "tool" and m.get("tool_call_id")
    }


def _pending_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool calls awaiting a decision in the most recent assistant turn.

    A history whose last assistant message carries `tool_calls` without matching
    `tool` results is a turn paused at a permission prompt. Returning a non-empty
    list signals *resume mode*: apply the user's decision(s) and continue instead
    of starting a new user turn.
    """
    last_assistant = next(
        (m for m in reversed(messages) if m.get("role") == "assistant" and m.get("tool_calls")),
        None,
    )
    if last_assistant is None:
        return []
    resolved = _tool_result_ids(messages)
    return [tc for tc in last_assistant["tool_calls"] if tc["id"] not in resolved]


def _trailing_partial_tag_len(buf: str, tag: str) -> int:
    """Length of the longest suffix of `buf` that is a non-empty prefix of `tag`.

    Used to hold back partial `<think>` / `</think>` markers that may be
    completed by the next streamed chunk. Example: if `buf` ends with
    "foo<th" and `tag` is "<think>", this returns 3 (the "<th" tail).
    """
    max_n = min(len(buf), len(tag) - 1)
    for n in range(max_n, 0, -1):
        if tag.startswith(buf[-n:]):
            return n
    return 0


def _strip_think_blocks(buf: str, in_think: bool) -> tuple[str, str, bool]:
    """Strip <think>...</think> spans that some reasoning models emit inline.

    Returns (emit_text, remaining_buf, new_in_think_flag). The remaining_buf
    holds a partial open/close tag that should be re-fed next chunk so that
    a tag split across SSE frames (e.g. "foo<th" then "ink>secret</think>")
    doesn't leak <think> markup to the user.
    """
    emit = ""
    while buf:
        if in_think:
            end = buf.find("</think>")
            if end == -1:
                # Don't emit anything while inside a think span. Hold a
                # potential partial closing tag at the tail so the next
                # chunk can complete it.
                hold = _trailing_partial_tag_len(buf, "</think>")
                return emit, buf[-hold:] if hold else "", in_think
            buf = buf[end + len("</think>") :]
            in_think = False
        else:
            start = buf.find("<think>")
            if start == -1:
                # No opening tag visible. Hold a potential partial opening
                # tag at the tail; emit everything before it.
                if hold := _trailing_partial_tag_len(buf, "<think>"):
                    emit += buf[:-hold]
                    return emit, buf[-hold:], in_think
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
    """Base provider for any server exposing `POST /v1/chat/completions`."""

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
        client_carries_history: bool = False,
    ):
        self.client_carries_history = client_carries_history
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
            # MLflow Gateway) cannot be probed from the assistant backend —
            # the frontend talks directly to the gateway endpoints API for
            # verification. Surface this clearly so the health endpoint
            # doesn't claim a successful probe it did not perform.
            raise NotImplementedError(
                f"{self._display_name} connection is verified by the frontend; "
                "the assistant backend has no probe to run."
            )
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
            raise NotImplementedError(f"Model listing is not supported for provider '{self.name}'")
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

    async def astream_stateless(
        self,
        prompt: str,
        tracking_uri: str,
        conversation_history: str | None = None,
        cwd: Path | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Event, None]:
        config = self._load_config()
        if config is None:
            yield Event.from_error(
                f"{self._display_name} is not configured. {self._connection_hint}"
            )
            return
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
        if conversation_history:
            try:
                messages = json.loads(conversation_history)
            except (json.JSONDecodeError, TypeError):
                _logger.warning("Failed to decode conversation history; starting fresh")
                messages = []

        if not messages:
            sys_content = ASSISTANT_SYSTEM_PROMPT.format(tracking_uri=tracking_uri)
            messages.append({"role": "system", "content": sys_content})

        tool_decisions = (context or {}).get("tool_decisions") or {}
        # A history whose last assistant turn carries tool_calls without results is
        # a turn paused at a permission prompt. We resume it (applying the decisions
        # in `tool_decisions`) ONLY when a decision was actually delivered. Deriving
        # this from history alone is unsafe: if the user cancels at the prompt (a
        # no-op for this provider, so the unresolved tool_calls stay in history) and
        # then sends a new message, we must start a fresh turn — not silently
        # re-resume the abandoned calls and drop their message.
        tool_calls_awaiting_decision = _pending_tool_calls(messages)
        # TODO (joshuawong-db) This should be refactored into a helper function when
        # more providers support tool calls as it has a close coupling with api.py logic.
        is_resuming = bool(tool_decisions) and bool(tool_calls_awaiting_decision)
        if not is_resuming:
            # Close out any orphaned tool_calls (e.g. cancelled at a prompt) before
            # the new user message: OpenAI requires a result for every tool_call, so
            # an unanswered one would make the gateway reject the next request.
            for tc in tool_calls_awaiting_decision:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "Tool call cancelled by user.",
                })
            messages.append({"role": "user", "content": user_text})
        tools = build_tools_schema()

        headers = self._auth_headers(api_key)

        try:
            async with aiohttp.ClientSession() as session:
                while True:
                    if is_resuming:
                        # The pending tool_calls are already in history; apply the
                        # user's decision(s) instead of calling the model again.
                        # Implicit assumption: if user text and permission decisions
                        # are both present, the user message is being dropped by the backend.
                        assistant_tool_calls = tool_calls_awaiting_decision
                        is_resuming = False
                    else:
                        # `visible_text` accumulates the post-<think>-strip text
                        # that gets persisted into `messages`. Storing the raw
                        # pre-strip stream would re-feed the model's own
                        # reasoning back to it on the next turn.
                        visible_text = ""
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
                                # SSE frames start with `data: `. Skip event-name
                                # lines and comments, tolerate vanilla JSONL too.
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
                                    think_buf += text
                                    emit, think_buf, in_think = _strip_think_blocks(
                                        think_buf, in_think
                                    )
                                    if emit:
                                        visible_text += emit
                                        yield Event.from_stream_event({
                                            "type": "content_delta",
                                            "delta": {"text": emit},
                                        })

                                if tcs := delta.get("tool_calls"):
                                    for tc in tcs:
                                        _merge_tool_call_chunk(tool_calls_acc, tc)

                        if not tool_calls_acc:
                            if visible_text.strip():
                                messages.append({"role": "assistant", "content": visible_text})
                                break
                            # The model streamed no text this round. Any tool output or error this
                            # turn already reached the client as its own block, so we just surface
                            # that the model added no response rather than finalize silently.
                            yield Event.from_error(
                                "The model returned an empty response. Please try again."
                            )
                            return

                        # Normalize accumulated tool calls into the OpenAI
                        # assistant message format expected on the next turn.
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
                            "content": visible_text or None,
                            "tool_calls": assistant_tool_calls,
                        })

                    paused = False
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

                        # Permission gating. With full access (config) tools run without
                        # prompting. Otherwise we prompt only for a call the static policy wouldn't
                        # already permit: allowlisted Bash commands (e.g. `mlflow`) and in-workspace
                        # file ops run without a prompt, as they did before tool-call permissions
                        # existed; the prompt is kept as an override for the previously hard-denied
                        # calls. The turn pauses at a per-call Yes/No prompt; a later resume
                        # delivers the choice via `tool_decisions`, and an explicit allow overrides
                        # the static allowlist for that call. Calls the static policy already allows
                        # are left to it, enforced by execute_tool.
                        needs_prompt = (
                            static_permission_error(tool_name, tool_input, config.permissions, cwd)
                            is not None
                        )
                        gated = not config.permissions.full_access and needs_prompt
                        decision = tool_decisions.get(tc["id"])

                        # Emit the tool-use block when a call is first surfaced
                        # (about to prompt or run) — but not for a call we are
                        # resuming, whose block was emitted on the paused turn.
                        if not (gated and decision is not None):
                            yield Event.from_message(
                                Message(
                                    role="assistant",
                                    content=[
                                        ToolUseBlock(id=tc["id"], name=tool_name, input=tool_input)
                                    ],
                                )
                            )

                        if gated and decision is None:
                            # End the turn at the prompt; a resume request will
                            # deliver the decision and continue from here.
                            yield Event.from_permission_request(tc["id"], tool_name, tool_input)
                            paused = True
                            break

                        if gated and decision != "allow":
                            denied = "Permission denied by user."
                            yield Event.from_message(
                                Message(
                                    role="user",
                                    content=[
                                        ToolResultBlock(
                                            tool_use_id=tc["id"],
                                            content=denied,
                                            is_error=True,
                                        )
                                    ],
                                )
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": denied,
                            })
                            continue

                        effective_permissions = (
                            PermissionsConfig(full_access=True) if gated else config.permissions
                        )
                        result_str, is_error = await execute_tool(
                            tool_name,
                            tool_input,
                            cwd=cwd,
                            tracking_uri=tracking_uri,
                            permissions=effective_permissions,
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

                    if paused:
                        break

            new_history = json.dumps(_trim_session(messages))
            yield Event.from_conversation_history(new_history)

        except Exception as e:
            _logger.exception("Error communicating with %s", self._display_name)
            yield Event.from_error(str(e))
