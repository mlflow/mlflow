import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, AsyncGenerator

import aiohttp
import requests

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

# Ollama has no server-side session state, so we encode the full message history
# as JSON in the session_id field. 50 KB is a conservative safety net: it is well
# below typical LLM context windows and small enough that it does not bloat the
# HTTP response payload. Older turns are dropped first; the system message at
# index 0 is always kept.
_MAX_SESSION_BYTES = 50 * 1024
_JSON_LIST_OVERHEAD_BYTES = 2
_JSON_LIST_SEPARATOR_BYTES = 2


def _message_size_bytes(message: dict[str, Any]) -> int:
    return len(json.dumps(message).encode())


def _trim_session(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    message_sizes = [_message_size_bytes(message) for message in messages]
    total_size = _JSON_LIST_OVERHEAD_BYTES + sum(message_sizes)
    if len(message_sizes) > 1:
        total_size += (len(message_sizes) - 1) * _JSON_LIST_SEPARATOR_BYTES

    while total_size > _MAX_SESSION_BYTES and len(messages) > 2:
        messages.pop(1)
        total_size -= message_sizes.pop(1) + _JSON_LIST_SEPARATOR_BYTES
    return messages


def _list_models_http(host: str) -> list[str]:
    response = requests.get(f"{host}/api/tags", timeout=10)
    response.raise_for_status()
    return [m["model"] for m in response.json().get("models", []) if m.get("model")]


class OllamaProvider(AssistantProvider):
    @property
    def name(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama"

    @property
    def description(self) -> str:
        return "AI-powered assistant using a locally running Ollama server"

    def is_available(self) -> bool:
        return True

    def _get_host(self) -> str:
        try:
            config = load_config(self.name)
            return config.base_url or "http://localhost:11434"
        except RuntimeError:
            return "http://localhost:11434"

    def check_connection(self, echo: Callable[[str], None] | None = None) -> None:
        host = self._get_host()
        if echo:
            echo(f"Connecting to Ollama at {host}...")
        try:
            _list_models_http(host)
            if echo:
                echo("Connection verified")
        except Exception as e:
            if echo:
                echo(f"Cannot connect to Ollama server: {e}")
            raise NotAuthenticatedError(
                f"Cannot connect to Ollama server at {host}. "
                "Make sure Ollama is running: ollama serve"
            ) from e

    def list_models(self, base_url: str | None = None) -> list[str]:
        host = base_url or self._get_host()
        try:
            return _list_models_http(host)
        except Exception as e:
            raise ProviderNotConfiguredError(
                f"Cannot connect to Ollama server at {host}: {e}"
            ) from e

    def resolve_skills_path(self, base_directory: Path) -> Path:
        return base_directory / ".ollama" / "skills"

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
        host = config.base_url or "http://localhost:11434"
        model = config.model if config.model and config.model != "default" else None

        if model is None:
            try:
                available = _list_models_http(host)
            except Exception as e:
                yield Event.from_error(
                    f"Cannot connect to Ollama at {host}: {e}. "
                    "Make sure Ollama is running: ollama serve"
                )
                return
            if not available:
                yield Event.from_error(
                    "No Ollama models found. Pull one first: ollama pull llama3.2, "
                    "then reopen the setup wizard to select a model."
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

        try:
            async with aiohttp.ClientSession() as session:
                while True:
                    accumulated_text = ""
                    tool_calls_raw: list[dict[str, Any]] = []
                    in_think_block = False
                    think_buf = ""

                    async with session.post(
                        f"{host}/api/chat",
                        json={"model": model, "messages": messages, "tools": tools, "stream": True},
                        timeout=aiohttp.ClientTimeout(total=300),
                    ) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            yield Event.from_error(f"Ollama error {resp.status}: {body}")
                            return

                        async for raw_line in resp.content:
                            line = raw_line.strip()
                            if not line:
                                continue
                            chunk = json.loads(line)
                            msg = chunk.get("message", {})

                            if delta := msg.get("content") or "":
                                accumulated_text += delta
                                think_buf += delta
                                emit = ""
                                while think_buf:
                                    if in_think_block:
                                        end = think_buf.find("</think>")
                                        if end == -1:
                                            think_buf = ""
                                            break
                                        think_buf = think_buf[end + len("</think>") :]
                                        in_think_block = False
                                    else:
                                        start = think_buf.find("<think>")
                                        if start == -1:
                                            emit += think_buf
                                            think_buf = ""
                                            break
                                        emit += think_buf[:start]
                                        think_buf = think_buf[start + len("<think>") :]
                                        in_think_block = True
                                if emit:
                                    yield Event.from_stream_event({
                                        "type": "content_delta",
                                        "delta": {"text": emit},
                                    })

                            if raw_tool_calls := msg.get("tool_calls"):
                                tool_calls_raw.extend(raw_tool_calls)

                    if not tool_calls_raw:
                        if accumulated_text:
                            messages.append({"role": "assistant", "content": accumulated_text})
                        break

                    messages.append({
                        "role": "assistant",
                        "content": accumulated_text,
                        "tool_calls": tool_calls_raw,
                    })

                    for tc in tool_calls_raw:
                        fn = tc.get("function", {})
                        tool_name = fn.get("name", "")
                        tool_input = fn.get("arguments", {})
                        if isinstance(tool_input, str):
                            tool_input = json.loads(tool_input)
                        tool_id = str(uuid.uuid4())

                        yield Event.from_message(
                            Message(
                                role="assistant",
                                content=[
                                    ToolUseBlock(id=tool_id, name=tool_name, input=tool_input)
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
                                        tool_use_id=tool_id,
                                        content=result_str,
                                        is_error=is_error,
                                    )
                                ],
                            )
                        )

                        messages.append({
                            "role": "tool",
                            "content": result_str,
                            "tool_name": tool_name,
                        })

            new_session_id = json.dumps(_trim_session(messages))
            yield Event.from_result(result=None, session_id=new_session_id)

        except Exception as e:
            _logger.exception("Error communicating with Ollama")
            yield Event.from_error(str(e))
