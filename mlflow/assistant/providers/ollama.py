import json
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

from mlflow.assistant.providers.base import (
    AssistantProvider,
    CLINotInstalledError,
    NotAuthenticatedError,
    load_config,
)
from mlflow.assistant.providers.prompts import ASSISTANT_SYSTEM_PROMPT
from mlflow.assistant.providers.tool_executor import build_tools_schema, execute_tool
from mlflow.assistant.types import Event, Message, ToolResultBlock, ToolUseBlock

_logger = logging.getLogger(__name__)

# Cap serialized session history at 50 KB. Older turns are dropped first (system
# message at index 0 is always kept) to prevent unbounded growth over long conversations.
_MAX_SESSION_BYTES = 50 * 1024


def _trim_session(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    while len(json.dumps(messages).encode()) > _MAX_SESSION_BYTES and len(messages) > 2:
        messages.pop(1)
    return messages


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
        try:
            import ollama  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_host(self) -> str:
        try:
            config = load_config(self.name)
            return config.base_url or "http://localhost:11434"
        except RuntimeError:
            return "http://localhost:11434"

    def check_connection(self, echo: Callable[[str], None] | None = None) -> None:
        try:
            import ollama
        except ImportError:
            if echo:
                echo("ollama package not found")
            raise CLINotInstalledError(
                "The 'ollama' Python package is not installed. Install it with: pip install ollama"
            )

        host = self._get_host()
        if echo:
            echo(f"Connecting to Ollama at {host}...")

        try:
            client = ollama.Client(host=host)
            client.list()
            if echo:
                echo("Connection verified")
        except Exception as e:
            if echo:
                echo(f"Cannot connect to Ollama server: {e}")
            raise NotAuthenticatedError(
                f"Cannot connect to Ollama server at {host}. "
                "Make sure Ollama is running: ollama serve"
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
        try:
            import ollama
        except ImportError:
            yield Event.from_error(
                "The 'ollama' Python package is not installed. Install it with: pip install ollama"
            )
            return

        config = load_config(self.name)
        model = config.model if config.model and config.model != "default" else "llama3.2"
        host = config.base_url or "http://localhost:11434"

        if context:
            user_text = f"<context>\n{json.dumps(context)}\n</context>\n\n{prompt}"
        else:
            user_text = prompt

        messages: list[dict[str, Any]] = []
        if session_id:
            try:
                messages = json.loads(session_id)
            except (json.JSONDecodeError, TypeError):
                messages = []

        if not messages:
            sys_content = ASSISTANT_SYSTEM_PROMPT.format(tracking_uri=tracking_uri)
            messages.append({"role": "system", "content": sys_content})

        messages.append({"role": "user", "content": user_text})

        tools = build_tools_schema()
        client = ollama.AsyncClient(host=host)

        try:
            while True:
                accumulated_text = ""
                tool_calls_raw: list[Any] = []
                in_think_block = False
                think_buf = ""

                response_stream = await client.chat(
                    model=model,
                    messages=messages,
                    tools=tools,
                    stream=True,
                )

                async for chunk in response_stream:
                    msg = chunk.message

                    if delta := msg.content or "":
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

                    if msg.tool_calls:
                        tool_calls_raw.extend(msg.tool_calls)

                if not tool_calls_raw:
                    if accumulated_text:
                        messages.append({"role": "assistant", "content": accumulated_text})
                    break

                messages.append({
                    "role": "assistant",
                    "content": accumulated_text,
                    "tool_calls": [tc.model_dump() for tc in tool_calls_raw],
                })

                for tc in tool_calls_raw:
                    fn = tc.function
                    tool_name = fn.name or ""
                    raw_args = fn.arguments
                    tool_input = dict(raw_args) if raw_args else {}
                    tool_id = str(uuid.uuid4())

                    yield Event.from_message(
                        Message(
                            role="assistant",
                            content=[ToolUseBlock(id=tool_id, name=tool_name, input=tool_input)],
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
