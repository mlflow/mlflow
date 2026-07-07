import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

from mlflow.assistant.providers.base import (
    AssistantProvider,
    CLINotInstalledError,
    NotAuthenticatedError,
    load_config,
)
from mlflow.assistant.providers.prompts import ASSISTANT_SYSTEM_PROMPT
from mlflow.assistant.types import Event, Message, TextBlock
from mlflow.server.assistant.session import clear_process_pid, save_process_pid
from mlflow.tracing.constant import CostKey, TokenUsageKey
from mlflow.tracing.utils import calculate_cost_by_model_and_token_usage

_logger = logging.getLogger(__name__)

_CODEX_BINARY = "codex"


class CodexProvider(AssistantProvider):
    @property
    def name(self) -> str:
        return "codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex"

    @property
    def description(self) -> str:
        return "AI-powered assistant using the OpenAI Codex CLI"

    def is_available(self) -> bool:
        return shutil.which(_CODEX_BINARY) is not None

    def check_connection(self, echo: Callable[[str], None] | None = None) -> None:
        codex_path = shutil.which(_CODEX_BINARY)
        if not codex_path:
            if echo:
                echo("codex CLI not found")
            raise CLINotInstalledError(
                "OpenAI Codex CLI is not installed. Install it with: npm install -g @openai/codex"
            )

        if echo:
            echo(f"codex CLI found: {codex_path}")
            echo("Checking connection... (this may take a few seconds)")

        try:
            result = subprocess.run(
                [
                    codex_path,
                    "exec",
                    "--json",
                    "--dangerously-bypass-approvals-and-sandbox",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "-",
                ],
                input=b"say hi",
                capture_output=True,
                timeout=30,
            )

            if result.returncode == 0:
                if echo:
                    echo("Connection verified")
                return

            stderr = result.stderr.decode("utf-8", errors="replace").lower()
            if (
                "auth" in stderr
                or "login" in stderr
                or "unauthorized" in stderr
                or "api key" in stderr
            ):
                error_msg = "Not authenticated. Please set OPENAI_API_KEY or run: codex login"
            else:
                error_msg = (
                    result.stderr.decode("utf-8", errors="replace").strip()
                    or f"Process exited with code {result.returncode}"
                )

            if echo:
                echo(f"Authentication failed: {error_msg}")
            raise NotAuthenticatedError(error_msg)

        except subprocess.TimeoutExpired:
            if echo:
                echo("Connection check timed out")
            raise NotAuthenticatedError("Connection check timed out")
        except subprocess.SubprocessError as e:
            if echo:
                echo(f"Error checking connection: {e}")
            raise NotAuthenticatedError(str(e))

    def resolve_skills_path(self, base_directory: Path) -> Path:
        return base_directory / ".codex" / "skills"

    async def astream(
        self,
        prompt: str,
        tracking_uri: str,
        session_id: str | None = None,
        mlflow_session_id: str | None = None,
        cwd: Path | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Event, None]:
        codex_path = shutil.which(_CODEX_BINARY)
        if not codex_path:
            yield Event.from_error(
                "codex CLI not found. Please install the OpenAI Codex CLI "
                "and ensure it's in your PATH."
            )
            return

        config = load_config(self.name)

        if context:
            user_text = f"<context>\n{json.dumps(context)}\n</context>\n\n{prompt}"
        else:
            user_text = prompt

        if session_id:
            user_message = user_text
        else:
            sys_prompt = ASSISTANT_SYSTEM_PROMPT.format(tracking_uri=tracking_uri)
            user_message = (
                f"<system_instructions>\n{sys_prompt}\n</system_instructions>\n\n{user_text}"
            )

        cmd = [
            codex_path,
            "exec",
            "--json",
            "--sandbox",
            "danger-full-access",
            "--skip-git-repo-check",
        ]

        if config.model and config.model != "default":
            cmd.extend(["-m", config.model])

        if session_id:
            cmd.extend(["resume", session_id])

        cmd.append("-")

        thread_id = ""
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                limit=100 * 1024 * 1024,
                env={**os.environ, "MLFLOW_TRACKING_URI": tracking_uri},
            )

            if mlflow_session_id and process.pid:
                save_process_pid(mlflow_session_id, process.pid)

            assert process.stdin is not None
            assert process.stdout is not None
            process.stdin.write(user_message.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()

            async for line in process.stdout:
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    data = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "thread.started":
                    thread_id = data.get("thread_id", "")
                    continue

                # Emit token usage before the result event, which closes the
                # stream on the client once received.
                if data.get("type") == "turn.completed" and (usage := data.get("usage")):
                    model = config.model if config.model and config.model != "default" else None
                    yield self._build_usage_event(usage, model)
                    continue

                event = self._parse_event(data)
                if event is not None:
                    yield event

            await process.wait()

            if process.returncode == -9:
                yield Event.from_interrupted()
                return

            if process.returncode != 0:
                assert process.stderr is not None
                stderr_bytes = await process.stderr.read()
                error_msg = (
                    stderr_bytes.decode("utf-8", errors="replace").strip()
                    or f"Process exited with code {process.returncode}"
                )
                yield Event.from_error(error_msg)
            else:
                yield Event.from_result(result=None, session_id=thread_id)

        except Exception as e:
            _logger.exception("Error running Codex CLI")
            yield Event.from_error(str(e))
        finally:
            if mlflow_session_id:
                clear_process_pid(mlflow_session_id)
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()

    @staticmethod
    def _build_usage_event(usage: dict[str, Any], model: str | None) -> Event:
        """Translate Codex CLI token usage into a UI usage stream event.

        Codex reports OpenAI-style usage on `turn.completed`, where
        `input_tokens` is the cache-inclusive prompt total and
        `cached_input_tokens` is the cached subset of it (not additive). We
        price it via the LiteLLM-backed cost catalog when the model is known;
        Codex bills against the user's plan rather than reporting a dollar
        cost, so total_cost_usd is None when the model isn't in the catalog.
        The shape matches the usage event emitted by the other providers so the
        UI handles them identically.
        """
        prompt_tokens = usage.get("input_tokens") or 0
        completion_tokens = usage.get("output_tokens") or 0
        cache_read = usage.get("cached_input_tokens")

        cost_usage: dict[str, int] = {
            TokenUsageKey.INPUT_TOKENS: prompt_tokens,
            TokenUsageKey.OUTPUT_TOKENS: completion_tokens,
        }
        if cache_read is not None:
            cost_usage[TokenUsageKey.CACHE_READ_INPUT_TOKENS] = cache_read

        cost = calculate_cost_by_model_and_token_usage(model, cost_usage)

        return Event.from_stream_event({
            "type": "usage",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "total_cost_usd": cost[CostKey.TOTAL_COST] if cost else None,
            },
        })

    def _parse_event(self, data: dict[str, Any]) -> Event | None:
        event_type = data.get("type")

        if event_type == "item.completed":
            item = data.get("item", {})
            if item.get("type") == "agent_message":
                if text := item.get("text", ""):
                    return Event.from_message(
                        Message(role="assistant", content=[TextBlock(text=text)])
                    )

        return None
