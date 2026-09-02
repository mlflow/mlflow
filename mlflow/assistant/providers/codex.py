import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Literal

from mlflow.assistant.custom_view import (
    STRINGIFIED_CUSTOM_VIEW_RESPONSE_SCHEMA,
    STRINGIFIED_CUSTOM_VIEW_STRUCTURED_OUTPUT_INSTRUCTIONS,
    custom_view_response_events,
    is_custom_view_request,
    parse_custom_view_response,
)
from mlflow.assistant.providers.base import (
    AssistantProvider,
    CLINotInstalledError,
    NotAuthenticatedError,
    assistant_sandbox_enabled,
    load_config_or_default,
)
from mlflow.assistant.providers.prompts import ASSISTANT_SYSTEM_PROMPT
from mlflow.assistant.types import Event, Message, TextBlock
from mlflow.server.assistant.session import (
    clear_container_id,
    clear_process_pid,
    get_session_sandbox_home,
    save_container_id,
    save_process_pid,
)
from mlflow.tracing.constant import CostKey, TokenUsageKey
from mlflow.tracing.utils import calculate_cost_by_model_and_token_usage

_logger = logging.getLogger(__name__)

_CODEX_BINARY = "codex"

# In the sandbox, if codex cannot reach the API it streams "Reconnecting..." error events
# continuously (it never gives up on its own), so the container's no-output idle-timeout never
# fires. Bail if only connection errors arrive, with no progress, for this long.
_CODEX_NO_PROGRESS_TIMEOUT = 60.0

# Environment variables forwarded into the sandbox so the Codex CLI can authenticate. Host secrets
# outside this allowlist are intentionally NOT passed through. Codex authenticates via
# OPENAI_API_KEY or an interactive ``codex login`` whose credentials live in the per-session HOME;
# OPENAI_BASE_URL lets an operator point at a custom endpoint, and the proxy vars cover outbound
# proxy configuration.
_SANDBOX_AUTH_ENV_PASSTHROUGH = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


class CodexProvider(AssistantProvider):
    @property
    def name(self) -> str:
        return "codex"

    @property
    def display_name(self) -> str:
        return "Codex"

    @property
    def description(self) -> str:
        return "AI-powered assistant using the Codex CLI"

    @property
    def client_tool_delivery(self) -> Literal["structured"]:
        return "structured"

    @property
    def allows_remote_access(self) -> bool:
        # In local mode the CLI runs on the host, so it must stay localhost-only. In sandbox mode
        # it runs isolated in a container, so it can safely serve remote clients.
        return assistant_sandbox_enabled()

    def is_available(self) -> bool:
        # In sandbox mode the CLI runs inside the operator-provided image, not on the host, so
        # availability follows the sandbox being active rather than a host binary the operator is
        # not expected to install.
        return assistant_sandbox_enabled() or shutil.which(_CODEX_BINARY) is not None

    def check_connection(self, echo: Callable[[str], None] | None = None) -> None:
        if assistant_sandbox_enabled():
            # The CLI runs inside the operator image, not on the host, so there is no host binary
            # or host login to verify here; image presence and auth are checked when a turn starts
            # the container.
            if echo:
                echo("Assistant sandbox enabled; the Codex CLI runs in the sandbox image.")
            return
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
        if assistant_sandbox_enabled():
            async for event in self._astream_in_sandbox(
                prompt, tracking_uri, session_id, mlflow_session_id, cwd, context
            ):
                yield event
            return

        codex_path = shutil.which(_CODEX_BINARY)
        if not codex_path:
            yield Event.from_error(
                "codex CLI not found. Please install the OpenAI Codex CLI "
                "and ensure it's in your PATH."
            )
            return

        config = load_config_or_default(self.name)

        if context:
            user_text = f"<context>\n{json.dumps(context)}\n</context>\n\n{prompt}"
        else:
            user_text = prompt
        structured_custom_view = is_custom_view_request(context)
        if structured_custom_view:
            user_text = f"{user_text}\n\n{STRINGIFIED_CUSTOM_VIEW_STRUCTURED_OUTPUT_INSTRUCTIONS}"

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

        schema_path: Path | None = None
        schema_fd: int | None = None
        thread_id = ""
        structured_response_text: str | None = None
        codex_error: str | None = None
        process = None
        try:
            if structured_custom_view:
                schema_fd, raw_schema_path = tempfile.mkstemp(
                    prefix="mlflow-custom-view-", suffix=".schema.json"
                )
                schema_path = Path(raw_schema_path)
                schema_file = os.fdopen(schema_fd, "w")
                schema_fd = None  # fdopen owns and closes the descriptor from here.
                with schema_file:
                    json.dump(STRINGIFIED_CUSTOM_VIEW_RESPONSE_SCHEMA, schema_file)
                cmd.extend(["--output-schema", str(schema_path)])

            if config.model and config.model != "default":
                cmd.extend(["-m", config.model])

            if session_id:
                cmd.extend(["resume", session_id])

            cmd.append("-")

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

                if data.get("type") == "error":
                    codex_error = self._unwrap_error_message(data.get("message"))
                    continue

                if data.get("type") == "turn.failed":
                    codex_error = self._unwrap_error_message(
                        (data.get("error") or {}).get("message")
                    )
                    continue

                if data.get("type") == "thread.started":
                    thread_id = data.get("thread_id", "")
                    continue

                item = data.get("item") or {}
                if (
                    structured_custom_view
                    and data.get("type") == "item.completed"
                    and item.get("type") == "agent_message"
                ):
                    structured_response_text = item.get("text")
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
                    codex_error
                    or stderr_bytes.decode("utf-8", errors="replace").strip()
                    or f"Process exited with code {process.returncode}"
                )
                yield Event.from_error(error_msg)
            else:
                if structured_custom_view:
                    if structured_response_text is None:
                        yield Event.from_error(
                            "Codex did not return a structured Custom View response",
                            session_id=thread_id or session_id,
                        )
                        return
                    try:
                        response = parse_custom_view_response(structured_response_text)
                    except Exception as e:
                        yield Event.from_error(
                            f"Codex returned invalid Custom View output: {e}",
                            session_id=thread_id or session_id,
                        )
                        return
                    for event in custom_view_response_events(response):
                        yield event
                yield Event.from_result(result=None, session_id=thread_id)

        except Exception as e:
            _logger.exception("Error running Codex CLI")
            yield Event.from_exception(e)
        finally:
            if schema_fd is not None:
                try:
                    os.close(schema_fd)
                except OSError:
                    _logger.warning("Failed to close temp Custom View schema file descriptor")
            if mlflow_session_id:
                clear_process_pid(mlflow_session_id)
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            if schema_path:
                try:
                    schema_path.unlink(missing_ok=True)
                except OSError:
                    _logger.warning("Failed to remove temp Custom View schema file %s", schema_path)

    async def _astream_in_sandbox(
        self,
        prompt: str,
        tracking_uri: str,
        session_id: str | None,
        mlflow_session_id: str | None,
        cwd: Path | None,
        context: dict[str, Any] | None,
    ) -> AsyncGenerator[Event, None]:
        """Run the Codex CLI inside a hardened Docker container instead of on the host.

        Mirrors ``astream``'s command construction and event parsing, but the CLI, the working
        directory, and Codex's session/login state (its HOME) all live inside the container. The
        host path is left unchanged; this runs only when the assistant sandbox is active (see
        ``assistant_sandbox_enabled``). Runtime behavior (image contents, credentials, volume
        permissions) is validated
        against a live Docker daemon rather than in unit tests.
        """
        from mlflow.assistant.providers.tool_executor import _uri_without_credentials
        from mlflow.server.sandbox import (
            SandboxUnavailableError,
            sandbox_input_path,
            start_sandbox_process,
            to_container_host_uri,
        )

        config = load_config_or_default(self.name)
        # Everything the CLI does runs inside the container, so it must reach the tracking server
        # by the container-routable host (a loopback URI is rewritten to host.docker.internal).
        container_tracking_uri = to_container_host_uri(tracking_uri)

        if context:
            user_text = f"<context>\n{json.dumps(context)}\n</context>\n\n{prompt}"
        else:
            user_text = prompt
        structured_custom_view = is_custom_view_request(context)
        if structured_custom_view:
            user_text = f"{user_text}\n\n{STRINGIFIED_CUSTOM_VIEW_STRUCTURED_OUTPUT_INSTRUCTIONS}"

        if session_id:
            user_message = user_text
        else:
            # Use the external tracking URI in the prompt, not the container-only one: the prompt
            # asks the agent to build UI links the user opens in their browser, where
            # host.docker.internal would not resolve. Only the env below uses the container URI.
            sys_prompt = ASSISTANT_SYSTEM_PROMPT.format(tracking_uri=tracking_uri)
            user_message = (
                f"<system_instructions>\n{sys_prompt}\n</system_instructions>\n\n{user_text}"
            )

        cmd = [
            "codex",
            "exec",
            "--json",
            "--sandbox",
            "danger-full-access",
            "--skip-git-repo-check",
        ]
        input_files: dict[str, str] = {}
        if structured_custom_view:
            # Codex reads the schema by path; write it into the container's input mount.
            input_files["schema.json"] = json.dumps(STRINGIFIED_CUSTOM_VIEW_RESPONSE_SCHEMA)
            cmd.extend(["--output-schema", sandbox_input_path("schema.json")])
        if config.model and config.model != "default":
            cmd.extend(["-m", config.model])
        if session_id:
            cmd.extend(["resume", session_id])
        cmd.append("-")

        env = {"MLFLOW_TRACKING_URI": container_tracking_uri}
        # Mirror the Bash sandbox: forward the registry URI too so `mlflow` registry commands in
        # the container reach the same registry, credential-stripped and loopback-rewritten.
        registry_uri = os.environ.get("MLFLOW_REGISTRY_URI")
        if registry_uri and (safe := _uri_without_credentials("MLFLOW_REGISTRY_URI", registry_uri)):
            env["MLFLOW_REGISTRY_URI"] = to_container_host_uri(safe)
        for var in _SANDBOX_AUTH_ENV_PASSTHROUGH:
            if value := os.environ.get(var):
                # OPENAI_BASE_URL is an endpoint the CLI connects to from inside the container, so
                # a loopback value must be rewritten to reach the host; other vars pass through.
                env[var] = to_container_host_uri(value) if var == "OPENAI_BASE_URL" else value

        # Persist Codex's HOME (login credentials and session/thread state) across turns of the
        # same session so multi-turn conversations resume correctly.
        home_dir = get_session_sandbox_home(mlflow_session_id) if mlflow_session_id else None

        try:
            # start_sandbox_process uses the blocking docker-py client; keep it off the loop.
            proc = await asyncio.to_thread(
                start_sandbox_process,
                cmd,
                workdir=cwd,
                environment=env,
                stdin_data=user_message.encode("utf-8"),
                input_files=input_files or None,
                home_dir=home_dir,
            )
        except SandboxUnavailableError as e:
            yield Event.from_error(f"Assistant sandbox is enabled but could not start: {e}")
            return

        thread_id = ""
        structured_response_text: str | None = None
        codex_error: str | None = None
        try:
            # Recorded inside this try so a failure here still runs proc.cleanup() below.
            #
            # A cancel that arrives while the container is still starting (before this line, and
            # sandbox mode records no PID) finds nothing to kill, so that turn keeps running until
            # the idle timeout reaps it; the container is not leaked (cleanup() still runs). Closing
            # that startup window would take a per-session cancel flag checked here and is left as a
            # follow-up.
            if mlflow_session_id:
                save_container_id(mlflow_session_id, proc.container_id)
            last_progress = time.monotonic()
            try:
                async for raw_line in proc.iter_stdout_lines():
                    line_str = raw_line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue
                    try:
                        data = json.loads(line_str)
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "error":
                        codex_error = self._unwrap_error_message(data.get("message"))
                        # codex retries a failed connection forever, streaming these errors the
                        # whole time (so the idle-timeout never fires). If only errors arrive for
                        # too long, stop it rather than hang the turn.
                        if time.monotonic() - last_progress > _CODEX_NO_PROGRESS_TIMEOUT:
                            _logger.warning(
                                "Codex made no progress for %ss (repeated connection errors); "
                                "stopping.",
                                _CODEX_NO_PROGRESS_TIMEOUT,
                            )
                            await proc.akill()
                            yield Event.from_error(
                                "Codex could not connect and kept retrying without progress; "
                                f"stopped after {int(_CODEX_NO_PROGRESS_TIMEOUT)}s. "
                                f"Last error: {codex_error}"
                            )
                            return
                        continue
                    # Any non-error line is progress; reset the no-progress deadline.
                    last_progress = time.monotonic()
                    if data.get("type") == "turn.failed":
                        codex_error = self._unwrap_error_message(
                            (data.get("error") or {}).get("message")
                        )
                        continue
                    if data.get("type") == "thread.started":
                        thread_id = data.get("thread_id", "")
                        continue
                    item = data.get("item") or {}
                    if (
                        structured_custom_view
                        and data.get("type") == "item.completed"
                        and item.get("type") == "agent_message"
                    ):
                        structured_response_text = item.get("text")
                        continue
                    if data.get("type") == "turn.completed" and (usage := data.get("usage")):
                        model = config.model if config.model and config.model != "default" else None
                        yield self._build_usage_event(usage, model)
                        continue
                    event = self._parse_event(data)
                    if event is not None:
                        yield event
            finally:
                if mlflow_session_id:
                    clear_container_id(mlflow_session_id)

            returncode = await proc.wait()
            # The idle-timeout watchdog kills a stuck CLI; surface that clearly. It exits 137 like
            # a cancellation kill, so this must be checked first.
            if proc.timed_out:
                yield Event.from_error(
                    "Codex produced no output for too long and was stopped. If a custom "
                    "OPENAI_BASE_URL is set, it may not be reachable from the sandbox."
                )
                return
            # A killed container exits 137 (128 + SIGKILL), which is how cancellation surfaces.
            if returncode == 137:
                yield Event.from_interrupted()
                return
            if returncode != 0:
                stderr = (await proc.read_stderr()).decode("utf-8", errors="replace").strip()
                yield Event.from_error(
                    codex_error or stderr or f"Process exited with code {returncode}"
                )
            else:
                if structured_custom_view:
                    if structured_response_text is None:
                        yield Event.from_error(
                            "Codex did not return a structured Custom View response",
                            session_id=thread_id or session_id,
                        )
                        return
                    try:
                        response = parse_custom_view_response(structured_response_text)
                    except Exception as e:
                        yield Event.from_error(
                            f"Codex returned invalid Custom View output: {e}",
                            session_id=thread_id or session_id,
                        )
                        return
                    for event in custom_view_response_events(response):
                        yield event
                yield Event.from_result(result=None, session_id=thread_id)
        except Exception as e:
            _logger.exception("Error running Codex CLI in sandbox")
            yield Event.from_exception(e)
        finally:
            await proc.aclose()

    @staticmethod
    def _unwrap_error_message(message: Any) -> str | None:
        value = message
        for _ in range(4):
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    return value
            elif isinstance(value, dict):
                error = value.get("error")
                if isinstance(error, dict) and error.get("message"):
                    value = error["message"]
                elif value.get("message"):
                    value = value["message"]
                else:
                    return json.dumps(value)
            elif value is None:
                return None
            else:
                return str(value)
        return str(value)

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
                # Subset of prompt_tokens re-read from the prompt cache (cheap). Surfaced
                # so the UI can distinguish fresh input from resent, cached context.
                "cache_read_tokens": cache_read or 0,
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
