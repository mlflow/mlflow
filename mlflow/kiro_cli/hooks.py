"""Agent-config upsert, surgical disable, and the five Kiro CLI hook handlers."""

import json
import os
import sys
from pathlib import Path
from typing import Any

from mlflow.environment_variables import (
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from mlflow.kiro_cli.config import (
    ENVIRONMENT_FIELD,
    HOOK_FIELD_COMMAND,
    HOOK_FIELD_HOOKS,
    MLFLOW_HOOK_IDENTIFIER,
    MLFLOW_TRACING_ENABLED,
    is_tracing_enabled,
    load_kiro_config,
    save_kiro_config,
)

# ============================================================================
# HOOK EVENT → SUBCOMMAND MAPPING
# ============================================================================

HOOK_EVENTS = {
    "agentSpawn": "agent-spawn-hook",
    "userPromptSubmit": "user-prompt-submit-hook",
    "preToolUse": "pre-tool-use-hook",
    "postToolUse": "post-tool-use-hook",
    "stop": "stop-hook",
}

# Maximum characters for transient-handler log snippets (Req 13.6)
_TRANSIENT_LOG_PREVIEW = 200


# ============================================================================
# HOOK CONFIGURATION UTILITIES
# ============================================================================


def upsert_hook(agent_config: dict[str, Any], event: str, subcommand: str) -> None:
    """Insert or update a single MLflow hook in the agent configuration.

    Kiro CLI uses a flat hook shape: ``"<event>": [HookEntry, ...]`` where each
    ``HookEntry`` is ``{"type": "command", "command": "..."}``  — *not* Claude's
    nested ``{"hooks": [...]}`` shape.

    If an existing entry's ``command`` contains :data:`MLFLOW_HOOK_IDENTIFIER`,
    its ``command`` is overwritten in place (preserving list position).  Otherwise
    a new entry is appended.

    Args:
        agent_config: The agent configuration dictionary to modify (mutated in place).
        event: The hook event key (e.g., ``"agentSpawn"``).
        subcommand: The CLI subcommand name (e.g., ``"agent-spawn-hook"``).
    """
    if HOOK_FIELD_HOOKS not in agent_config:
        agent_config[HOOK_FIELD_HOOKS] = {}

    if event not in agent_config[HOOK_FIELD_HOOKS]:
        agent_config[HOOK_FIELD_HOOKS][event] = []

    mlflow_cmd = "uv run mlflow" if "UV" in os.environ else "mlflow"
    hook_command = f"{mlflow_cmd} autolog kiro-cli {subcommand}"

    # Scan for an existing MLflow entry and update in place
    for entry in agent_config[HOOK_FIELD_HOOKS][event]:
        cmd = entry.get(HOOK_FIELD_COMMAND, "")
        if MLFLOW_HOOK_IDENTIFIER in cmd:
            entry[HOOK_FIELD_COMMAND] = hook_command
            return

    # No existing MLflow entry — append a new one
    agent_config[HOOK_FIELD_HOOKS][event].append({
        "type": "command",
        HOOK_FIELD_COMMAND: hook_command,
    })


def setup_hooks_config(agent_config_path: Path) -> None:
    """Create or merge MLflow hooks into ``.kiro/agents/kiro_default.json``.

    Loads the existing agent config (or starts from a default skeleton),
    forces ``name = "kiro_default"``, upserts all five hook events, and
    writes the result with 2-space indent / UTF-8.

    Args:
        agent_config_path: Path to the agent config JSON file.
    """
    config = load_kiro_config(agent_config_path)

    # Start from skeleton when the file was missing or empty
    if not config:
        config = {"name": "kiro_default", HOOK_FIELD_HOOKS: {}}

    # Force the canonical agent name; preserve every other top-level field
    config["name"] = "kiro_default"

    if HOOK_FIELD_HOOKS not in config:
        config[HOOK_FIELD_HOOKS] = {}

    # Upsert all five events
    for event, subcommand in HOOK_EVENTS.items():
        upsert_hook(config, event, subcommand)

    save_kiro_config(agent_config_path, config)


# ============================================================================
# SURGICAL DISABLE
# ============================================================================


def disable_tracing_hooks(agent_config_path: Path, settings_path: Path) -> bool:
    """Surgically remove only MLflow-identified content from both config files.

    In the Agent_Config, for each of the five events, entries whose ``command``
    contains :data:`MLFLOW_HOOK_IDENTIFIER` are dropped.  Empty event lists and
    an empty ``hooks`` object are cleaned up.  If the file reduces to only
    ``{"name": "kiro_default"}`` with no other top-level fields, the file is
    deleted.

    In the Settings_File, the four MLflow env keys are removed from ``env``.
    If ``env`` becomes empty it is deleted; if no top-level keys remain the
    file is deleted.

    Args:
        agent_config_path: Path to ``.kiro/agents/kiro_default.json``.
        settings_path: Path to ``.kiro/settings.json``.

    Returns:
        ``True`` if any MLflow state was removed from either file.
    """
    hooks_removed = _disable_agent_config(agent_config_path)
    env_removed = _disable_settings(settings_path)
    return hooks_removed or env_removed


def _disable_agent_config(agent_config_path: Path) -> bool:
    """Remove MLflow hook entries from the agent config file.

    Returns ``True`` if any entries were removed.
    """
    if not agent_config_path.exists():
        return False

    config = load_kiro_config(agent_config_path)
    if not config:
        return False

    removed = False
    hooks = config.get(HOOK_FIELD_HOOKS, {})

    for event in list(hooks.keys()):
        entries = hooks[event]
        filtered = [
            entry
            for entry in entries
            if MLFLOW_HOOK_IDENTIFIER not in entry.get(HOOK_FIELD_COMMAND, "")
        ]
        if len(filtered) < len(entries):
            removed = True
        if filtered:
            hooks[event] = filtered
        else:
            del hooks[event]

    # Clean up empty hooks object
    if HOOK_FIELD_HOOKS in config and not config[HOOK_FIELD_HOOKS]:
        del config[HOOK_FIELD_HOOKS]

    # Decide whether to unlink or rewrite
    only_name = set(config.keys()) == {"name"} and config.get("name") == "kiro_default"
    if only_name:
        agent_config_path.unlink(missing_ok=True)
    else:
        save_kiro_config(agent_config_path, config)

    return removed


def _disable_settings(settings_path: Path) -> bool:
    """Remove MLflow env keys from the settings file.

    Returns ``True`` if any keys were removed.
    """
    if not settings_path.exists():
        return False

    config = load_kiro_config(settings_path)
    if not config:
        return False

    removed = False
    env = config.get(ENVIRONMENT_FIELD, {})

    mlflow_keys = [
        MLFLOW_TRACING_ENABLED,
        MLFLOW_TRACKING_URI.name,
        MLFLOW_EXPERIMENT_ID.name,
        MLFLOW_EXPERIMENT_NAME.name,
    ]
    for key in mlflow_keys:
        if key in env:
            del env[key]
            removed = True

    # Clean up empty env object
    if ENVIRONMENT_FIELD in config and not config[ENVIRONMENT_FIELD]:
        del config[ENVIRONMENT_FIELD]

    # Unlink file if no top-level keys remain; otherwise rewrite
    if not config:
        settings_path.unlink(missing_ok=True)
    else:
        save_kiro_config(settings_path, config)

    return removed


# ============================================================================
# HOOK HANDLER BOILERPLATE
# ============================================================================


def _run_transient_handler(handler_name: str, body_fn) -> None:
    """Shared boilerplate for the four transient (non-stop) hook handlers.

    1. Fast-path: if tracing is disabled, emit ``{"continue": true}`` and return.
    2. Read stdin JSON.
    3. Call *body_fn(hook_data)* inside a try/except.
    4. On any exception: log ERROR, emit ``{"continue": true}``, exit 0.

    Transient handlers **always** exit 0 and emit ``{"continue": true}``.
    """
    if not is_tracing_enabled():
        print(json.dumps({"continue": True}))  # noqa: T201
        return

    try:
        from mlflow.kiro_cli.tracing import get_hook_response, get_logger, read_hook_input

        hook_data = read_hook_input()
        body_fn(hook_data, get_logger)
        print(json.dumps(get_hook_response()))  # noqa: T201
    except Exception as e:
        # Import may have succeeded or failed — try to log
        try:
            from mlflow.kiro_cli.tracing import get_logger

            get_logger().error("Error in %s handler: %s", handler_name, e, exc_info=True)
        except Exception:
            pass
        print(json.dumps({"continue": True}))  # noqa: T201


# ============================================================================
# TRANSIENT HOOK HANDLERS
# ============================================================================


def agent_spawn_hook_handler() -> None:
    """CLI hook handler for ``agentSpawn`` — log session start, pass through."""

    def _body(hook_data: dict[str, Any], get_logger) -> None:
        from mlflow.kiro_cli.tracing import KIRO_TRACING_LEVEL

        session_id = hook_data.get("session_id", "")
        cwd = hook_data.get("cwd", "")
        get_logger().log(
            KIRO_TRACING_LEVEL,
            "agentSpawn: session_id=%s, cwd=%s",
            session_id,
            cwd,
        )

    _run_transient_handler("agentSpawn", _body)


def user_prompt_submit_hook_handler() -> None:
    """CLI hook handler for ``userPromptSubmit`` — log prompt preview, pass through."""

    def _body(hook_data: dict[str, Any], get_logger) -> None:
        from mlflow.kiro_cli.tracing import KIRO_TRACING_LEVEL

        session_id = hook_data.get("session_id", "")
        cwd = hook_data.get("cwd", "")
        prompt = str(hook_data.get("prompt", ""))[:_TRANSIENT_LOG_PREVIEW]
        get_logger().log(
            KIRO_TRACING_LEVEL,
            "userPromptSubmit: session_id=%s, cwd=%s, prompt=%s",
            session_id,
            cwd,
            prompt,
        )

    _run_transient_handler("userPromptSubmit", _body)


def pre_tool_use_hook_handler() -> None:
    """CLI hook handler for ``preToolUse`` — log tool invocation, pass through.

    Never exits 2 (never blocks the tool call).
    """

    def _body(hook_data: dict[str, Any], get_logger) -> None:
        from mlflow.kiro_cli.tracing import KIRO_TRACING_LEVEL

        session_id = hook_data.get("session_id", "")
        cwd = hook_data.get("cwd", "")
        tool_name = hook_data.get("tool_name", "")
        tool_input_str = json.dumps(hook_data.get("tool_input", ""))[:_TRANSIENT_LOG_PREVIEW]
        get_logger().log(
            KIRO_TRACING_LEVEL,
            "preToolUse: session_id=%s, cwd=%s, tool_name=%s, tool_input=%s",
            session_id,
            cwd,
            tool_name,
            tool_input_str,
        )

    _run_transient_handler("preToolUse", _body)


def post_tool_use_hook_handler() -> None:
    """CLI hook handler for ``postToolUse`` — log tool response, pass through."""

    def _body(hook_data: dict[str, Any], get_logger) -> None:
        from mlflow.kiro_cli.tracing import KIRO_TRACING_LEVEL

        session_id = hook_data.get("session_id", "")
        cwd = hook_data.get("cwd", "")
        tool_name = hook_data.get("tool_name", "")
        tool_response_str = json.dumps(hook_data.get("tool_response", ""))[:_TRANSIENT_LOG_PREVIEW]
        get_logger().log(
            KIRO_TRACING_LEVEL,
            "postToolUse: session_id=%s, cwd=%s, tool_name=%s, tool_response=%s",
            session_id,
            cwd,
            tool_name,
            tool_response_str,
        )

    _run_transient_handler("postToolUse", _body)


# ============================================================================
# STOP HOOK HANDLER
# ============================================================================


def stop_hook_handler() -> None:
    """CLI hook handler for ``stop`` — read transcript and emit MLflow trace.

    Follows the design §stop Handler 7-step algorithm:

    1. Extract ``session_id`` / ``cwd`` → validate ``session_id``.
    2. Compute transcript paths under ``Path.home() / ".kiro/sessions/cli"``.
    3. Short-circuit on missing ``.jsonl``.
    4. ``setup_mlflow()``.
    5. ``process_turn(...)``.
    6. Emit ``AutologgingEvent({"flavor": "kiro_cli"})`` telemetry.
    7. Print ``{"continue": true}`` and exit 0.

    Error handling:
    - Missing ``session_id``: log ERROR, emit ``{"continue": true}``, exit 0.
    - Missing ``.jsonl``: log WARNING, emit ``{"continue": true}``, exit 0.
    - Trace-emission failure: log ERROR with stack, emit
      ``{"continue": false, "stopReason": ...}``, exit 1.
    """
    # ── Fast-path: tracing disabled ──────────────────────────────────────
    if not is_tracing_enabled():
        print(json.dumps({"continue": True}))  # noqa: T201
        return

    try:
        from mlflow.kiro_cli.tracing import (
            get_hook_response,
            get_logger,
            process_turn,
            read_hook_input,
            setup_mlflow,
        )
        from mlflow.telemetry.events import AutologgingEvent
        from mlflow.telemetry.track import _record_event

        hook_data = read_hook_input()
        session_id = hook_data.get("session_id")
        cwd = hook_data.get("cwd", "")

        # ── Step 1: validate session_id ──────────────────────────────────
        if not session_id:
            get_logger().error("stop hook: missing session_id in payload")
            print(json.dumps(get_hook_response()))  # noqa: T201
            return

        # ── Step 2: compute transcript paths ─────────────────────────────
        transcript_dir = Path.home() / ".kiro" / "sessions" / "cli"
        transcript_jsonl = transcript_dir / f"{session_id}.jsonl"
        transcript_json = transcript_dir / f"{session_id}.json"

        # ── Step 3: short-circuit on missing jsonl ───────────────────────
        if not transcript_jsonl.exists():
            get_logger().warning("stop hook: transcript not found at %s", transcript_jsonl)
            print(json.dumps(get_hook_response()))  # noqa: T201
            return

        # ── Step 4: setup MLflow ─────────────────────────────────────────
        setup_mlflow()

        # ── Step 5: process the turn ─────────────────────────────────────
        process_turn(transcript_jsonl, transcript_json, session_id, cwd)

        # ── Step 6: telemetry ────────────────────────────────────────────
        _record_event(AutologgingEvent, {"flavor": "kiro_cli"})

        # ── Step 7: success response ─────────────────────────────────────
        print(json.dumps(get_hook_response()))  # noqa: T201

    except Exception as e:
        # Try to log the error; if logging itself fails, swallow silently
        try:
            from mlflow.kiro_cli.tracing import get_logger

            get_logger().error("Error in stop hook: %s", e, exc_info=True)
        except Exception:
            pass
        print(json.dumps({"continue": False, "stopReason": str(e)}))  # noqa: T201
        sys.exit(1)
