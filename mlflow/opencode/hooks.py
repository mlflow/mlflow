"""CLI-invocable hook handlers for Opencode tracing."""

import json
import sys
from typing import Any

from mlflow.opencode.tracing import (
    get_logger,
    is_tracing_enabled,
    process_session,
    setup_mlflow,
)


def read_hook_input() -> dict[str, Any]:
    try:
        input_data = sys.stdin.read()
        if not input_data.strip():
            return {}
        return json.loads(input_data)
    except json.JSONDecodeError as e:
        get_logger().error("Failed to parse hook input: %s", e)
        return {}


def session_completed_handler() -> None:
    if not is_tracing_enabled():
        return

    try:
        setup_mlflow()
        hook_data = read_hook_input()

        session_id = hook_data.get("sessionID")
        session_info = hook_data.get("session", {})
        messages = hook_data.get("messages", [])

        if not session_id:
            get_logger().warning("No session ID in hook data")
            return

        process_session(session_id, session_info, messages)

    except Exception as e:
        get_logger().error("Error in session_completed hook: %s", e, exc_info=True)
        sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python -m mlflow.opencode.hooks <hook_name>\n")
        sys.exit(1)

    hook_name = sys.argv[1]

    if hook_name == "session_completed":
        session_completed_handler()
    else:
        sys.stderr.write(f"Unknown hook: {hook_name}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
