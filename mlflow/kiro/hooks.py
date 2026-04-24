"""Hook handler for the Kiro CLI integration with MLflow."""

import json
import sys
from pathlib import Path

from mlflow.kiro.config import KIRO_ENV_FILE, KIRO_HOOKS_DIR
from mlflow.kiro.tracing import (
    KIRO_TRACING_LEVEL,
    get_hook_response,
    get_logger,
    is_tracing_enabled,
    process_session,
    read_hook_input,
    setup_mlflow,
)


def stop_hook_handler() -> None:
    """CLI hook handler invoked by Kiro on the *Agent Stop* event.

    Reads the session JSON from stdin, creates an MLflow trace, and writes the
    Kiro hook response to stdout.
    """
    if not is_tracing_enabled():
        print(json.dumps(get_hook_response()))  # noqa: T201
        return

    try:
        session_data = read_hook_input()
        session_id = session_data.get("session_id", "<unknown>")

        setup_mlflow()

        get_logger().log(
            KIRO_TRACING_LEVEL,
            "Stop hook invoked for session: %s",
            session_id,
        )

        trace = process_session(session_data)

        if trace is not None:
            response = get_hook_response()
        else:
            response = get_hook_response(
                error=(
                    "Failed to process Kiro session. "
                    "Check .kiro/mlflow/kiro_tracing.log for details."
                )
            )

        print(json.dumps(response))  # noqa: T201

    except Exception as exc:
        get_logger().error("Error in Kiro stop hook: %s", exc, exc_info=True)
        print(json.dumps(get_hook_response(error=str(exc))))  # noqa: T201
        sys.exit(1)
