"""Legacy compatibility helpers for the retired Python Claude hook runtime."""

import json
import sys

from mlflow.claude_code.tracing import get_hook_response


def stop_hook_handler() -> None:
    """No-op shim for repositories still wired to the old Python hook."""
    print(json.dumps(get_hook_response()))  # noqa: T201
    print(  # noqa: T201
        "MLflow Claude tracing has moved to the marketplace plugin runtime. "
        "Run `mlflow autolog claude` again to migrate this project.",
        file=sys.stderr,
    )
