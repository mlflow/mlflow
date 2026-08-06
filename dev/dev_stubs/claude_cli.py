"""Credential-free stub `claude` CLI for reviewing provider-gated Assistant UI.

The MLflow Assistant's "Claude Code" provider only reveals its chat panel after
an auth probe succeeds: it shells out to
``claude -p hi --max-turns 1 --output-format json`` and unlocks the UI when that
exits 0 (see ``mlflow/assistant/providers/claude_code.py``). When the dev server
runs without ``ANTHROPIC_API_KEY`` -- the CI ui-review bot (to keep secrets out
of PR backend code), or a local dev who hasn't installed/authed the real CLI --
the provider reports "not authenticated" and the panel never renders, leaving
provider-gated UI (e.g. restore-chat-on-reload) unreviewable.

This script stands in for ``claude``. ``run_dev_server.py --stub-providers claude``
wraps it in a ``claude`` shim and prepends its directory to the dev server's PATH
(only the dev server process and its children; a real ``claude`` elsewhere on the
machine -- e.g. the ui-review bot's own review step -- is untouched).

It never contacts Anthropic, so it costs nothing, needs no credentials, and is
deterministic:

- ``--output-format json`` (the auth probe): print one success result and exit 0.
- ``--output-format stream-json`` (a live chat turn): emit canned stream-json
  events -- an assistant text message plus a closing result -- so the chat panel
  can be exercised end to end and its messages persisted for restore-on-reload.

The reply is clearly labeled synthetic so reviewers don't mistake it for a real
model response. Real provider behavior stays covered by unit/integration tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

STUB_REPLY = (
    "This is a synthetic reply from the MLflow dev stub Claude CLI. The real "
    "Claude Code provider is replaced so the Assistant chat panel can be reviewed "
    "without credentials or LLM calls. No model was invoked to produce this message."
)

STUB_MODEL = "mlflow-dev-stub"


def _emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _result_event(session_id: str) -> dict[str, Any]:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": STUB_REPLY,
        "session_id": session_id,
        "duration_ms": 1,
        "num_turns": 1,
        "total_cost_usd": 0.0,
        "usage": {
            "input_tokens": 8,
            "output_tokens": 24,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def main(argv: list[str]) -> int:
    # The real `claude` CLI takes many flags; parse only the few the stub needs and
    # let parse_known_args drop the rest. add_help/allow_abbrev are off so `-p`,
    # `--verbose`, etc. fall through to the ignored extras rather than being matched.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--output-format", default="text")
    parser.add_argument("--resume", default=None)
    args, _ = parser.parse_known_args(argv)

    if args.version:
        print(f"0.0.0 ({STUB_MODEL})")
        return 0

    # Reuse the resume id so a continued conversation keeps a stable session.
    session_id = args.resume or f"mlflow-dev-stub-{uuid.uuid4().hex[:12]}"

    if args.output_format == "stream-json":
        _emit({
            "type": "system",
            "subtype": "init",
            "session_id": session_id,
            "model": STUB_MODEL,
            "tools": [],
        })
        _emit({
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": STUB_REPLY}]},
        })
        _emit(_result_event(session_id))
        return 0

    # Auth probe (`--output-format json`) and any other invocation: the provider
    # only checks the exit code, but emit a valid result object for good measure.
    _emit(_result_event(session_id))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
