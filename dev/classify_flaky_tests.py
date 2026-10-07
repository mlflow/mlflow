"""Classify detected flaky tests and decide which to annotate @pytest.mark.flaky.

Second stage of the flaky-test pipeline. `detect_flaky_tests.py` produces the
*deterministic* signal (a test that failed on one run attempt and passed on the next
attempt of the same commit). This stage adds *judgment*: given each test's failure
count and error message, an LLM classifies the root cause and recommends whether to
retry the test (annotate @pytest.mark.flaky) or leave it for a human to fix — because
a retry masks a genuine product bug and should not be applied blindly.

The verdict is advisory: it is written into the report / PR body so a human reviewer
sees the rationale, and the annotate step only ever acts on `action == "annotate"`.

Auth uses an `ANTHROPIC_API_KEY` secret and a direct call to the Anthropic Messages
API. `ANTHROPIC_BASE_URL` / `FLAKY_CLASSIFIER_MODEL` may override the endpoint and
model (e.g. to route through an internal gateway).

Usage:
  python dev/classify_flaky_tests.py --in flakes.json --out classified.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import urllib.request
from typing import Any

from detect_flaky_tests import FRAMEWORKS

DEFAULT_MODEL = os.environ.get("FLAKY_CLASSIFIER_MODEL", "claude-sonnet-4-6")
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")

# The prompt is identical across frameworks except for how a retry is expressed;
# `{retry_mechanism}` is filled per framework (see RETRY_MECHANISMS below) so the
# classifier's `annotate` verdict names the mechanism a human would actually apply.
PROMPT_TEMPLATE = """\
You are triaging a flaky test in the MLflow CI suite. A "flake" here is a test that \
FAILED on one CI run attempt and PASSED on a re-run of the exact same commit — so the \
code did not change between the two outcomes.

Your job: decide whether this test should be annotated with `{retry_mechanism}` \
(which makes CI automatically retry it), or whether the flake likely reflects a real \
bug that a human should FIX instead.

Guiding principle: a retry annotation is appropriate for NON-DETERMINISTIC \
infrastructure/timing issues that are safe to just re-run (network hiccups, load-\
induced timeouts, port/resource contention, background-thread races in the test \
harness). A retry is NOT appropriate — and would dangerously mask a defect — when the \
error suggests a genuine logic bug, a real product race condition, or a deterministic \
failure that only "passed on retry" by luck.

## Test
{nodeid}

## How often it flaked (distinct commits, in the window)
{count}

## Representative error message
{error}

## Instructions
Return a JSON object:
- `category`: one of "timeout", "resource-contention", "network", "test-harness-race", \
"product-race", "logic-bug", "unknown".
- `action`: one of "annotate" (safe to retry via {retry_mechanism}), "fix" (likely a \
real bug — do not retry, a human should fix it), "investigate" (not enough signal to \
decide).
- `attempts`: suggested retry count (2 or 3) if action is "annotate", else null.
- `confidence`: "high", "medium", or "low".
- `rationale`: one or two sentences explaining the decision, for the human reviewer.

Weigh the failure count: a test that flaked repeatedly across many commits is more \
likely a genuine flake safe to retry; a single occurrence with a logic-bug-shaped error \
should lean toward "investigate" or "fix"."""

# How each framework expresses an automatic retry, injected into the prompt above.
# Keyed by the same framework names as detect's FRAMEWORKS registry: every framework the
# detector can mine must have a retry mechanism here, or `--framework X` would pass
# argparse in detect and then KeyError here. The assert below enforces that at import so
# the two registries can't silently drift out of sync.
RETRY_MECHANISMS: dict[str, str] = {
    "pytest": "@pytest.mark.flaky",
}

assert RETRY_MECHANISMS.keys() == FRAMEWORKS.keys(), (
    "RETRY_MECHANISMS and detect_flaky_tests.FRAMEWORKS must cover the same frameworks; "
    f"got {sorted(RETRY_MECHANISMS)} vs {sorted(FRAMEWORKS)}"
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [
                "timeout",
                "resource-contention",
                "network",
                "test-harness-race",
                "product-race",
                "logic-bug",
                "unknown",
            ],
        },
        "action": {"type": "string", "enum": ["annotate", "fix", "investigate"]},
        "attempts": {"type": ["integer", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "rationale": {"type": "string"},
    },
    "required": ["category", "action", "attempts", "confidence", "rationale"],
    "additionalProperties": False,
}


def _auth_headers() -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    # Two auth schemes: a bearer token (used by internal gateways fronting the API) takes
    # precedence, falling back to the standard `x-api-key` direct-to-Anthropic header.
    if token := os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    else:
        headers["x-api-key"] = os.environ["ANTHROPIC_API_KEY"]
    return headers


def classify(nodeid: str, count: int, error: str, retry_mechanism: str) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(
        nodeid=nodeid, count=count, error=error or "(none)", retry_mechanism=retry_mechanism
    )
    request_body = {
        "model": DEFAULT_MODEL,
        "max_tokens": 1024,
        # temperature=0 for the most deterministic verdict we can get: the same flake
        # should classify the same way run-to-run so the report/PR stays stable.
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        # Constrain the reply to `_SCHEMA` so the model emits exactly the fields the
        # downstream stages read — no prose to parse, and an off-schema reply is rejected
        # by the API rather than reaching us.
        "output_config": {"format": {"type": "json_schema", "schema": _SCHEMA}},
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/messages",
        data=json.dumps(request_body).encode(),
        headers=_auth_headers(),
    )
    try:
        with urllib.request.urlopen(req) as resp:
            response = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        # Surface the API error body (rate limit, auth, bad request) before re-raising;
        # there is no per-test fallback, so a failed call aborts the classify step.
        print(f"API Error {e.code}: {e.read().decode()}", file=sys.stderr)
        raise
    # Two-level decode: the outer envelope is the Messages API response; the schema-
    # constrained verdict is itself a JSON string in the first content block's `text`.
    verdict: dict[str, Any] = json.loads(response["content"][0]["text"])
    return verdict


def _aggregate(flakes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-event flake records into one entry per test, with a failure count."""
    by_test: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for f in flakes:
        by_test[f.get("test") or f"shard:{f['shard']}"].append(f)
    tests = []
    for key, evs in by_test.items():
        e0 = evs[0]
        tests.append({
            "test": e0.get("test"),
            "shard": e0["shard"],
            "count": len(evs),
            "error": e0.get("error"),
        })
    return sorted(tests, key=lambda t: t["count"], reverse=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="infile", required=True, help="flakes.json from the detector")
    p.add_argument("--out", help="Write classified JSON here")
    p.add_argument(
        "--framework",
        default="pytest",
        choices=sorted(RETRY_MECHANISMS),
        help="Test framework the flakes came from (selects the retry mechanism named in "
        "the prompt). Default: pytest.",
    )
    args = p.parse_args()

    retry_mechanism = RETRY_MECHANISMS[args.framework]

    with open(args.infile) as f:
        flakes = json.load(f)

    results = []
    for t in _aggregate(flakes):
        # Only test-level entries can be annotated; shard-level ones lack a nodeid.
        if not t["test"]:
            verdict = {
                "category": "unknown",
                "action": "investigate",
                "attempts": None,
                "confidence": "low",
                "rationale": "No test-level nodeid recovered (shard/infra flake).",
            }
        else:
            verdict = classify(t["test"], t["count"], t["error"] or "", retry_mechanism)
        results.append({**t, "verdict": verdict})
        label = t["test"] or t["shard"]
        v = verdict
        print(
            f"[{v['action']:>11}] {label}  ({t['count']}×, {v['category']}, {v['confidence']})\n"
            f"              {v['rationale']}"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
