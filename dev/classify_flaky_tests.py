"""Classify detected flaky tests and recommend how to handle each one.

Second stage of the flaky-test pipeline. `detect_flaky_tests.py` produces the
*deterministic* signal (a test that failed on one run attempt and passed on the next
attempt of the same commit). This stage adds *judgment*: given each test's failure
count and error message, an LLM classifies the root cause and recommends an action.

The allowed actions depend on the framework (`--framework`, default pytest):
  - pytest: `annotate` (safe to retry via @pytest.mark.flaky), `fix`, or `investigate`.
    A retry masks a genuine product bug, so the annotate step only acts on `annotate`.
  - jest: report-only — `fix` or `investigate` only. The MLflow frontend forbids
    `jest.retryTimes`, so a JS flake is never auto-retried; the verdict just tells a
    human where to look.

The verdict is advisory: it is written into the report so a human reviewer sees the
rationale (and, for pytest, gates the annotate step).

Auth uses an `ANTHROPIC_API_KEY` secret and a direct call to the Anthropic Messages
API. `ANTHROPIC_BASE_URL` / `FLAKY_CLASSIFIER_MODEL` may override the endpoint and
model (e.g. to route through an internal gateway).

Usage:
  python dev/classify_flaky_tests.py --in flakes.json --out classified.json
  python dev/classify_flaky_tests.py --in flakes.json --framework jest --out classified.json
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import sys
import urllib.request
from typing import Any

from detect_flaky_tests import FRAMEWORKS

DEFAULT_MODEL = os.environ.get("FLAKY_CLASSIFIER_MODEL", "claude-sonnet-4-6")
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")

# pytest's pipeline can auto-retry a flake (via @pytest.mark.flaky), so its classifier
# decides annotate-vs-fix. The JS pipeline cannot: the frontend ESLint config forbids
# `jest.retryTimes` ("Tests must pass without retries. If a test is flaky, fix it"), so
# the jest classifier is report-only — every flake is a fix/investigate for a human, and
# `annotate` is not an option it can pick.
_PYTEST_PROMPT = """\
You are triaging a flaky test in the MLflow CI suite. A "flake" here is a test that \
FAILED on one CI run attempt and PASSED on a re-run of the exact same commit — so the \
code did not change between the two outcomes.

Your job: decide whether this test should be annotated with `@pytest.mark.flaky` \
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
- `action`: one of "annotate" (safe to retry via @pytest.mark.flaky), "fix" (likely a \
real bug — do not retry, a human should fix it), "investigate" (not enough signal to \
decide).
- `attempts`: suggested retry count (2 or 3) if action is "annotate", else null.
- `confidence`: "high", "medium", or "low".
- `rationale`: one or two sentences explaining the decision, for the human reviewer.

Weigh the failure count: a test that flaked repeatedly across many commits is more \
likely a genuine flake safe to retry; a single occurrence with a logic-bug-shaped error \
should lean toward "investigate" or "fix"."""

_JEST_PROMPT = """\
You are triaging a flaky test in the MLflow JS CI suite. A "flake" here is a test that \
FAILED on one CI run attempt and PASSED on a re-run of the exact same commit — so the \
code did not change between the two outcomes.

Retrying is NOT an option: the MLflow frontend forbids `jest.retryTimes` as a matter of \
policy ("Tests must pass without retries. If a test is flaky, fix it instead of adding \
retries."). Your job is therefore purely diagnostic — classify the likely root cause \
and point a human at the fix. Every flake here must be FIXED or INVESTIGATED, never \
masked with a retry.

## Test
{nodeid}

## How often it flaked (distinct commits, in the window)
{count}

## Representative error message
{error}

## Instructions
Return a JSON object:
- `category`: one of "timeout", "resource-contention", "network", "test-harness-race", \
"product-race", "logic-bug", "unknown". For JS flakes, common causes are unawaited \
state updates (missing `await`/`waitFor`, React "not wrapped in act(...)" warnings), \
fake-timer/real-timer races, and unmocked network — most map to "test-harness-race".
- `action`: one of "fix" (a concrete root cause is identifiable and a human should fix \
the test/product) or "investigate" (not enough signal to pinpoint the cause yet).
- `confidence`: "high", "medium", or "low".
- `rationale`: one or two sentences explaining the decision, for the human reviewer — \
lead with the most likely root cause and the fix direction."""


def _schema(actions: list[str], *, with_attempts: bool) -> dict[str, Any]:
    """Verdict JSON schema. `attempts` only exists for frameworks that can retry."""
    properties: dict[str, Any] = {
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
        "action": {"type": "string", "enum": actions},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "rationale": {"type": "string"},
    }
    if with_attempts:
        properties["attempts"] = {"type": ["integer", "null"]}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


@dataclasses.dataclass(frozen=True)
class FrameworkClassifier:
    """Per-framework prompt + verdict schema. pytest can `annotate` (auto-retry); jest is
    report-only (fix/investigate) because retries are banned in the JS suite.
    """

    prompt: str
    schema: dict[str, Any]


CLASSIFIERS: dict[str, FrameworkClassifier] = {
    "pytest": FrameworkClassifier(
        _PYTEST_PROMPT, _schema(["annotate", "fix", "investigate"], with_attempts=True)
    ),
    "jest": FrameworkClassifier(_JEST_PROMPT, _schema(["fix", "investigate"], with_attempts=False)),
}

# Keyed by the same framework names as detect's FRAMEWORKS registry: every framework the
# detector can mine must have a classifier here, or `--framework X` would pass argparse in
# detect and then KeyError here. The assert enforces that at import so the two registries
# can't silently drift out of sync.
assert CLASSIFIERS.keys() == FRAMEWORKS.keys(), (
    "CLASSIFIERS and detect_flaky_tests.FRAMEWORKS must cover the same frameworks; "
    f"got {sorted(CLASSIFIERS)} vs {sorted(FRAMEWORKS)}"
)


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


def classify(
    nodeid: str, count: int, error: str, classifier: FrameworkClassifier
) -> dict[str, Any]:
    prompt = classifier.prompt.format(nodeid=nodeid, count=count, error=error or "(none)")
    request_body = {
        "model": DEFAULT_MODEL,
        "max_tokens": 1024,
        # temperature=0 for the most deterministic verdict we can get: the same flake
        # should classify the same way run-to-run so the report/PR stays stable.
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        # Constrain the reply to the framework's schema so the model emits exactly the
        # fields the downstream stages read — no prose to parse, and an off-schema reply
        # (e.g. jest picking the disallowed `annotate`) is rejected by the API.
        "output_config": {"format": {"type": "json_schema", "schema": classifier.schema}},
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


def render_summary(results: list[dict[str, Any]], framework: str) -> str:
    """Markdown report of the classified flakes, ranked by flake count.

    The detector's own summary (flakes.md) carries no verdicts, so this is what the
    weekly issue should publish: each flake's LLM action, category, confidence, and
    rationale — the fix guidance a human actually acts on.
    """
    lines = [
        f"# Classified flaky {framework} tests",
        "",
        f"**{len(results)}** distinct flaky tests/shards, ranked by flake count. Each "
        "carries an LLM triage verdict (root cause + recommended action).",
        "",
    ]
    for r in results:
        v = r["verdict"]
        label = r["test"] or f"{r['shard']} (whole shard — no test line in log)"
        lines.append(f"### `{label}`")
        lines.append(
            f"- **action: {v['action']}** · category: {v['category']} · "
            f"confidence: {v['confidence']} · flaked {r['count']}×"
        )
        if attempts := v.get("attempts"):
            lines.append(f"- suggested retry attempts: {attempts}")
        lines.append(f"- shard: `{r['shard']}`")
        if r.get("error"):
            lines.append(f"- error: `{r['error']}`")
        lines.append(f"- _rationale:_ {v['rationale']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="infile", required=True, help="flakes.json from the detector")
    p.add_argument("--out", help="Write classified JSON here")
    p.add_argument(
        "--summary", help="Write a Markdown report (with verdicts) here for the issue body."
    )
    p.add_argument(
        "--framework",
        default="pytest",
        choices=sorted(CLASSIFIERS),
        help="Test framework the flakes came from (selects the prompt and allowed "
        "verdicts; jest is report-only since retries are banned). Default: pytest.",
    )
    args = p.parse_args()

    classifier = CLASSIFIERS[args.framework]

    with open(args.infile) as f:
        flakes = json.load(f)

    results = []
    for t in _aggregate(flakes):
        # Only test-level entries carry a nodeid; shard-level ones can't be classified.
        if not t["test"]:
            verdict: dict[str, Any] = {
                "category": "unknown",
                "action": "investigate",
                "confidence": "low",
                "rationale": "No test-level nodeid recovered (shard/infra flake).",
            }
            # Keep the fallback conformant with the framework's verdict schema: pytest
            # lists `attempts` as required, so include it (null) there; jest's schema
            # omits it entirely.
            if "attempts" in classifier.schema["properties"]:
                verdict["attempts"] = None
        else:
            verdict = classify(t["test"], t["count"], t["error"] or "", classifier)
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
    if args.summary:
        with open(args.summary, "w") as f:
            f.write(render_summary(results, args.framework) + "\n")


if __name__ == "__main__":
    main()
