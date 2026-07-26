"""Classify detected flaky tests and decide which to annotate @pytest.mark.flaky.

Second stage of the flaky-test pipeline. `detect_flaky_tests.py` produces the
*deterministic* signal (a test that failed on one run attempt and passed on the next
attempt of the same commit). This stage adds *judgment*: given each test's failure
count and error message, an LLM classifies the root cause and recommends whether to
retry the test (annotate @pytest.mark.flaky) or leave it for a human to fix — because
a retry masks a genuine product bug and should not be applied blindly.

The verdict is advisory: it is written into the report / PR body so a human reviewer
sees the rationale, and the annotate step only ever acts on `action == "annotate"`.

Auth mirrors .github/workflows/triage.py: an `ANTHROPIC_API_KEY` secret and a direct
call to the Anthropic Messages API. `ANTHROPIC_BASE_URL` / `FLAKY_CLASSIFIER_MODEL`
may override the endpoint and model (e.g. to route through an internal gateway).

Usage:
  python dev/classify_flaky_tests.py --in flakes.json --out classified.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MODEL = os.environ.get("FLAKY_CLASSIFIER_MODEL", "claude-sonnet-4-6")
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")

# Network tuning, overridable via env so a slow/loaded internal gateway can be given
# more headroom without a code change.
REQUEST_TIMEOUT = float(os.environ.get("FLAKY_CLASSIFIER_TIMEOUT", "60"))
MAX_RETRIES = int(os.environ.get("FLAKY_CLASSIFIER_MAX_RETRIES", "3"))
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

PROMPT_TEMPLATE = """\
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

# Verdict used when a test can't be classified at all — either it has no test-level
# nodeid (shard/infra flake) or the API call exhausted its retries. Keeping this in one
# place ensures the report always has a uniform shape for downstream consumers.
def _fallback_verdict(rationale: str) -> dict[str, Any]:
    return {
        "category": "unknown",
        "action": "investigate",
        "attempts": None,
        "confidence": "low",
        "rationale": rationale,
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


def _post_with_retries(req: urllib.request.Request) -> dict[str, Any]:
    """POST with a hard socket timeout and bounded retries on transient failures.

    Retries on rate limits and 5xx (honoring Retry-After when the server sends one) and
    on connection-level failures/timeouts, which urllib raises as URLError/TimeoutError
    rather than HTTPError. Anything else (4xx auth/validation errors) fails immediately
    since retrying won't help.
    """
    last_err: BaseException | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code in RETRYABLE_STATUSES and attempt < MAX_RETRIES:
                delay = float(e.headers.get("Retry-After", 2 ** attempt))
                print(
                    f"  API {e.code} on attempt {attempt}/{MAX_RETRIES}, "
                    f"retrying in {delay:.0f}s: {body[:300]}",
                    file=sys.stderr,
                )
                time.sleep(delay)
                last_err = e
                continue
            print(f"  API Error {e.code}: {body[:500]}", file=sys.stderr)
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            # Bare TimeoutError is what urlopen raises when `timeout=` is exceeded on a
            # socket read; URLError covers DNS/connection failures.
            if attempt < MAX_RETRIES:
                delay = 2 ** attempt
                print(
                    f"  Network error on attempt {attempt}/{MAX_RETRIES} "
                    f"({e}), retrying in {delay}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
                last_err = e
                continue
            print(f"  Network error, retries exhausted: {e}", file=sys.stderr)
            raise
    raise RuntimeError("Exhausted retries") from last_err


def _extract_verdict(response: dict[str, Any], nodeid: str) -> dict[str, Any]:
    """Pull the schema-constrained verdict out of the Messages API envelope.

    Raises a descriptive RuntimeError instead of a bare KeyError/IndexError if the
    response doesn't have the shape we expect, so failures are debuggable from CI logs
    without reproducing locally.
    """
    if "error" in response:
        raise RuntimeError(
            f"API returned an error payload for {nodeid!r}: {response['error']}"
        )
    content = response.get("content")
    if not content or not isinstance(content, list):
        raise RuntimeError(
            f"Unexpected response shape for {nodeid!r} (no `content` list): "
            f"{json.dumps(response)[:500]}"
        )
    text = content[0].get("text")
    if text is None:
        raise RuntimeError(
            f"Unexpected response shape for {nodeid!r} (first content block has no "
            f"`text`, got keys {list(content[0].keys())}): {json.dumps(response)[:500]}"
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Verdict text for {nodeid!r} was not valid JSON: {text[:500]}"
        ) from e


def classify(nodeid: str, count: int, error: str) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(nodeid=nodeid, count=count, error=error or "(none)")
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
    response = _post_with_retries(req)
    return _extract_verdict(response, nodeid)


def _most_common_error(evs: list[dict[str, Any]]) -> str | None:
    """Pick the error message that recurred most often across a test's flake events.

    A test can flake for more than one reason (e.g. mostly a timeout, but once a real
    assertion failure), and the first event isn't necessarily representative. Using the
    mode gives the classifier the failure mode that's actually characteristic of the
    flake, while ties fall back to encounter order so the result stays deterministic.
    Events with no error message are excluded from the vote.
    """
    counts = collections.Counter(e["error"] for e in evs if e.get("error"))
    if not counts:
        return None
    top_count = max(counts.values())
    for e in evs:
        err = e.get("error")
        if err and counts[err] == top_count:
            return err
    return None  # unreachable, but keeps the type checker happy


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
            "error": _most_common_error(evs),
        })
    return sorted(tests, key=lambda t: t["count"], reverse=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="infile", required=True, help="flakes.json from the detector")
    p.add_argument("--out", help="Write classified JSON here")
    args = p.parse_args()

    with open(args.infile) as f:
        flakes = json.load(f)

    results = []
    failures = 0
    for t in _aggregate(flakes):
        # Only test-level entries can be annotated; shard-level ones lack a nodeid.
        if not t["test"]:
            verdict = _fallback_verdict("No test-level nodeid recovered (shard/infra flake).")
        else:
            try:
                verdict = classify(t["test"], t["count"], t["error"] or "")
            except Exception as e:
                # Fault isolation: one test that exhausts its retries or gets a
                # malformed response shouldn't take down classification for every other
                # test in the run. Record it as low-confidence "investigate" so a human
                # sees it flagged in the report rather than the whole step aborting and
                # discarding everything classified so far.
                print(f"  giving up on {t['test']!r}: {e}", file=sys.stderr)
                verdict = _fallback_verdict(f"Classification failed after retries: {e}")
                failures += 1
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

    if failures:
        print(
            f"\n{failures} test(s) could not be classified after retries; "
            f"they were recorded as 'investigate' with low confidence. See stderr above.",
            file=sys.stderr,
        )
        # Non-zero exit so CI surfaces this as a degraded (not silently swallowed) run,
        # while `--out` still has every test that *did* classify successfully.
        sys.exit(1)


if __name__ == "__main__":
    main()
