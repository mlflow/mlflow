"""Detect flaky tests from user-triggered CI re-runs.

Ground-truth flake signal: a job that **failed on one run attempt and passed on the
next attempt of the same commit** flaked by definition (same code, different outcome),
and a human already judged it worth re-running by hitting "Re-run failed jobs".

This script mines the GitHub Actions API for those fail->pass jobs over a time window,
then reads the **failing attempt's job logs** to pin down the exact test(s) and error
message(s). Logs are used rather than JUnit XML artifacts on purpose: GitHub artifacts
are not run-attempt-scoped and are cleared when a re-run starts, so a prior attempt's
artifact can never be fetched — but the per-attempt jobs/logs endpoints DO retain
history, and pytest's failure lines carry real, runnable nodeids.

No CI or test-suite changes are required: this is pure read-only mining of history.

Output:
  - A JSON report (``--out``) of confirmed flakes: shard, test nodeid, error snippet, runs.
  - A Markdown summary (``--summary``) suitable for a GitHub step summary / issue body.

Usage:
  python dev/detect_flaky_tests.py --repo mlflow/mlflow --since 2026-07-03 \
      --out flakes.json --summary flakes.md
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

WORKFLOW_NAME = "MLflow tests"

# A pytest failure/error line names a runnable nodeid followed by " - <message>".
# The MLflow conftest prefixes some lines with "FAILED | MEM ... DISK ..."; the nodeid
# pattern anchors regardless of that prefix. We only trust lines that also carry a
# FAILED/ERROR token to avoid matching nodeids quoted in tracebacks or rerun commands.
_NODEID_RE = re.compile(r"(tests/[^\s:]+\.py::[^\s]+)\s+-\s+(.+?)\s*$")
_OUTCOME_RE = re.compile(r"\b(FAILED|ERROR)\b")
_TIMESTAMP_RE = re.compile(r"^[0-9T:.Z\-]+\s+")
# Actions logs carry both an ISO timestamp prefix and ANSI SGR color codes (pytest
# colorizes FAILED / the test name); both must be stripped before the nodeid regex.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclasses.dataclass
class FlakyTest:
    shard: str  # e.g. "python (4)"
    test: str | None  # nodeid, or None if no test line was recoverable (shard-level only)
    error: str | None  # first-line failure message
    run_id: int
    sha: str
    attempt_fail: int
    attempt_pass: int
    event: str

    def key(self) -> str:
        # Dedup identity: prefer test-level, fall back to shard-level.
        return self.test or f"shard:{self.shard}"


def gh_api(path: str, paginate: bool = False) -> str | None:
    cmd = ["gh", "api"] + (["--paginate"] if paginate else []) + [path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ! gh api {path} failed: {proc.stderr.strip()[:200]}", file=sys.stderr)
        return None
    return proc.stdout


def gh_api_objects(path: str, paginate: bool = False) -> list[dict[str, Any]]:
    """Parse gh output, tolerating --paginate's concatenated JSON objects."""
    raw = gh_api(path, paginate)
    if not raw:
        return []
    decoder = json.JSONDecoder()
    objs: list[dict[str, Any]] = []
    idx = 0
    s = raw.strip()
    while idx < len(s):
        while idx < len(s) and s[idx].isspace():
            idx += 1
        if idx >= len(s):
            break
        obj, idx = decoder.raw_decode(s, idx)
        objs.append(obj)
    return objs


def get_workflow_id(repo: str) -> str | None:
    for page in gh_api_objects(f"repos/{repo}/actions/workflows", paginate=True):
        for wf in page.get("workflows", []):
            if wf.get("name") == WORKFLOW_NAME:
                return str(wf["id"])
    return None


def list_multiattempt_runs(repo: str, wf_id: str, since: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    q = f"repos/{repo}/actions/workflows/{wf_id}/runs?per_page=100&created=%3E%3D{since}"
    for page in gh_api_objects(q, paginate=True):
        runs.extend(page.get("workflow_runs", []))
    return [r for r in runs if r.get("run_attempt", 1) > 1]


def attempt_jobs(repo: str, run_id: int, attempt: int) -> list[dict[str, Any]]:
    """All jobs for one run attempt (paginated: master has >20 jobs)."""
    jobs: list[dict[str, Any]] = []
    path = f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100"
    for page in gh_api_objects(path, paginate=True):
        jobs.extend(page.get("jobs", []))
    return jobs


def failing_tests_from_log(repo: str, job_id: int) -> dict[str, str]:
    """{nodeid: first-line error} parsed from a job's log. Empty if none recoverable."""
    log = gh_api(f"repos/{repo}/actions/jobs/{job_id}/logs")
    if not log:
        return {}
    failures: dict[str, str] = {}
    for raw in log.splitlines():
        line = _ANSI_RE.sub("", raw)
        line = _TIMESTAMP_RE.sub("", line)
        if not _OUTCOME_RE.search(line):
            continue
        if m := _NODEID_RE.search(line):
            # First occurrence wins; keeps the concise summary-line message.
            failures.setdefault(m.group(1), m.group(2)[:300])
    return failures


def detect(repo: str, since: str) -> list[FlakyTest]:
    wf_id = get_workflow_id(repo)
    if not wf_id:
        print(f"Could not resolve workflow '{WORKFLOW_NAME}' in {repo}", file=sys.stderr)
        return []
    runs = list_multiattempt_runs(repo, wf_id, since)
    print(f"Examining {len(runs)} multi-attempt runs since {since} ...", file=sys.stderr)

    flakes: list[FlakyTest] = []
    for r in runs:
        run_id = r["id"]
        attempts = r["run_attempt"]
        sha = r["head_sha"][:8]
        event = r.get("event", "?")
        for a in range(1, attempts):
            b = a + 1
            jobs_a = attempt_jobs(repo, run_id, a)
            concl_b = {j["name"]: j.get("conclusion") for j in attempt_jobs(repo, run_id, b)}
            # A job that FAILED on attempt a and SUCCEEDED on attempt b is a ground-truth flake.
            for job in jobs_a:
                name = job["name"]
                if job.get("conclusion") != "failure" or concl_b.get(name) != "success":
                    continue
                if failing := failing_tests_from_log(repo, job["id"]):
                    for nodeid, err in failing.items():
                        flakes.append(FlakyTest(name, nodeid, err, run_id, sha, a, b, event))
                else:
                    # Job flaked but no test line was recoverable (infra flake, setup
                    # failure, or a log format we don't parse): report at shard level.
                    flakes.append(FlakyTest(name, None, None, run_id, sha, a, b, event))
    return flakes


def render_summary(flakes: list[FlakyTest], since: str) -> str:
    by_key: dict[str, list[FlakyTest]] = collections.defaultdict(list)
    for f in flakes:
        by_key[f.key()].append(f)
    lines = [
        f"# Flaky test report (since {since})",
        "",
        f"Confirmed fail→pass events: **{len(flakes)}** across "
        f"**{len(by_key)}** distinct tests/shards.",
        "",
        "A _flake_ here is a job that failed on one run attempt and passed on the next "
        "attempt of the same commit. Test-level entries are parsed from the failing "
        "attempt's logs; shard-level entries are jobs whose failing test could not be "
        "recovered from the log (often an infra/setup flake).",
        "",
        "Ranked by frequency:",
        "",
    ]
    ranked = sorted(by_key.items(), key=lambda kv: len(kv[1]), reverse=True)
    for _key, evs in ranked:
        e0 = evs[0]
        label = f"`{e0.test}`" if e0.test else f"`{e0.shard}` (whole shard — no test line in log)"
        lines.append(f"- **{len(evs)}×** {label}")
        if e0.error:
            lines.append(f"  - error: `{e0.error}`")
        shas = ", ".join(sorted({e.sha for e in evs}))
        lines.append(f"  - shard: `{e0.shard}` · commits: {shas}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="mlflow/mlflow")
    p.add_argument("--since", help="ISO date (default: 14 days ago)")
    p.add_argument("--out", help="Write JSON report here")
    p.add_argument("--summary", help="Write Markdown summary here")
    args = p.parse_args()

    since = args.since or (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
    flakes = detect(args.repo, since)

    summary = render_summary(flakes, since)
    print(summary)
    if args.summary:
        with open(args.summary, "w") as f:
            f.write(summary + "\n")
    if args.out:
        with open(args.out, "w") as f:
            json.dump([dataclasses.asdict(x) for x in flakes], f, indent=2)


if __name__ == "__main__":
    main()
