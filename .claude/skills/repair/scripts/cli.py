import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

FAILED_CONCLUSIONS = {"FAILURE", "TIMED_OUT", "STARTUP_FAILURE"}
EXCLUDED_WORKFLOWS = {"rerun.yml", "repair.yml"}
PR_BATCH_SIZE = 20
CHECK_PAGE_SIZE = 20
CONNECTION_BATCH_SIZE = 20
REPO = "mlflow/mlflow"
PR_BUDGET = 5
HISTORY_MARKER = "<!-- repair-history -->"
APP = "mlflow-app[bot]"
APP_ID = 191841109
# Added test lines that skip or expect failure instead of fixing the code.
TEST_WEAKENING_RE = re.compile(
    r"pytest\.mark\.(skip|skipif|xfail)\b|pytest\.(skip|xfail|importorskip)\(|unittest\.skip"
    r"|unittest\.expectedFailure|\.skipTest\(|\b(it|test|describe)\.(skip|only|todo|fixme)\("
    r"|\b[xf](it|test|describe)\("
)
TEST_DEFINITION_RE = re.compile(
    r"^\s*(?:async\s+)?def\s+(test_\w+)|^\s*(?:it|test)\(\s*([\"'`])(.+?)\2"
)
TEST_FILE_RE = re.compile(r"(^|/)tests?/|(^|/)test_[^/]*\.py$|\.test\.[jt]sx?$")

COMMENT_FIELDS = """
  id
  body
  author { __typename }
"""
COMMENTS_FIELDS = f"""
  nodes {{ {COMMENT_FIELDS} }}
  pageInfo {{ hasNextPage endCursor }}
"""

CHECK_FIELDS = """
  __typename
  ... on CheckRun {
    id
    databaseId
    name
    status
    conclusion
    detailsUrl
    permalink
    checkSuite {
      status
      conclusion
      app { slug }
      workflowRun {
        databaseId
        createdAt
        event
        runAttempt
        workflow { resourcePath }
      }
    }
  }
"""
CONTEXT_FIELDS = f"""
  nodes {{ {CHECK_FIELDS} }}
  pageInfo {{ hasNextPage endCursor }}
"""
PR_FIELDS = f"""
  id
  number
  headRefName
  headRefOid
  baseRefName
  baseRefOid
  state
  isDraft
  mergeable
  isCrossRepository
  author {{ __typename login }}
  comments(first: 10) {{ {COMMENTS_FIELDS} }}
  statusCheckRollup {{
    id
    contexts(first: {CHECK_PAGE_SIZE}) {{ {CONTEXT_FIELDS} }}
  }}
"""
SEARCH_QUERY = f"""
query($search: String!, $cursor: String) {{
  search(query: $search, type: ISSUE, first: {PR_BATCH_SIZE}, after: $cursor) {{
    nodes {{ ... on PullRequest {{ {PR_FIELDS} }} }}
    pageInfo {{ hasNextPage endCursor }}
  }}
}}
"""


@dataclass(frozen=True)
class Connection:
    node_id: str
    node_type: str
    name: str
    fields: str
    first_page: dict[str, Any]


def graphql(query: str, **variables: str | None) -> dict[str, Any]:
    command = ["gh", "api", "graphql", "-f", f"query={query}"]
    for name, value in variables.items():
        if value is not None:
            command.extend(["-f", f"{name}={value}"])
    response: dict[str, Any] = json.loads(
        subprocess.check_output(command, text=True, stderr=subprocess.PIPE)
    )
    if response.get("errors") or not isinstance(response.get("data"), dict):
        raise RuntimeError(f"GraphQL lookup failed: {response.get('errors')}")
    data: dict[str, Any] = response["data"]
    return data


def next_cursor(connection: dict[str, Any]) -> str | None:
    page_info = connection["pageInfo"]
    if not page_info["hasNextPage"]:
        return None
    if not page_info["endCursor"]:
        raise RuntimeError("Missing cursor for an incomplete GraphQL connection")
    cursor: str = page_info["endCursor"]
    return cursor


def parse_numbers(value: str) -> list[str]:
    if not value.strip():
        return []
    numbers = [part.strip() for part in value.split(",")]
    if any(re.fullmatch(r"[1-9][0-9]*", number) is None for number in numbers):
        raise ValueError("pr_numbers must be a comma-separated list of positive PR numbers")
    return sorted(set(numbers), key=int)


def fetch_prs(repository: str, numbers: list[str]) -> list[dict[str, Any]]:
    owner, name = repository.split("/")
    prs: list[dict[str, Any]] = []
    if numbers:
        for offset in range(0, len(numbers), PR_BATCH_SIZE):
            batch = numbers[offset : offset + PR_BATCH_SIZE]
            aliases = "\n".join(
                f"pr_{number}: pullRequest(number: {number}) {{ {PR_FIELDS} }}" for number in batch
            )
            query = (
                "query($owner: String!, $name: String!) {"
                " repository(owner: $owner, name: $name) {" + aliases + "}}"
            )
            data = graphql(query, owner=owner, name=name)["repository"]
            if data is None or any(data.get(f"pr_{number}") is None for number in batch):
                raise RuntimeError("Could not resolve all requested PRs")
            prs.extend(data[f"pr_{number}"] for number in batch)
        return prs

    cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    search = f"repo:{repository} is:pr is:open draft:false label:auto-repair updated:>={cutoff}"
    cursor = None
    seen: set[str] = set()
    while True:
        connection = graphql(SEARCH_QUERY, search=search, cursor=cursor)["search"]
        prs.extend(connection["nodes"])
        cursor = next_cursor(connection)
        if cursor is None:
            return prs
        if cursor in seen:
            raise RuntimeError("Repeated pagination cursor for PR search")
        seen.add(cursor)


def fetch_connections(
    connections: list[Connection], page_size: int = 100
) -> list[list[dict[str, Any]]]:
    results: list[list[dict[str, Any]]] = [
        list(connection.first_page["nodes"]) for connection in connections
    ]
    cursors: dict[int, str] = {}
    seen: dict[int, set[str]] = {index: set() for index in range(len(connections))}
    for index, connection in enumerate(connections):
        cursor = next_cursor(connection.first_page)
        if cursor is not None:
            cursors[index] = cursor

    while cursors:
        batch = list(cursors)[:CONNECTION_BATCH_SIZE]
        declarations = []
        selections = []
        variables = {}
        for index in batch:
            connection = connections[index]
            declarations.extend([f"$id_{index}: ID!", f"$cursor_{index}: String!"])
            variables[f"id_{index}"] = connection.node_id
            variables[f"cursor_{index}"] = cursors[index]
            selections.append(f"""
              connection_{index}: node(id: $id_{index}) {{
                ... on {connection.node_type} {{
                  {connection.name}(first: {page_size}, after: $cursor_{index}) {{
                    nodes {{ {connection.fields} }}
                    pageInfo {{ hasNextPage endCursor }}
                  }}
                }}
              }}
            """)
        query = "query(" + ", ".join(declarations) + ") {" + "\n".join(selections) + "}"
        data = graphql(query, **variables)
        for index in batch:
            connection = connections[index]
            node = data.get(f"connection_{index}")
            if node is None or node.get(connection.name) is None:
                raise RuntimeError(f"Could not resolve {connection.node_type}.{connection.name}")
            page = node[connection.name]
            results[index].extend(page["nodes"])
            seen[index].add(cursors[index])
            cursor = next_cursor(page)
            if cursor in seen[index]:
                raise RuntimeError(f"Repeated pagination cursor for {connection.name}")
            if cursor is None:
                del cursors[index]
            else:
                cursors[index] = cursor

    return results


def check_connection(rollup: dict[str, Any]) -> Connection:
    # The rollup ID pins pagination to the originally fetched commit.
    return Connection(
        rollup["id"], "StatusCheckRollup", "contexts", CHECK_FIELDS, rollup["contexts"]
    )


def fetch_checks(rollup: dict[str, Any]) -> list[dict[str, Any]]:
    return fetch_connections([check_connection(rollup)], page_size=CHECK_PAGE_SIZE)[0]


def read_history(pr: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    comments = fetch_connections(
        [Connection(pr["id"], "PullRequest", "comments", COMMENT_FIELDS, pr["comments"])],
        page_size=10,
    )[0]
    comment = next(
        (
            comment
            for comment in comments
            if comment["author"] is not None
            and comment["author"]["__typename"] == "Bot"
            and any(
                marker in comment["body"] for marker in (HISTORY_MARKER, "<!-- repair-ledger -->")
            )
        ),
        None,
    )
    if comment is None:
        return None, {}
    body = comment["body"].split("```json", 1)[1].split("```", 1)[0]
    return comment["id"], json.loads(body)


def context_hash(context: dict[str, Any]) -> str:
    runs = [
        {**run, "job_runs": sorted(run["job_runs"], key=lambda job: job["id"])}
        for run in context["failed_runs"]
    ]
    context = {**context, "failed_runs": sorted(runs, key=lambda run: run["run_id"])}
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def skip_reason(context: dict[str, Any], history: dict[str, Any]) -> str | None:
    if entry := history.get(context_hash(context)):
        return f"context already processed: {entry['run_url']}"
    if len(history) >= PR_BUDGET:
        return f"PR budget exhausted ({PR_BUDGET} successful runs)"
    return None


def latest_workflow_runs(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for check in checks:
        if check["__typename"] != "CheckRun":
            continue
        suite = check["checkSuite"]
        app = suite["app"]
        if app is None or app["slug"] != "github-actions" or suite["workflowRun"] is None:
            continue
        run = suite["workflowRun"]
        if run["event"] == "dynamic":
            continue
        key = run["workflow"]["resourcePath"], run["event"]
        created_at = datetime.fromisoformat(run["createdAt"].replace("Z", "+00:00"))
        if run["databaseId"] is None:
            raise RuntimeError("Missing workflow run ID")
        rank = created_at, run["databaseId"]
        if key not in latest or rank > latest[key]["rank"]:
            latest[key] = {"rank": rank, "run": run, "checks": []}
        if rank == latest[key]["rank"]:
            latest[key]["checks"].append(check)

    return list(latest.values())


def unfinished_jobs(runs: list[dict[str, Any]]) -> list[str]:
    return [
        check["name"] for run in runs for check in run["checks"] if check["status"] != "COMPLETED"
    ]


def failed_job_context(check: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": check["databaseId"],
        "name": check["name"],
        "status": check["status"],
        "conclusion": check["conclusion"],
        "url": check["detailsUrl"] or check["permalink"],
    }


def build_candidate(pr: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    failed_runs: dict[int, dict[str, Any]] = {}
    for item in runs:
        suite = item["checks"][0]["checkSuite"]
        if suite["conclusion"] not in FAILED_CONCLUSIONS:
            continue
        job_runs = [
            failed_job_context(check)
            for check in item["checks"]
            if check["conclusion"] in FAILED_CONCLUSIONS
        ]
        if not job_runs:
            continue
        run = item["run"]
        run_id: int = run["databaseId"]
        attempt: int | None = run.get("runAttempt")
        group = failed_runs.setdefault(
            run_id,
            {
                "run_id": run_id,
                "workflow": run["workflow"]["resourcePath"].rsplit("/", 1)[-1],
                "status": suite["status"],
                "conclusion": suite["conclusion"],
                "run_attempt": attempt,
                "url": (
                    f"https://github.com/{REPO}/actions/runs/{run_id}/attempts/{attempt}"
                    if attempt is not None
                    else None
                ),
                "job_runs": [],
            },
        )
        group["job_runs"].extend(job_runs)
    author = pr["author"] or {}
    return {
        "number": pr["number"],
        "author": author.get("login"),
        "author_is_bot": author.get("__typename") == "Bot",
        "cross_repository": pr["isCrossRepository"],
        "head_ref": pr["headRefName"],
        "head_sha": pr["headRefOid"],
        "base_ref": pr["baseRefName"],
        "base_sha": pr["baseRefOid"],
        "failed_runs": list(failed_runs.values()),
    }


def parse_max_retries(value: str) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValueError("MAX_RETRIES must be a non-negative integer")
    return int(value)


def select_candidates(prs: list[dict[str, Any]], max_retries: int) -> list[dict[str, Any]]:
    eligible = []
    for pr in prs:
        if pr["state"] != "OPEN" or pr["isDraft"] or pr["mergeable"] == "CONFLICTING":
            continue
        rollup = pr["statusCheckRollup"]
        if rollup is None:
            sys.stderr.write(f"Skipping PR #{pr['number']}: check status is unavailable.\n")
            continue
        eligible.append(pr)

    connections = [check_connection(pr["statusCheckRollup"]) for pr in eligible]
    candidates = []
    checks_by_pr = fetch_connections(connections, page_size=CHECK_PAGE_SIZE)
    for pr, checks in zip(eligible, checks_by_pr, strict=True):
        runs = latest_workflow_runs(checks)
        if pending := unfinished_jobs(runs):
            sys.stderr.write(
                f"Skipping PR #{pr['number']}: unfinished jobs: {', '.join(pending)}\n"
            )
            continue
        pending_suites = [
            item["run"]["workflow"]["resourcePath"].rsplit("/", 1)[-1]
            for item in runs
            if item["checks"][0]["checkSuite"]["status"] != "COMPLETED"
        ]
        if pending_suites:
            sys.stderr.write(
                f"Skipping PR #{pr['number']}: unfinished workflows: {', '.join(pending_suites)}\n"
            )
            continue
        candidate = build_candidate(pr, runs)
        if any(
            run["run_attempt"] is None or run["run_attempt"] - 1 >= max_retries
            for run in candidate["failed_runs"]
        ):
            sys.stderr.write(
                f"Skipping PR #{pr['number']}: a failed run has exhausted its retry budget "
                "or has no attempt count.\n"
            )
            continue
        candidate["failed_runs"] = [
            run for run in candidate["failed_runs"] if run["workflow"] not in EXCLUDED_WORKFLOWS
        ]
        try:
            _, history = read_history(pr)
        except (ValueError, IndexError) as error:
            sys.stderr.write(f"Skipping PR #{pr['number']}: invalid repair history: {error}\n")
            continue
        if reason := skip_reason(candidate, history):
            sys.stderr.write(f"Skipping PR #{pr['number']}: {reason}\n")
            continue
        candidates.append(candidate)
    return candidates


def discover(args: argparse.Namespace) -> int:
    try:
        max_retries = parse_max_retries(os.environ.get("MAX_RETRIES") or "3")
        prs = fetch_prs(REPO, parse_numbers(args.prs))
        candidates = select_candidates(prs, max_retries)
    except subprocess.CalledProcessError as error:
        sys.stderr.write((error.stderr or str(error)) + "\n")
        return 1
    except (RuntimeError, ValueError, KeyError, TypeError) as error:
        sys.stderr.write(f"Could not select Repair PRs: {error}\n")
        return 1
    args.out_path.write_text(json.dumps(candidates) + "\n")
    return 0


def write_step_summary(message: str) -> None:
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a") as stream:
            stream.write(message + "\n")
    else:
        sys.stderr.write(message + "\n")


def write_step_outputs(**values: str) -> None:
    if path := os.environ.get("GITHUB_OUTPUT"):
        with Path(path).open("a") as stream:
            stream.writelines(f"{k}={v}\n" for k, v in values.items())


def fetch_pr_comments(number: int) -> dict[str, Any]:
    pr: dict[str, Any] = graphql(
        f"""
        query {{
          repository(owner: "mlflow", name: "mlflow") {{
            pullRequest(number: {number}) {{
              id
              comments(first: 10) {{ {COMMENTS_FIELDS} }}
            }}
          }}
        }}
        """,
    )["repository"]["pullRequest"]
    return pr


def check_budget(args: argparse.Namespace) -> int:
    context = json.loads(args.context_path.read_text())
    # Re-read under the per-PR job lock: discovery may have preceded another repair.
    _, history = read_history(fetch_pr_comments(context["number"]))
    reason = skip_reason(context, history)
    if reason is not None:
        write_step_summary(f"Skipping PR #{context['number']}: {reason}")
    write_step_outputs(eligible=str(reason is None).lower())
    return 0


def record(args: argparse.Namespace) -> int:
    context = json.loads(args.context_path.read_text())
    pr = fetch_pr_comments(context["number"])
    comment_id, history = read_history(pr)
    history[context_hash(context)] = {
        "run_url": args.run_url,
        "head_sha": context["head_sha"],
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    rows = ["| Recorded (UTC) | PR head | Workflow run |", "| --- | --- | --- |"]
    for entry in sorted(
        history.values(), key=lambda entry: entry.get("recorded_at", ""), reverse=True
    ):
        recorded_at = entry.get("recorded_at", "").removesuffix("+00:00").replace("T", " ")
        head_link = "—"
        if head_sha := entry.get("head_sha"):
            head_link = (
                f"[`{head_sha[:7]}`](https://github.com/{REPO}/pull/{context['number']}"
                f"/commits/{head_sha})"
            )
        rows.append(f"| {recorded_at or '—'} | {head_link} | [View run]({entry['run_url']}) |")
    table = "\n".join(rows)
    workflow = "`repair.yml`"
    if workflow_sha := os.environ.get("GITHUB_WORKFLOW_SHA"):
        workflow_url = f"https://github.com/{REPO}/blob/{workflow_sha}/.github/workflows/repair.yml"
        workflow = f"[{workflow}]({workflow_url})"
    body = (
        f"{HISTORY_MARKER}\n\n"
        "### Repair history\n\n"
        "> [!NOTE]\n"
        f"> You can ignore this comment. {workflow} records live runs here and "
        "checks this history to skip repeat contexts and enforce the per-PR limit.\n\n"
        f"{table}\n\n"
        "<details>\n<summary>History data</summary>\n\n"
        f"```json\n{json.dumps(history, indent=2)}\n```\n\n"
        "</details>"
    )
    if args.dry_run:
        write_step_summary(body)
        return 0
    if comment_id is None:
        comment_id = graphql(
            """
            mutation($id: ID!, $body: String!) {
              addComment(input: {subjectId: $id, body: $body}) {
                commentEdge { node { id } }
              }
            }
            """,
            id=pr["id"],
            body=body,
        )["addComment"]["commentEdge"]["node"]["id"]
    else:
        graphql(
            """
            mutation($id: ID!, $body: String!) {
              updateIssueComment(input: {id: $id, body: $body}) { issueComment { id } }
            }
            """,
            id=comment_id,
            body=body,
        )
    graphql(
        """
        mutation($id: ID!) {
          minimizeComment(input: {subjectId: $id, classifier: OFF_TOPIC}) {
            minimizedComment { isMinimized }
          }
        }
        """,
        id=comment_id,
    )
    write_step_summary(f"Recorded repair {len(history)}/{PR_BUDGET}: {args.run_url}")
    return 0


def validate_rerun(context: dict[str, Any], action: dict[str, Any], out_dir: Path) -> None:
    allowed = {run["run_id"] for run in context["failed_runs"]}
    if not set(action["run_ids"]) <= allowed:
        raise ValueError("Rerun IDs must belong to the captured failed runs")


def rerun_action(
    context: dict[str, Any], action: dict[str, Any], dry_run: bool, out_dir: Path
) -> None:
    runs = [run for run in context["failed_runs"] if run["run_id"] in action["run_ids"]]

    def submit(run: dict[str, Any]) -> None:
        if dry_run:
            write_step_summary(f"Would rerun failed jobs for {run['workflow']}: {run['url']}")
        else:
            subprocess.check_call([
                "gh",
                "run",
                "rerun",
                str(run["run_id"]),
                "--repo",
                REPO,
                "--failed",
            ])

    failed = False
    protect_run: dict[str, Any] | None = None
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        for run in runs:
            if run["workflow"] == "protect.yml":
                protect_run = run
            else:
                futures.append(pool.submit(submit, run))
        for future in futures:
            try:
                future.result()
            except subprocess.CalledProcessError as error:
                sys.stderr.write((error.stderr or str(error)) + "\n")
                failed = True
    # Submit Protect last so it sees the new attempts instead of the previous failures.
    if protect_run is not None:
        try:
            submit(protect_run)
        except subprocess.CalledProcessError as error:
            sys.stderr.write((error.stderr or str(error)) + "\n")
            failed = True
    if failed:
        raise ValueError("One or more rerun submissions failed")


@dataclass(frozen=True)
class ActionHandler:
    validate: Callable[[dict[str, Any], dict[str, Any], Path], None]
    execute: Callable[[dict[str, Any], dict[str, Any], bool, Path], None]


def validate_push(context: dict[str, Any], action: dict[str, Any], out_dir: Path) -> None:
    behind_by = subprocess.check_output(
        [
            "gh",
            "api",
            f"repos/{REPO}/compare/{context['base_ref']}...{context['head_sha']}",
            "--jq",
            ".behind_by",
        ],
        text=True,
    ).strip()
    if behind_by == "0":
        raise ValueError(f"The PR is already up to date with {context['base_ref']}")


def fetch_head_sha(number: int) -> str:
    return subprocess.check_output(
        ["gh", "api", f"repos/{REPO}/pulls/{number}", "--jq", ".head.sha"], text=True
    ).strip()


def push_action(
    context: dict[str, Any], action: dict[str, Any], dry_run: bool, out_dir: Path
) -> None:
    pr_url = f"https://github.com/{REPO}/pull/{context['number']}"
    if dry_run:
        write_step_summary(f"Would merge the base branch into {pr_url}")
        return
    # expected_head_sha rejects the update if the PR moved after the context was captured.
    subprocess.check_call([
        "gh",
        "api",
        "--method",
        "PUT",
        f"repos/{REPO}/pulls/{context['number']}/update-branch",
        "-f",
        f"expected_head_sha={context['head_sha']}",
    ])
    # The update is asynchronous: GitHub accepts the request and pushes the merge later.
    for _ in range(20):
        time.sleep(3)
        if fetch_head_sha(context["number"]) != context["head_sha"]:
            break
    else:
        raise RuntimeError("The PR head did not move within 60s")


def is_triage_pr(context: dict[str, Any]) -> bool:
    return (
        context.get("author_is_bot") is True
        and context.get("author") in {APP, APP.removesuffix("[bot]")}
        and context["head_ref"].startswith("triage/")
    )


def allowed_patch_paths() -> tuple[str, ...]:
    path = Path(__file__).resolve().parents[2] / "patch-paths.txt"
    lines = (line.strip() for line in path.read_text().splitlines())
    return tuple(line for line in lines if line and not line.startswith("#"))


def validate_fix(context: dict[str, Any], action: dict[str, Any], out_dir: Path) -> None:
    if context.get("cross_repository") is not False or not is_triage_pr(context):
        raise ValueError("Fixes are limited to triage PRs on mlflow/mlflow branches")
    patch = out_dir / "fix.patch"
    if patch.is_symlink() or not patch.is_file():
        raise ValueError("A fix needs a regular fix.patch file in the output directory")


def check_staged_fix(git: Callable[..., str]) -> None:
    def names(*args: str) -> list[str]:
        return [name for name in git(*args).split("\0") if name]

    if not (paths := names("diff", "--cached", "-z", "--name-only", "--no-renames")):
        raise ValueError("fix.patch changes no files")
    allowed = allowed_patch_paths()
    if outside := [path for path in paths if not path.startswith(allowed)]:
        raise ValueError(f"fix.patch changes paths outside patch-paths.txt: {outside}")
    if any(
        not entry.startswith(("100644 ", "100755 "))
        for entry in names("ls-files", "-s", "-z", "--", *paths)
    ):
        raise ValueError("fix.patch contains a symlink or submodule")
    deleted = names("diff", "--cached", "-z", "--name-only", "--no-renames", "--diff-filter=D")
    if deleted_tests := [path for path in deleted if TEST_FILE_RE.search(path)]:
        raise ValueError(f"fix.patch deletes test files: {deleted_tests}")
    headers = ("+++ b/", "+++ /dev/null", "--- a/", "--- /dev/null")
    added: list[tuple[str, str]] = []
    removed: list[tuple[str, str]] = []
    for line in git("diff", "--cached", "--no-renames", "-U0").splitlines():
        if line.startswith("diff --git "):
            path = line.rsplit(" b/", 1)[1]
        elif line[:1] in ("+", "-") and not line.startswith(headers):
            (added if line[0] == "+" else removed).append((path, line[1:]))
    if weakening := [line.strip() for _, line in added if TEST_WEAKENING_RE.search(line)]:
        raise ValueError(f"fix.patch skips or xfails tests: {weakening}")

    def tests(source: list[tuple[str, str]]) -> set[tuple[str, str]]:
        return {(f, m[1] or m[3]) for f, line in source if (m := TEST_DEFINITION_RE.match(line))}

    if dropped := sorted(tests(removed) - tests(added)):
        raise ValueError(f"fix.patch removes tests: {dropped}")


def fix_action(
    context: dict[str, Any], action: dict[str, Any], dry_run: bool, out_dir: Path
) -> None:
    checkout = Path(os.environ["PR_CHECKOUT"])

    def git(*args: str | Path) -> str:
        command = ["git", "-C", checkout, "-c", "core.hooksPath=/dev/null", *args]
        return subprocess.check_output(command, text=True).strip()

    if git("rev-parse", "HEAD") != context["head_sha"]:
        raise ValueError("The PR checkout does not match the captured head")
    git("apply", "--index", "--whitespace=error", "--", out_dir / "fix.patch")
    check_staged_fix(git)
    pr_url = f"https://github.com/{REPO}/pull/{context['number']}"
    if dry_run:
        write_step_summary(
            f"Would push to {pr_url}:\n\n```\n{git('diff', '--cached', '--stat')}\n```"
        )
        return
    identity = [
        "-c",
        f"user.name={APP}",
        "-c",
        f"user.email={APP_ID}+{APP}@users.noreply.github.com",
    ]
    trailer = f"Repair-Run: {os.environ['RUN_URL']}"
    git(*identity, "commit", "-s", "-m", action["commit_message"], "-m", trailer)
    # The lease rejects the push if the PR moved after the context was captured.
    ref = f"refs/heads/{context['head_ref']}"
    auth = ["-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential"]
    lease = f"--force-with-lease={ref}:{context['head_sha']}"
    git(*auth, "push", lease, f"https://github.com/{REPO}.git", f"HEAD:{ref}")
    write_step_summary(f"Pushed a fix to {pr_url}: {git('rev-parse', 'HEAD')}")
    # Ask NaiLaOpus to review the fix. A failed request mustn't stop the fix being recorded.
    comments = f"repos/{REPO}/issues/{context['number']}/comments"
    if subprocess.call(["gh", "api", comments, "-f", "body=/review"]) != 0:
        write_step_summary(f"Could not request a NaiLaOpus review on {pr_url}")


ACTION_HANDLERS = {
    "rerun": ActionHandler(validate_rerun, rerun_action),
    "push": ActionHandler(validate_push, push_action),
    "fix": ActionHandler(validate_fix, fix_action),
}


def load_plan(context: dict[str, Any], path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text())
    subprocess.check_output(
        [
            "uvx",
            "check-jsonschema@0.37.4",
            "--schemafile",
            Path(__file__).parents[1] / "payload.schema.yml",
            path,
        ],
        text=True,
    )
    if (action := payload["action"]) is not None:
        ACTION_HANDLERS[action["type"]].validate(context, action, path.parent)
    return payload


def validate(args: argparse.Namespace) -> int:
    context = json.loads(args.context_path.read_text())
    payload = load_plan(context, args.payload_path)
    action = payload["action"]
    write_step_outputs(action_type=action["type"] if action is not None else "")
    return 0


def execute(args: argparse.Namespace) -> int:
    context = json.loads(args.context_path.read_text())
    payload = load_plan(context, args.payload_path)
    write_step_summary(
        f"### Repair PR #{context['number']}\n\n```json\n{json.dumps(payload, indent=2)}\n```"
    )
    if (action := payload["action"]) is not None:
        ACTION_HANDLERS[action["type"]].execute(
            context, action, args.dry_run, args.payload_path.parent
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair workflow commands")
    commands = parser.add_subparsers(dest="command", required=True)
    discovery = commands.add_parser("discover", help="Select PRs without unfinished CI jobs")
    discovery.add_argument("--prs", default=os.environ.get("PR_NUMBERS", ""))
    discovery.add_argument("--out-path", type=Path, required=True)
    discovery.set_defaults(func=discover)
    budget = commands.add_parser("check-budget", help="Check the PR history before repairing")
    budget.add_argument("--context", dest="context_path", type=Path, required=True)
    budget.set_defaults(func=check_budget)
    recording = commands.add_parser("record", help="Record a successful repair in the PR history")
    recording.add_argument("--context", dest="context_path", type=Path, required=True)
    recording.add_argument("--run-url", required=True)
    recording.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    recording.set_defaults(func=record)
    validation = commands.add_parser("validate", help="Validate a repair payload")
    validation.add_argument("--payload", dest="payload_path", type=Path, required=True)
    validation.add_argument("--context", dest="context_path", type=Path, required=True)
    validation.set_defaults(func=validate)
    execution = commands.add_parser("execute", help="Execute a repair payload")
    execution.add_argument("--payload", dest="payload_path", type=Path, required=True)
    execution.add_argument("--context", dest="context_path", type=Path, required=True)
    execution.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    execution.set_defaults(func=execute)
    args = parser.parse_args()
    handler: Callable[[argparse.Namespace], int] = args.func
    try:
        return handler(args)
    except subprocess.CalledProcessError as error:
        sys.stderr.write((error.stderr or str(error)) + "\n")
        return 1
    except (RuntimeError, ValueError, KeyError, TypeError, OSError) as error:
        sys.stderr.write(f"Repair failed: {error}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
