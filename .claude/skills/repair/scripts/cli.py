import argparse
import html
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

FAILED_CONCLUSIONS = {"FAILURE", "TIMED_OUT", "STARTUP_FAILURE"}
EXCLUDED_WORKFLOWS = {"rerun.yml", "repair.yml"}
CONNECTION_BATCH_SIZE = 20
REPO = "mlflow/mlflow"

STEP_FIELDS = """
  name
  number
  status
  conclusion
"""
# Limit the first step page to keep nested PR/check/step searches below GitHub's node limit.
CHECK_FIELDS = f"""
  __typename
  ... on CheckRun {{
    id
    databaseId
    name
    status
    conclusion
    detailsUrl
    permalink
    steps(first: 20) {{
      nodes {{ {STEP_FIELDS} }}
      pageInfo {{ hasNextPage endCursor }}
    }}
    checkSuite {{
      status
      conclusion
      app {{ slug }}
      workflowRun {{
        databaseId
        createdAt
        event
        runAttempt
        workflow {{ resourcePath }}
      }}
    }}
  }}
"""
CONTEXT_FIELDS = f"""
  nodes {{ {CHECK_FIELDS} }}
  pageInfo {{ hasNextPage endCursor }}
"""
PR_FIELDS = f"""
  number
  headRefOid
  state
  isDraft
  mergeable
  statusCheckRollup {{
    id
    contexts(first: 100) {{ {CONTEXT_FIELDS} }}
  }}
"""
SEARCH_QUERY = f"""
query($search: String!, $cursor: String) {{
  search(query: $search, type: ISSUE, first: 100, after: $cursor) {{
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
        for offset in range(0, len(numbers), 100):
            batch = numbers[offset : offset + 100]
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
    search = f"repo:{repository} is:pr is:open draft:false label:auto-repair created:>={cutoff}"
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


def fetch_connections(connections: list[Connection]) -> list[list[dict[str, Any]]]:
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
                  {connection.name}(first: 100, after: $cursor_{index}) {{
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
    return fetch_connections([check_connection(rollup)])[0]


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
    steps = (
        fetch_connections([
            Connection(check["id"], "CheckRun", "steps", STEP_FIELDS, check["steps"])
        ])[0]
        if check["steps"] is not None
        else []
    )
    return {
        "id": check["databaseId"],
        "name": check["name"],
        "status": check["status"],
        "conclusion": check["conclusion"],
        "url": check["detailsUrl"] or check["permalink"],
        "failed_steps": [step for step in steps if step["conclusion"] in FAILED_CONCLUSIONS],
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
    return {
        "number": pr["number"],
        "sha": pr["headRefOid"],
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
    for pr, checks in zip(eligible, fetch_connections(connections), strict=True):
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
    sys.stdout.write(json.dumps(candidates) + "\n")
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


def validate_rerun(context: dict[str, Any], action: dict[str, Any]) -> None:
    allowed = {run["run_id"] for run in context["failed_runs"]}
    if not set(action["run_ids"]) <= allowed:
        raise ValueError("Rerun IDs must belong to the captured failed runs")


def rerun_action(context: dict[str, Any], action: dict[str, Any], dry_run: bool) -> None:
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
    validate: Callable[[dict[str, Any], dict[str, Any]], None]
    execute: Callable[[dict[str, Any], dict[str, Any], bool], None]


ACTION_HANDLERS = {"rerun": ActionHandler(validate_rerun, rerun_action)}


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
    for action in payload["actions"]:
        ACTION_HANDLERS[action["type"]].validate(context, action)
    return payload


def validate(args: argparse.Namespace) -> int:
    context = json.loads(args.context_path.read_text())
    payload = load_plan(context, args.payload_path)
    write_step_outputs(has_actions=str(bool(payload["actions"])).lower())
    return 0


def execute(args: argparse.Namespace) -> int:
    context = json.loads(args.context_path.read_text())
    payload = load_plan(context, args.payload_path)
    write_step_summary(f"### Repair PR #{context['number']}\n{html.escape(payload['summary'])}")
    for action in payload["actions"]:
        ACTION_HANDLERS[action["type"]].execute(context, action, args.dry_run)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair workflow commands")
    commands = parser.add_subparsers(dest="command", required=True)
    discovery = commands.add_parser("discover", help="Select PRs without unfinished CI jobs")
    discovery.add_argument("--prs", default=os.environ.get("PR_NUMBERS", ""))
    discovery.set_defaults(func=discover)
    plan_args = argparse.ArgumentParser(add_help=False)
    plan_args.add_argument("--payload", dest="payload_path", type=Path, required=True)
    plan_args.add_argument("--context", dest="context_path", type=Path, required=True)
    for name, callback in {"validate": validate, "execute": execute}.items():
        subcommand = commands.add_parser(
            name, parents=[plan_args], help=f"{name.capitalize()} a repair payload"
        )
        if name == "execute":
            subcommand.add_argument(
                "--dry-run", action=argparse.BooleanOptionalAction, default=True
            )
        subcommand.set_defaults(func=callback)
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
