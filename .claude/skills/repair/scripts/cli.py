import argparse
import hashlib
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
PATCH_ACTIONS = {"fix-ci", "resolve-conflicts"}
EXCLUDED_WORKFLOWS = {"rerun.yml", "repair.yml"}
CONNECTION_BATCH_SIZE = 20
REPO = "mlflow/mlflow"

CHECK_FIELDS = """
  __typename
  ... on CheckRun {
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
    return {
        "number": pr["number"],
        "sha": pr["headRefOid"],
        "mergeable": pr["mergeable"],
        "failed_runs": list(failed_runs.values()),
    }


def parse_max_retries(value: str) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValueError("MAX_RETRIES must be a non-negative integer")
    return int(value)


def select_candidates(prs: list[dict[str, Any]], max_retries: int) -> list[dict[str, Any]]:
    eligible = []
    for pr in prs:
        if pr["state"] != "OPEN" or pr["isDraft"]:
            continue
        rollup = pr["statusCheckRollup"]
        if rollup is None and pr["mergeable"] != "CONFLICTING":
            sys.stderr.write(f"Skipping PR #{pr['number']}: check status is unavailable.\n")
            continue
        eligible.append(pr)

    checked = [pr for pr in eligible if pr["statusCheckRollup"] is not None]
    connections = [check_connection(pr["statusCheckRollup"]) for pr in checked]
    checks_by_number = dict(
        zip([pr["number"] for pr in checked], fetch_connections(connections), strict=True)
    )
    candidates = []
    for pr in eligible:
        runs = latest_workflow_runs(checks_by_number.get(pr["number"], []))
        conflicting = pr["mergeable"] == "CONFLICTING"
        if pending := unfinished_jobs(runs):
            sys.stderr.write(
                f"Skipping PR #{pr['number']}: unfinished jobs: {', '.join(pending)}\n"
            )
            if not conflicting:
                continue
            runs = []
        pending_suites = [
            item["run"]["workflow"]["resourcePath"].rsplit("/", 1)[-1]
            for item in runs
            if item["checks"][0]["checkSuite"]["status"] != "COMPLETED"
        ]
        if pending_suites:
            sys.stderr.write(
                f"Skipping PR #{pr['number']}: unfinished workflows: {', '.join(pending_suites)}\n"
            )
            if not conflicting:
                continue
            runs = []
        candidate = build_candidate(pr, runs)
        if any(
            run["run_attempt"] is None or run["run_attempt"] - 1 >= max_retries
            for run in candidate["failed_runs"]
        ):
            sys.stderr.write(
                f"Skipping PR #{pr['number']}: a failed run has exhausted its retry budget "
                "or has no attempt count.\n"
            )
            if not conflicting:
                continue
            candidate["failed_runs"] = []
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


def command(*args: str | Path, input: str | None = None) -> str:
    return subprocess.check_output(args, input=input, text=True, stderr=subprocess.PIPE)


def rest(endpoint: str) -> Any:
    return json.loads(command("gh", "api", endpoint))


def git(*args: str | Path, input: str | None = None) -> str:
    return command(
        "git",
        "--no-replace-objects",
        "-C",
        os.environ["PR_CHECKOUT"],
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "user.name=mlflow-app[bot]",
        "-c",
        "user.email=191841109+mlflow-app[bot]@users.noreply.github.com",
        *args,
        input=input,
    ).strip()


def summary(message: str) -> None:
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a") as stream:
            stream.write(message + "\n")
    else:
        sys.stderr.write(message + "\n")


def outputs(**values: str) -> None:
    if path := os.environ.get("GITHUB_OUTPUT"):
        with Path(path).open("a") as stream:
            for key, value in values.items():
                stream.write(f"{key}={value}\n")


def context_file() -> Path:
    return Path(os.environ["CONTEXT_PATH"])


def output_dir() -> Path:
    return Path(os.environ["OUT_DIR"])


def context_key(context: dict[str, Any]) -> str:
    identity = {
        "number": context["number"],
        "sha": context["sha"],
        "base": context["base"],
        "failed_runs": sorted([
            (run["run_id"], run["run_attempt"]) for run in context["failed_runs"]
        ]),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def clean_checkout(context: dict[str, Any]) -> None:
    if git("rev-parse", "HEAD") != context["sha"] or git("status", "--porcelain"):
        raise ValueError("PR checkout must be clean and match the captured head")


def prepare_merge(context: dict[str, Any]) -> dict[str, Any]:
    clean_checkout(context)
    if context["head_repo"] != REPO:
        return {"status": "fork", "conflicted_paths": []}
    base = context["base"]["sha"]
    git("fetch", "origin", base)
    try:
        git("merge-base", "--is-ancestor", base, "HEAD")
    except subprocess.CalledProcessError as error:
        if error.returncode != 1:
            raise
    else:
        return {"status": "up-to-date", "conflicted_paths": []}
    try:
        git("merge", "--no-commit", "--no-ff", base)
    except subprocess.CalledProcessError:
        conflicts = git("diff", "--name-only", "--diff-filter=U", "-z").split("\0")
        conflicts = [path for path in conflicts if path]
        if not conflicts:
            raise
        return {"status": "conflicting", "conflicted_paths": conflicts}
    return {"status": "clean", "conflicted_paths": []}


def git_metadata() -> dict[str, str | None]:
    directory = Path(os.environ["PR_CHECKOUT"]) / ".git"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Expected a regular Git directory")
    result: dict[str, str | None] = {}
    for name in (
        "commondir",
        "config",
        "config.worktree",
        "info/grafts",
        "info/attributes",
        "objects/info/alternates",
    ):
        path = directory / name
        if path.is_symlink() or any(
            parent.is_symlink() for parent in path.parents if directory in parent.parents
        ):
            raise ValueError("Unexpected symlink in Git metadata")
        if path.exists() and not path.is_file():
            raise ValueError("Expected regular Git configuration files")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return result


def prepare(args: argparse.Namespace) -> int:
    candidate = json.loads(os.environ["CANDIDATE"])
    pr = rest(f"repos/{REPO}/pulls/{candidate['number']}")
    if pr["state"] != "open" or pr["draft"] or pr["head"]["sha"] != candidate["sha"]:
        raise ValueError("PR changed since discovery")
    context = dict(
        candidate,
        head_ref=pr["head"]["ref"],
        head_repo=(pr["head"]["repo"] or {}).get("full_name"),
        base={"ref": pr["base"]["ref"], "sha": pr["base"]["sha"]},
    )
    context["key"] = context_key(context)
    directory = context_file().parent
    directory.mkdir(parents=True, exist_ok=True)
    output_dir().mkdir(parents=True, exist_ok=True)
    context["merge"] = prepare_merge(context)
    context["repair_commits"] = git(
        "log",
        "--first-parent",
        "--format=%B",
        "--grep=^Repair-Context:",
        f"{context['base']['sha']}..HEAD",
    ).count("Repair-Context:")
    context["git_metadata"] = git_metadata()
    logs = directory / "logs"
    logs.mkdir()
    for run in context["failed_runs"]:
        for job in run["job_runs"]:
            job_id = int(job["id"])
            path = logs / f"{job_id}.log"
            try:
                content = command("gh", "api", f"repos/{REPO}/actions/jobs/{job_id}/logs")
            except subprocess.CalledProcessError as error:
                content = f"Could not fetch job log: {error.stderr or error}\n"
            path.write_text(content[-200_000:])
            job["log_path"] = str(path)
    context_file().write_text(json.dumps(context))
    outputs(
        context_hash=hashlib.sha256(context_file().read_bytes()).hexdigest(),
    )
    return 0


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate payload key")
    return result


def load_plan(context: dict[str, Any], directory: Path) -> dict[str, Any]:
    path = directory / "payload.json"
    patch = directory / "fix.patch"
    if directory.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular payload file")
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Payload exceeds 1 MB")
    payload: dict[str, Any] = json.loads(path.read_text(), object_pairs_hook=unique_keys)
    actions = payload["actions"]
    kinds = [action["type"] for action in actions]
    if len(kinds) != len(set(kinds)) or set(kinds) - ({"rerun", "sync-base"} | PATCH_ACTIONS):
        raise ValueError("Duplicate or unsupported action")
    if "rerun" in kinds and len(kinds) != 1:
        raise ValueError("Reruns cannot accompany a head-changing action")
    if (set(kinds) & ({"sync-base"} | PATCH_ACTIONS)) and context["repair_commits"] >= 3:
        raise ValueError("PR reached the limit of 3 automatic repair commits")
    patch_kinds = set(kinds) & PATCH_ACTIONS
    if len(patch_kinds) > 1:
        raise ValueError("Use one patch action for the final tree")
    if "resolve-conflicts" in kinds and (
        "sync-base" not in kinds or context["merge"]["status"] != "conflicting"
    ):
        raise ValueError("Conflict resolution requires a conflicting base sync")
    if "fix-ci" in kinds and context["merge"]["status"] == "conflicting":
        raise ValueError("Use resolve-conflicts for a conflicting merge")
    if "sync-base" in kinds:
        if kinds[0] != "sync-base" or context["merge"]["status"] not in {"clean", "conflicting"}:
            raise ValueError("Base sync must be first and needed")
        if context["merge"]["status"] == "conflicting" and "resolve-conflicts" not in kinds:
            raise ValueError("A conflicting base sync requires a resolution patch")
    elif patch_kinds and context["merge"]["status"] in {"clean", "conflicting"}:
        raise ValueError("Patch must include the prepared base sync")
    if bool(set(kinds) & ({"sync-base"} | PATCH_ACTIONS)) and context["head_repo"] != REPO:
        raise ValueError("Cannot publish changes to a fork")
    allowed = {run["run_id"] for run in context["failed_runs"]}
    for action in actions:
        if action["type"] == "rerun" and not set(action["run_ids"]) <= allowed:
            raise ValueError("Rerun IDs must belong to the captured failed runs")
    if patch.is_symlink() or (patch.exists() and not patch.is_file()):
        raise ValueError("Expected a regular patch file")
    if patch.exists() != bool(patch_kinds):
        raise ValueError("Patch file must match the action plan")
    if patch.exists() and patch.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Patch exceeds 20 MB")
    return payload


def apply_fix(patch: Path) -> None:
    git("apply", "--index", "--whitespace=error", "--", patch)
    paths = git("diff", "--cached", "--name-only", "--no-renames", "-z").split("\0")[:-1]
    if not paths:
        raise ValueError("Patch is empty")
    staged = git("ls-files", "--stage", "-z", "--", *paths).split("\0")[:-1]
    if any(not entry.startswith(("100644 ", "100755 ")) for entry in staged):
        raise ValueError("Patch contains a symlink or submodule")


def validate(args: argparse.Namespace) -> int:
    if expected := os.environ.get("EXPECTED_CONTEXT_HASH"):
        if hashlib.sha256(context_file().read_bytes()).hexdigest() != expected:
            raise ValueError("Captured context changed during repair")
    context = json.loads(context_file().read_text())
    if context["key"] != context_key(context):
        raise ValueError("Context identity changed")
    command(
        "uvx",
        "check-jsonschema@0.37.4",
        "--schemafile",
        Path(__file__).parents[1] / "payload.schema.yml",
        output_dir() / "payload.json",
    )
    payload = load_plan(context, output_dir())
    if os.environ.get("REUSE_CHECKOUT") == "true":
        if git_metadata() != context["git_metadata"]:
            raise ValueError("Agent changed the Git configuration")
        if git("rev-parse", "HEAD") != context["sha"]:
            raise ValueError("Agent changed the PR head")
        # The patch is outside this disposable checkout; rebuild its final tree from the payload.
        git("reset", "--hard", context["sha"])
        git("clean", "-fdx")
    clean_checkout(context)
    kinds = [action["type"] for action in payload["actions"]]
    if set(kinds) & PATCH_ACTIONS:
        patch = output_dir() / "fix.patch"
        if not ("resolve-conflicts" in kinds and patch.stat().st_size == 0):
            apply_fix(patch)
    elif "sync-base" in kinds:
        git("fetch", "origin", context["base"]["sha"])
        git("merge", "--no-commit", "--no-ff", context["base"]["sha"])
    workflow_changes = any(
        name.startswith(".github/workflows/")
        for name in git("diff", "--cached", "--name-only", "--no-renames", "-z").split("\0")
    )
    outputs(
        has_actions=str(bool(kinds)).lower(),
        context_key=context["key"],
        workflow_changes=str(workflow_changes).lower(),
    )
    return 0


def live_pr(context: dict[str, Any]) -> dict[str, Any]:
    pr: dict[str, Any] = rest(f"repos/{REPO}/pulls/{context['number']}")
    if (
        pr["state"] != "open"
        or pr["draft"]
        or pr["head"]["ref"] != context["head_ref"]
        or (pr["head"]["repo"] or {}).get("full_name") != context["head_repo"]
        or pr["base"]["ref"] != context["base"]["ref"]
        or pr["base"]["sha"] != context["base"]["sha"]
    ):
        raise ValueError("PR identity or base changed before publication")
    return pr


def publish_tree(context: dict[str, Any], sync: bool) -> None:
    tree = git("write-tree")
    parents = [context["sha"]] + ([context["base"]["sha"]] if sync else [])
    marker = f"Repair-Context: {context['key']}"
    current = live_pr(context)["head"]["sha"]
    if current != context["sha"]:
        previous = rest(f"repos/{REPO}/git/commits/{current}")
        if (
            marker in previous["message"].splitlines()
            and [parent["sha"] for parent in previous["parents"]] == parents
            and previous["tree"]["sha"] == tree
        ):
            summary(f"Repair already published as `{current}`.")
            return
        raise ValueError("PR head changed before publication")
    message = (
        f"Repair PR #{context['number']}\n\n{marker}\n\n"
        "Signed-off-by: mlflow-app[bot] <191841109+mlflow-app[bot]@users.noreply.github.com>\n"
    )
    parent_args = [value for parent in parents for value in ("-p", parent)]
    commit = git("commit-tree", tree, *parent_args, input=message)
    ref = f"refs/heads/{context['head_ref']}"
    git(
        "-c",
        "credential.helper=",
        "-c",
        "credential.helper=!gh auth git-credential",
        "push",
        f"--force-with-lease={ref}:{context['sha']}",
        f"https://github.com/{REPO}.git",
        f"{commit}:{ref}",
    )
    summary(f"Published repair commit `{commit}`.")


def rerun_action(context: dict[str, Any], action: dict[str, Any], dry_run: bool) -> None:
    if live_pr(context)["head"]["sha"] != context["sha"]:
        raise ValueError("PR head changed before reruns")
    runs = [run for run in context["failed_runs"] if run["run_id"] in action["run_ids"]]
    for run in runs:
        current = rest(f"repos/{REPO}/actions/runs/{run['run_id']}")
        if (
            current["head_sha"] != context["sha"]
            or current["run_attempt"] != run["run_attempt"]
            or current["status"] != "completed"
            or current["conclusion"].upper() not in FAILED_CONCLUSIONS
        ):
            raise ValueError("Workflow run changed before rerun submission")

    def submit(run: dict[str, Any]) -> None:
        if dry_run:
            summary(f"Would rerun failed jobs for {run['workflow']}: {run['url']}")
        else:
            command("gh", "run", "rerun", str(run["run_id"]), "--repo", REPO, "--failed")

    failed = False
    ordinary = [run for run in runs if run["workflow"] != "protect.yml"]
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(ordinary)))) as pool:
        futures = [pool.submit(submit, run) for run in ordinary]
        for future in futures:
            try:
                future.result()
            except subprocess.CalledProcessError as error:
                sys.stderr.write((error.stderr or str(error)) + "\n")
                failed = True
    for run in runs:
        if run["workflow"] == "protect.yml":
            try:
                submit(run)
            except subprocess.CalledProcessError as error:
                sys.stderr.write((error.stderr or str(error)) + "\n")
                failed = True
    if failed:
        raise ValueError("One or more rerun submissions failed")


def sync_base_action(context: dict[str, Any], patch_follows: bool, dry_run: bool) -> None:
    if dry_run:
        summary(f"Would sync base `{context['base']['sha']}` into PR #{context['number']}.")
    elif not patch_follows:
        publish_tree(context, sync=True)


def publish_patch(context: dict[str, Any], sync: bool, dry_run: bool) -> None:
    if dry_run:
        patch = (output_dir() / "fix.patch").read_text(errors="replace")
        summary(f"### Patch preview\n<pre>{html.escape(patch[:60_000])}</pre>")
    else:
        publish_tree(context, sync=sync)


def execute(args: argparse.Namespace) -> int:
    if args.command != "publish":
        validate(args)
    context = json.loads(context_file().read_text())
    payload = load_plan(context, output_dir())
    dry_run = os.environ.get("DRY_RUN", "true") == "true"
    kinds = [action["type"] for action in payload["actions"]]
    if args.command != "publish" and (not kinds or kinds[-1] != args.command):
        raise ValueError("Action subcommand must match the final action in the validated plan")
    summary(f"### Repair PR #{context['number']}\n{html.escape(payload['summary'])}")
    for action in payload["actions"]:
        match action["type"]:
            case "rerun":
                rerun_action(context, action, dry_run)
            case "sync-base":
                sync_base_action(context, bool(set(kinds) & PATCH_ACTIONS), dry_run)
            case "fix-ci" | "resolve-conflicts":
                publish_patch(context, "sync-base" in kinds, dry_run)
    result = Path(os.environ.get("RESULT_DIR", "/tmp/repair-result"))
    if result.is_symlink() or (result.exists() and not result.is_dir()):
        raise ValueError("Expected a regular result directory")
    result.mkdir(parents=True, exist_ok=True)
    if (result / "result.json").is_symlink():
        raise ValueError("Expected a regular result file")
    (result / "result.json").write_text(
        json.dumps({"context_key": context["key"], "payload": payload})
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair workflow commands")
    commands = parser.add_subparsers(dest="command", required=True)
    discovery = commands.add_parser("discover", help="Select PRs without unfinished CI jobs")
    discovery.add_argument("--prs", default=os.environ.get("PR_NUMBERS", ""))
    discovery.set_defaults(func=discover)
    for name, callback in {
        "prepare": prepare,
        "validate": validate,
        "publish": execute,
        "rerun": execute,
        "sync-base": execute,
        "fix-ci": execute,
        "resolve-conflicts": execute,
    }.items():
        commands.add_parser(name).set_defaults(func=callback)
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
