import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

APP = "mlflow-app[bot]"
APP_ID = 191841109
COPILOT = "copilot-pull-request-reviewer[bot]"
NAILAOPUS = "nailaopus[bot]"
MAX_ROUNDS = 5
PATCH_PATHS = Path(__file__).resolve().parents[2] / "patch-paths.txt"
PR_CONTEXT = Path(__file__).resolve().parents[3] / "scripts" / "pr_context.py"
CLOSE_THREAD = """
mutation($id: ID!) {
  resolveReviewThread(input: {threadId: $id}) { thread { isResolved } }
}
"""


def command(*args: str | Path, input: str | None = None) -> str:
    return subprocess.check_output(args, input=input, text=True)


def api(*args: str, data: dict[str, Any] | None = None) -> Any:
    if data is not None:
        args += ("--method", "POST", "--input", "-")
    result = command("gh", "api", *args, input=json.dumps(data) if data is not None else None)
    return json.loads(result) if result.strip() else None


def list_items(endpoint: str) -> list[dict[str, Any]]:
    return [item for page in api("--paginate", "--slurp", endpoint) for item in page]


def summary(text: str) -> None:
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
        stream.write(f"{text}\n")


def eligible(pr: dict[str, Any], repo: str) -> bool:
    return (
        pr["state"] == "open"
        and not pr["draft"]
        and pr["user"]["id"] == APP_ID
        and pr["user"]["login"] == APP
        and pr["user"]["type"] == "Bot"
        and (pr["head"]["repo"] or {}).get("full_name") == repo
        and pr["base"]["ref"] == os.environ["DEFAULT_BRANCH"]
        # Other bot PRs, such as lockfile updates, auto-merge without a maintainer approval.
        and pr["head"]["ref"].startswith("triage/")
    )


def fetch_timeline(repo: str, number: str, review_id: str) -> list[dict[str, Any]]:
    pr_context = json.loads(
        command(
            sys.executable,
            PR_CONTEXT,
            "--repo",
            repo,
            "--pr-number",
            number,
            "--review-id",
            review_id,
        )
    )
    timeline: list[dict[str, Any]] = pr_context["timeline"]
    return timeline


def app_comment(comment: dict[str, Any]) -> bool:
    return (comment["user"] or {}).get("id") == APP_ID


def marker(reviewer: str, review_id: str, key: str) -> str:
    return f"<!-- resolve:{reviewer}:{review_id}:{key} -->"


def collect(repo: str, number: str, review_id: str, context_path: Path) -> None:
    endpoint = f"repos/{repo}/pulls/{number}"
    pr = api(endpoint)
    if not eligible(pr, repo):
        summary("This PR is outside the resolver's scope.")
        return
    review = api(f"{endpoint}/reviews/{review_id}")
    user = review["user"] or {}
    reviewer = user.get("login", "")
    if review["state"] not in {"APPROVED", "CHANGES_REQUESTED", "COMMENTED"} or reviewer == APP:
        return
    if user.get("type") == "User":
        if review["author_association"] not in {"OWNER", "MEMBER", "COLLABORATOR"}:
            return
        role = api(f"repos/{repo}/collaborators/{reviewer}/permission")["role_name"]
        if role not in {"write", "maintain", "admin"}:
            summary("The reviewer lacks write access.")
            return
    elif reviewer not in {COPILOT, NAILAOPUS}:
        summary("This bot reviewer is outside the resolver's scope.")
        return

    comments = list_items(f"{endpoint}/comments")
    issue_comments = list_items(f"repos/{repo}/issues/{number}/comments")
    # Each handled review leaves markers in the bot's replies; count them as rounds.
    rounds = {
        match.group(1)
        for comment in comments + issue_comments
        if app_comment(comment)
        for match in re.finditer(r"<!-- resolve:[^:]+:(\d+):", comment["body"].rpartition("\n")[2])
    }
    if review_id not in rounds and len(rounds) >= MAX_ROUNDS:
        summary(f"The PR has reached its limit of {MAX_ROUNDS} automatic review rounds.")
        return

    timeline = fetch_timeline(repo, number, review_id)
    context = {
        "head_sha": pr["head"]["sha"],
        "head_ref": pr["head"]["ref"],
        "review_id": int(review_id),
        "timeline": timeline,
    }
    if not any(item.get("id") == context["review_id"] for item in timeline):
        raise ValueError("The submitted review is missing from the PR's reviews")
    review_body = review["body"] or ""
    if not review_thread_ids(context) and (
        not review_body.strip() or reviewer == COPILOT or review["state"] == "APPROVED"
    ):
        summary("The submitted review has no open feedback to address.")
        return
    context_path.write_text(json.dumps(context))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"head_sha={context['head_sha']}\n")


def current_review(context: dict[str, Any]) -> dict[str, Any]:
    return next(item for item in context["timeline"] if item.get("id") == context["review_id"])


def all_threads(context: dict[str, Any]) -> list[dict[str, Any]]:
    return [thread for item in context["timeline"] for thread in item.get("threads", [])]


def review_thread_ids(context: dict[str, Any]) -> set[str]:
    return {
        thread["id"]
        for thread in all_threads(context)
        if any(c["review_id"] == context["review_id"] for c in thread["comments"])
    }


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate key in the resolution payload")
    return result


def validate(context_path: Path, out_dir: Path) -> None:
    payload = out_dir / "payload.json"
    patch = out_dir / "fix.patch"
    media = out_dir / "media"
    if out_dir.is_symlink() or media.is_symlink() or (media.exists() and not media.is_dir()):
        raise ValueError("Invalid output directory")
    files = [payload, *(media.iterdir() if media.exists() else ())]
    if patch.exists() or patch.is_symlink():
        files.append(patch)
    if any(path.is_symlink() or not path.is_file() or path.name.startswith(".") for path in files):
        raise ValueError("Expected regular, non-hidden output files")
    if payload.stat().st_size > 1024 * 1024 or sum(p.stat().st_size for p in files) > 200 * 2**20:
        raise ValueError("Resolution output exceeds the artifact size limit")
    command(
        "uvx",
        "check-jsonschema@0.37.4",
        "--schemafile",
        ".claude/skills/resolve/payload.schema.yml",
        payload,
    )
    responses = json.loads(payload.read_text(), object_pairs_hook=unique_keys)["responses"]
    context = json.loads(context_path.read_text())
    thread_ids = review_thread_ids(context)
    optional = {"review"} if current_review(context)["body"].strip() else set()
    if not thread_ids <= responses.keys() <= thread_ids | optional:
        raise ValueError("Responses must cover every thread and only a nonempty review body")
    if patch.exists() != any(item["status"] == "resolved" for item in responses.values()):
        raise ValueError("A patch must accompany resolved feedback")
    command(
        *media_command(media), "--check", input="\n\n".join(r["body"] for r in responses.values())
    )


def media_command(media: Path) -> tuple[str | Path, ...]:
    return (
        "uv",
        "run",
        "--locked",
        "--no-default-groups",
        "--package",
        "skills",
        "skills",
        "embed-media",
        "--dir",
        media,
    )


def embed(out_dir: Path) -> None:
    payload = out_dir / "payload.json"
    result = json.loads(payload.read_text())
    for response in result["responses"].values():
        if str(out_dir / "media") in response["body"]:
            response["body"] = command(
                *media_command(out_dir / "media"),
                "--repository-id",
                os.environ["REPOSITORY_ID"],
                input=response["body"],
            )
    (out_dir / "publish.json").write_text(json.dumps(result))


def git(*args: str | Path) -> str:
    return command(
        "git", "-C", Path(os.environ["PR_CHECKOUT"]), "-c", "core.hooksPath=/dev/null", *args
    ).strip()


def apply_patch(patch: Path) -> None:
    git("apply", "--index", "--whitespace=error", "--", patch)
    paths = git("diff", "--cached", "--name-only", "--no-renames", "-z").split("\0")[:-1]
    if not paths:
        raise ValueError("Patch is empty")
    lines = (line.strip() for line in PATCH_PATHS.read_text().splitlines())
    allowed = tuple(line for line in lines if line and not line.startswith("#"))
    if outside := [path for path in paths if not path.startswith(allowed)]:
        raise ValueError(f"Patch changes paths outside {PATCH_PATHS.name}: {', '.join(outside)}")
    staged = git("ls-files", "--stage", "-z", "--", *paths).split("\0")[:-1]
    if any(not entry.startswith(("100644 ", "100755 ")) for entry in staged):
        raise ValueError("Patch contains a symlink or submodule")


def update_branch(repo: str, review_id: str, context: dict[str, Any], head: str, tree: str) -> str:
    base = context["head_sha"]
    trailer = f"Resolve-Review: {review_id}"
    if head != base:
        previous = api(f"repos/{repo}/git/commits/{head}")
        if (
            trailer in previous["message"].splitlines()
            and [parent["sha"] for parent in previous["parents"]] == [base]
            and previous["tree"]["sha"] == tree
        ):
            return head
        raise ValueError("PR head changed before publication")
    git(
        "-c",
        f"user.name={APP}",
        "-c",
        f"user.email={APP_ID}+{APP}@users.noreply.github.com",
        "commit",
        "-s",
        "-m",
        "Address PR review feedback",
        "-m",
        trailer,
    )
    ref = f"refs/heads/{context['head_ref']}"
    git(
        "-c",
        "credential.helper=",
        "-c",
        "credential.helper=!gh auth git-credential",
        "push",
        f"--force-with-lease={ref}:{base}",
        f"{os.environ['GITHUB_SERVER_URL']}/{repo}.git",
        f"HEAD:{ref}",
    )
    return git("rev-parse", "HEAD")


def publish(repo: str, number: str, review_id: str, context_path: Path, out_dir: Path) -> None:
    context = json.loads(context_path.read_text())
    dry_run = os.environ["DRY_RUN"] == "true"
    file = "payload.json" if dry_run else "publish.json"
    responses = json.loads((out_dir / file).read_text())["responses"]
    patch = out_dir / "fix.patch"
    tree = git("write-tree") if patch.is_file() else None
    if dry_run:
        details = json.dumps(responses, indent=2)
        diff = html.escape(patch.read_text(errors="replace")[:60_000]) if tree else ""
        details = html.escape(details[:60_000])
        closures = ", ".join(
            key
            for key, item in responses.items()
            if key != "review" and item["status"] == "resolved"
        )
        summary(
            f"### Resolution preview for {current_review(context)['author']}\n\n"
            f"<pre>{details}</pre>\n\n<pre>{diff}</pre>\n\n"
            f"**Threads to close**\n\n{html.escape(closures) or 'None'}"
        )
        return

    endpoint = f"repos/{repo}/pulls/{number}"
    pr = api(endpoint)
    review = api(f"{endpoint}/reviews/{review_id}")
    if (
        not eligible(pr, repo)
        or pr["head"]["ref"] != context["head_ref"]
        or review["state"] not in {"APPROVED", "CHANGES_REQUESTED", "COMMENTED"}
        or (review["user"] or {}).get("login") != current_review(context)["author"]
    ):
        raise ValueError("PR or review changed before publication")
    head = pr["head"]["sha"]
    if tree:
        head = update_branch(repo, review_id, context, head, tree)
    elif head != context["head_sha"]:
        raise ValueError("PR head changed before publication")

    issue_comments = f"repos/{repo}/issues/{number}/comments"
    previous = list_items(f"{endpoint}/comments")
    if "review" in responses:
        previous += list_items(issue_comments)
    posted = {comment["body"].rpartition("\n")[2] for comment in previous if app_comment(comment)}
    # Threads may have been resolved since the context was collected.
    latest = {"timeline": fetch_timeline(repo, number, review_id)}
    closed = {thread["id"] for thread in all_threads(latest) if thread["resolved"]}
    current = current_review(context)
    roots = {thread["id"]: thread["comments"][0]["id"] for thread in all_threads(context)}
    blocked = False
    for key, response in responses.items():
        # Blocked replies tell the reviewer what input is needed; their threads stay open.
        if response["status"] == "blocked":
            blocked = True
            summary(f"<p>Blocked {html.escape(key)}: {html.escape(response['body'])}</p>")
        note = marker(current["author"], review_id, key)
        resolved = response["status"] == "resolved"
        if note not in posted:
            body = response["body"]
            if resolved:
                body += f"\n\n[Fix commit]({os.environ['GITHUB_SERVER_URL']}/{repo}/commit/{head})."
            if key == "review":
                url = current["url"]
                body = f"In response to [this review]({url})\n\n{body}\n\n{note}"
                api(issue_comments, data={"body": body})
            else:
                api(f"{endpoint}/comments/{roots[key]}/replies", data={"body": f"{body}\n\n{note}"})
        if key != "review" and resolved and key not in closed:
            result = api("graphql", "-f", f"query={CLOSE_THREAD}", "-f", f"id={key}")
            if not result["data"]["resolveReviewThread"]["thread"]["isResolved"]:
                raise ValueError("GitHub did not close the resolved thread")

    # Blocked items need a human, so the review loop stops here.
    if blocked:
        return
    pr = api(endpoint)
    if not eligible(pr, repo) or pr["head"]["sha"] != head:
        raise ValueError("PR head changed before the review request")
    reviewer = current["author"]
    # NaiLaOpus reviews every pushed fix, whoever asked for it. After a NaiLaOpus round
    # that only answered its threads, it re-checks them so it can approve. It can't be
    # requested as a reviewer, so a `/review` comment starts it.
    if tree or (reviewer == NAILAOPUS and review["state"] != "APPROVED"):
        note = marker(reviewer, review_id, "review-request")
        if not any(
            app_comment(comment) and note in comment["body"]
            for comment in list_items(issue_comments)
        ):
            api(issue_comments, data={"body": f"/review\n\n{note}"})
            summary("Requested a NaiLaOpus review.")
    # Approvers need no re-request, and NaiLaOpus uses the comment above.
    if review["state"] == "APPROVED" or reviewer == NAILAOPUS:
        return
    requested = api(f"{endpoint}/requested_reviewers")["users"]
    reviews = list_items(f"{endpoint}/reviews")
    if any(user["login"] == reviewer for user in requested) or any(
        (other["user"] or {}).get("login") == reviewer
        and other["id"] > int(review_id)
        and other["commit_id"] == head
        and other["state"] in {"APPROVED", "CHANGES_REQUESTED", "COMMENTED"}
        for other in reviews
    ):
        return
    api(
        f"{endpoint}/requested_reviewers/rerequest",
        "-H",
        "X-GitHub-Api-Version: 2026-03-10",
        data={"reviewers": [reviewer]},
    )
    summary(f"Re-requested review from {reviewer}.")


def main() -> None:
    context_path = Path(os.environ["CONTEXT_PATH"])
    out_dir = Path(os.environ["OUT_DIR"])
    match sys.argv[1:]:
        case ["validate"]:
            validate(context_path, out_dir)
            if "PR_CHECKOUT" in os.environ:
                if git("rev-parse", "HEAD") != json.loads(context_path.read_text())["head_sha"]:
                    raise ValueError("Publisher checkout does not match the collected head")
                if (patch := out_dir / "fix.patch").is_file():
                    apply_patch(patch)
        case ["embed"]:
            embed(out_dir)
        case [operation] if operation in {"collect", "publish"}:
            repo = os.environ["GH_REPO"]
            number = os.environ["PR_NUMBER"]
            review_id = os.environ["REVIEW_ID"]
            if repo != "mlflow/mlflow" or not number.isdecimal() or not review_id.isdecimal():
                raise ValueError("Expected an MLflow PR and numeric review ID")
            if operation == "collect":
                collect(repo, number, review_id, context_path)
            else:
                publish(repo, number, review_id, context_path, out_dir)
        case _:
            raise ValueError("Expected collect, validate, embed, or publish")


if __name__ == "__main__":
    main()
