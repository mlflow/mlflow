import argparse
import html
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

SKILL_DIR = Path(__file__).resolve().parents[1]
SKILLS_DIR = SKILL_DIR.parent
SCHEMA = SKILL_DIR / "payload.schema.yml"
TRUSTED_BOTS = {"Copilot", "mlflow-app[bot]"}
PR_CONTEXT = SKILLS_DIR.parent / "scripts" / "pr_context.py"
VALIDATE_REVIEW = SKILLS_DIR.parent / "scripts" / "validate_review.py"


def command(*args: str | Path, input: str | None = None) -> str:
    return subprocess.check_output(args, input=input, text=True, stderr=subprocess.PIPE)


def api(*args: str, data: dict[str, Any] | None = None) -> Any:
    if data is not None:
        args += ("--method", "POST", "--input", "-")
    result = command("gh", "api", *args, input=json.dumps(data) if data is not None else None)
    return json.loads(result) if result.strip() else None


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


def collect(args: argparse.Namespace) -> int:
    if not re.fullmatch(r"/verify\s*", args.comment, re.ASCII):
        sys.stdout.write("Not a /verify command.\n")
        return 0
    context = json.loads(
        command(sys.executable, PR_CONTEXT, "--repo", args.repo, "--pr-number", args.pr_number)
    )
    if context["state"] != "open":
        raise ValueError("PR is not open")
    author = context["author"]
    match author["type"]:
        case "User":
            permission = api(f"repos/{args.repo}/collaborators/{author['login']}/permission")
            if permission["permission"] not in {"admin", "write"}:
                write_step_summary("The PR author does not have write access.")
                return 0
        case "Bot" if author["login"] in TRUSTED_BOTS:
            pass
        case _:
            write_step_summary("The PR author is not trusted.")
            return 0
    head = context["head"]["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("Unexpected PR head SHA")
    args.out_path.write_text(json.dumps(context, indent=2) + "\n")
    write_step_outputs(head=head, allowed="true")
    return 0


def media_command(media_dir: Path) -> tuple[str | Path, ...]:
    return (
        "uv",
        "run",
        "--quiet",
        "--directory",
        SKILLS_DIR,
        "--package",
        "skills",
        "skills",
        "embed-media",
        "--dir",
        media_dir,
    )


def validate(args: argparse.Namespace) -> int:
    # The diff stays on the verification runner, so the publisher skips the position check.
    diff_args = () if args.diff_path is None else ("--diff", args.diff_path)
    command(
        sys.executable,
        VALIDATE_REVIEW,
        "--schema",
        SCHEMA,
        "--payload",
        args.payload_path,
        *diff_args,
    )
    payload = json.loads(args.payload_path.read_text())
    command(
        *media_command(args.media_dir),
        "--check",
        input="\n\n".join(comment["body"] for comment in payload["comments"]),
    )
    sys.stdout.write("Review payload is valid.\n")
    return 0


def embed(args: argparse.Namespace) -> int:
    payload = json.loads(args.payload_path.read_text())
    for comment in payload["comments"]:
        comment["body"] = command(
            *media_command(args.media_dir),
            "--repository-id",
            os.environ["REPOSITORY_ID"],
            input=comment["body"],
        )
    args.payload_path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


def publish(args: argparse.Namespace) -> int:
    endpoint = f"repos/{args.repo}/pulls/{args.pr_number}"
    pr = api(endpoint)
    if pr["state"] != "open":
        write_step_summary("PR closed; no review will be posted.")
        return 0
    notice = f"> [Verification run]({args.run_url})."
    # Without a payload, verification did not produce a publishable review.
    if args.payload_path is not None:
        payload = json.loads(args.payload_path.read_text())
        review = {
            "commit_id": args.head_sha,
            "event": payload["event"],
            "body": f"{notice}\n\n{payload['body']}",
            "comments": payload["comments"],
        }
    else:
        review = {
            "commit_id": args.head_sha,
            "event": "COMMENT",
            "body": f"{notice}\n\nVerification did not complete. See the workflow run for details.",
        }
    # An approval enables auto-merge unless someone has yet to review the PR.
    auto_merge = (
        review["event"] == "APPROVE"
        and not pr["draft"]
        and not pr["requested_reviewers"]
        and not pr["requested_teams"]
    )
    if args.dry_run:
        preview = html.escape(json.dumps(review, indent=2, ensure_ascii=False))
        write_step_summary(
            "### Verification preview\n"
            "No review or media was published. Local media paths are not GitHub links.\n"
            f"<pre>{preview}</pre>\n\n"
            f"Auto-merge would {'be enabled' if auto_merge else 'not be enabled'}."
        )
    else:
        sys.stdout.write(api(f"{endpoint}/reviews", data=review)["html_url"] + "\n")
        if auto_merge:
            number = args.pr_number
            # Fails if the head moved since verification.
            command(
                "gh", "pr", "merge", number, "--repo", args.repo,
                "--auto", "--squash", "--match-head-commit", args.head_sha,
            )  # fmt: skip
            command("gh", "pr", "edit", number, "--repo", args.repo, "--add-label", "auto-repair")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify workflow commands")
    commands = parser.add_subparsers(dest="command", required=True)
    collection = commands.add_parser("collect", help="Authorize the request and snapshot the PR")
    collection.add_argument("--repo", required=True)
    collection.add_argument("--pr-number", required=True)
    collection.add_argument("--comment", required=True)
    collection.add_argument("--out-path", type=Path, required=True)
    collection.set_defaults(func=collect)
    validation = commands.add_parser("validate", help="Validate a review payload")
    validation.add_argument("--payload", dest="payload_path", type=Path, required=True)
    validation.add_argument("--media-dir", type=Path, required=True)
    validation.add_argument("--diff", dest="diff_path", type=Path)
    validation.set_defaults(func=validate)
    embedding = commands.add_parser("embed", help="Upload cited media and rewrite its links")
    embedding.add_argument("--payload", dest="payload_path", type=Path, required=True)
    embedding.add_argument("--media-dir", type=Path, required=True)
    embedding.set_defaults(func=embed)
    publication = commands.add_parser("publish", help="Post or preview the review")
    publication.add_argument("--repo", required=True)
    publication.add_argument("--pr-number", required=True)
    publication.add_argument("--head-sha", required=True)
    publication.add_argument("--run-url", required=True)
    publication.add_argument("--payload", dest="payload_path", type=Path)
    publication.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    publication.set_defaults(func=publish)
    args = parser.parse_args()
    handler: Callable[[argparse.Namespace], int] = args.func
    try:
        return handler(args)
    except subprocess.CalledProcessError as error:
        sys.stderr.write((error.stdout or "") + (error.stderr or str(error)) + "\n")
        return 1
    except (RuntimeError, ValueError, KeyError, TypeError, OSError) as error:
        sys.stderr.write(f"Verify failed: {error}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
