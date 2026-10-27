import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

REPOSITORY = "mlflow/mlflow"


def post(endpoint: str, data: Any = None) -> dict[str, Any]:
    result = subprocess.run(
        ["gh", "api", f"repos/{REPOSITORY}/{endpoint}", "--method", "POST", "--input", "-"],
        input=json.dumps(data or {}),
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    return cast(dict[str, Any], json.loads(result.stdout)) if result.stdout.strip() else {}


def issue_body(group: dict[str, Any]) -> str:
    # Triage treats issues from mlflow-app with this frontmatter as Nightly failures.
    frontmatter = "<!-- triage\ntype: nightly\n-->"
    body = f"{frontmatter}\n\n{group['summary']}\n\n### Evidence\n\n{group['evidence']}"
    jobs = "\n".join(
        f"- [{job['workflow_name']} / {job['job_name']}]({job['job_run_url']})"
        for job in group["jobs"]
    )
    return f"{body}\n\n### Failed jobs\n\n{jobs}"


def publish(analysis: dict[str, Any], *, dry_run: bool) -> None:
    new_count = 0
    for group in analysis["groups"]:
        if group["verdict"] != "new":
            sys.stdout.write(f"{group['verdict']}: {group['summary']}\n")
            continue
        body = issue_body(group)
        new_count += 1
        if dry_run:
            sys.stdout.write(f"Would create issue: {group['title']}\n\n{body}\n\n")
        else:
            number = post("issues", {"title": group["title"], "body": body, "labels": ["nightly"]})[
                "number"
            ]
            sys.stdout.write(f"Created https://github.com/{REPOSITORY}/issues/{number}\n")
    if new_count == 0:
        sys.stdout.write("No new nightly issues.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print or create nightly issues")
    parser.add_argument("analysis", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print issue drafts without creating issues"
    )
    args = parser.parse_args()
    publish(json.loads(args.analysis.read_text()), dry_run=args.dry_run)
