# /// script
# requires-python = "==3.10"
# dependencies = [
#   "requests",
# ]
# ///
# ruff: noqa: T201
import os
import subprocess
import sys
import uuid

import requests


def main():
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
    if not GITHUB_TOKEN:
        print("GITHUB_TOKEN not found in environment variables", file=sys.stderr)
        sys.exit(1)

    OWNER = "mlflow"
    REPO = "mlflow"
    MLFLOW_3_BRANCH_NAME = "mlflow-3"
    PR_TITLE = "Sync with master"
    PR_BRANCH_NAME = f"sync-{uuid.uuid4()}"

    # Exit if there is already a PR with the same title
    def iter_pull_requests():
        per_page = 100
        page = 1
        while True:
            resp = requests.get(
                f"https://api.github.com/repos/{OWNER}/{REPO}/pulls",
                headers={"Authorization": f"token {GITHUB_TOKEN}"},
                params={
                    "state": "open",
                    "base": MLFLOW_3_BRANCH_NAME,
                    "per_page": per_page,
                    "page": page,
                },
            )
            resp.raise_for_status()
            prs = resp.json()
            yield from prs
            if len(prs) < per_page:
                break
            page += 1

    if pr := next((pr for pr in iter_pull_requests() if pr["title"] == PR_TITLE), None):
        print(f"PR already exists: {pr['html_url']}")
        sys.exit(0)

    # Fetch master and mlflow-3 branches
    subprocess.check_call(["git", "config", "user.name", "mlflow-automation"])
    subprocess.check_call(
        ["git", "config", "user.email", "mlflow-automation@users.noreply.github.com"]
    )
    subprocess.check_call(["git", "fetch", "origin", "master"])
    subprocess.check_call(["git", "fetch", "origin", MLFLOW_3_BRANCH_NAME])

    # Sync mlflow-3 branch with master
    subprocess.check_call(
        ["git", "checkout", "-b", PR_BRANCH_NAME, f"origin/{MLFLOW_3_BRANCH_NAME}"]
    )
    prc = subprocess.run(["git", "merge", "origin/master"])
    if prc.returncode != 0:
        print(
            (
                "Merge failed, possibly due to conflicts. "
                "Follow the instructions in `sync.md` to resolve the conflicts "
                "and create a PR manually."
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    # Create a pull request
    subprocess.check_call(["git", "push", "origin", PR_BRANCH_NAME])
    pr = requests.post(
        f"https://api.github.com/repos/{OWNER}/{REPO}/pulls",
        headers={"Authorization": f"token {GITHUB_TOKEN}"},
        json={
            "title": PR_TITLE,
            "head": PR_BRANCH_NAME,
            "base": MLFLOW_3_BRANCH_NAME,
            "body": "This PR was created automatically by the sync workflow.",
        },
    )
    pr.raise_for_status()
    pr_data = pr.json()
    print(f"PR created: {pr_data['html_url']}")


if __name__ == "__main__":
    main()
