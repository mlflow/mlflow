import argparse
import os

import requests


class Session(requests.Session):
    def get(self, *args, **kwargs):
        resp = super().get(*args, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def post(self, *args, **kwargs):
        resp = super().post(*args, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def patch(self, *args, **kwargs):
        resp = super().patch(*args, **kwargs)
        resp.raise_for_status()
        return resp.json()


MARKER = "<!-- documentation preview -->"


def upsert_comment(session, repo, pull_number, comment_body):
    comments = session.get(f"https://api.github.com/repos/{repo}/issues/{pull_number}/comments")
    preview_docs_comment = next(filter(lambda c: MARKER in c["body"], comments), None)
    comment_body_with_marker = MARKER + "\n\n" + comment_body
    if preview_docs_comment is None:
        print("Creating comment")
        session.post(
            f"https://api.github.com/repos/{repo}/issues/{pull_number}/comments",
            json={"body": comment_body_with_marker},
        )
    else:
        print("Updating comment")
        session.patch(preview_docs_comment["url"], json={"body": comment_body_with_marker})


def _get_comment_template(
    commit_sha: str, workflow_run_link: str, docs_workflow_run_url: str, main_message: str
) -> str:
    return f"""
Documentation preview for {commit_sha} {main_message}

<details>
<summary>More info</summary>

- Ignore this comment if this PR does not change the documentation.
- It takes a few minutes for the preview to be available.
- The preview is updated when a new commit is pushed to this PR.
- This comment was created by [this workflow run]({workflow_run_link}).
- The documentation was built by [this workflow run]({docs_workflow_run_url}).

</details>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--pull-number", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--stage", choices=["completed", "failed"], required=True)
    parser.add_argument("--netlify-url", required=False)
    parser.add_argument("--docs-workflow-run-url", required=True)
    args = parser.parse_args()

    github_session = Session()
    if github_token := os.environ.get("GITHUB_TOKEN"):
        github_session.headers.update({"Authorization": f"token {github_token}"})

    repo = "mlflow/mlflow"
    workflow_run_link = f"https://github.com/{repo}/actions/runs/{args.workflow_run_id}"

    if args.stage == "completed":
        if not args.netlify_url:
            raise ValueError("netlify-url is required for completed stage")
        main_message = f"is available at:\n\n- {args.netlify_url}"
        comment_body = _get_comment_template(
            args.commit_sha, workflow_run_link, args.docs_workflow_run_url, main_message
        )
        upsert_comment(github_session, repo, args.pull_number, comment_body)

    elif args.stage == "failed":
        main_message = "failed to build or deploy."
        comment_body = _get_comment_template(
            args.commit_sha, workflow_run_link, args.docs_workflow_run_url, main_message
        )
        upsert_comment(github_session, repo, args.pull_number, comment_body)


# TODO: rewrite this in JavaScript so we don't have to setup both node (to deploy to netlify)
# and python (to upsert pr comments with this script)
if __name__ == "__main__":
    main()
