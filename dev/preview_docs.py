import argparse
import os
import subprocess
import tempfile
import time
import zipfile
from urllib.parse import urlparse

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


def deploy_to_netlify(artifact_url: str, pull_number: int, site_name: str, action_url: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = os.path.join(tmpdir, "docs-html.zip")
        response = requests.get(artifact_url, stream=True)
        response.raise_for_status()
        with open(artifact_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(artifact_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        build_dir = os.path.join(tmpdir, "build")
        if not os.path.isdir(build_dir):
            raise Exception(f"'build' directory not found in the artifact: {os.listdir(tmpdir)}")

        alias = f"pr-{pull_number}"
        message = f"PR Preview #{pull_number} - GitHub Action: {action_url}"
        command = [
            "netlify",
            "deploy",
            "--dir",
            build_dir,
            "--alias",
            alias,
            "--no-build",
            "--message",
            message,
        ]
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        return f"https://{alias}--{site_name}.netlify.app"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--pull-number", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--netlify-site-name", required=True)
    parser.add_argument("--action-url", required=True)
    parser.add_argument("--netlify-site-name", required=True)
    parser.add_argument("--action-url", required=True)
    args = parser.parse_args()

    github_session = Session()
    if github_token := os.environ.get("GITHUB_TOKEN"):
        github_session.headers.update({"Authorization": f"token {github_token}"})

    circle_session = Session()
    if circle_token := os.environ.get("CIRCLE_TOKEN"):
        circle_session.headers.update({"Circle-Token": circle_token})

    # Get the ID of the build_doc job
    repo = "mlflow/mlflow"
    build_doc_job_name = "build_doc"
    job_id = None
    job_url = None
    workflow_run_link = f"https://github.com/{repo}/actions/runs/{args.workflow_run_id}"
    for _ in range(5):
        status = github_session.get(
            f"https://api.github.com/repos/{repo}/commits/{args.commit_sha}/status"
        )
        build_doc_status = next(
            filter(lambda s: s["context"].endswith(build_doc_job_name), status["statuses"]),
            None,
        )
        if build_doc_status:
            job_id = urlparse(build_doc_status["target_url"]).path.split("/")[-1]
            job_url = build_doc_status["target_url"]
            break
        print(f"Waiting for {build_doc_job_name} job status to be available...")
        time.sleep(3)
    else:
        print(f"Could not find {build_doc_job_name} job status")
        comment_body = f"""
Failed to find a documentation preview for {args.commit_sha}.

<details>
<summary>More info</summary>

- If the `ci/circleci: {build_doc_job_name}` job status is successful, you can see the preview with
  the following steps:
  1. Click `Details`.
  2. Click `Artifacts`.
  3. Click `docs/build/html/index.html`.
- This comment was created by {workflow_run_link}.

</details>
"""
        upsert_comment(github_session, repo, args.pull_number, comment_body)
        return

    # Post initial comment with CircleCI job link
    initial_comment_body = f"""
Documentation preview for {args.commit_sha} will be available when [this CircleCI job]({job_url})
completes successfully.

<details>
<summary>More info</summary>

- Ignore this comment if this PR does not change the documentation.
- It takes a few minutes for the preview to be available.
- The preview is updated when a new commit is pushed to this PR.
- This comment was created by {workflow_run_link}.

</details>
"""
    upsert_comment(github_session, repo, args.pull_number, initial_comment_body)

    # Wait for the build_doc job to complete
    print("Waiting for CircleCI job to complete...")
    for _ in range(60):  # Wait up to 3 minutes (60 * 3 seconds)
        status = github_session.get(
            f"https://api.github.com/repos/{repo}/commits/{args.commit_sha}/status"
        )
        build_doc_status = next(
            filter(lambda s: s["context"].endswith(build_doc_job_name), status["statuses"]),
            None,
        )
        if build_doc_status and build_doc_status["state"] == "success":
            print("CircleCI job completed successfully")
            break
        elif build_doc_status and build_doc_status["state"] == "failure":
            print("CircleCI job failed")
            failure_comment_body = f"""
Documentation preview for {args.commit_sha} failed to build.

The [CircleCI job]({job_url}) failed. Please check the job logs for more details.

<details>
<summary>More info</summary>

- This comment was created by {workflow_run_link}.

</details>
"""
            upsert_comment(github_session, repo, args.pull_number, failure_comment_body)
            return
        print(f"Job status: {build_doc_status['state'] if build_doc_status else 'not found'}, waiting...")
        time.sleep(3)
    else:
        print("Timed out waiting for CircleCI job to complete")
        timeout_comment_body = f"""
Documentation preview for {args.commit_sha} is taking longer than expected to build.

The [CircleCI job]({job_url}) is still running. Please check back later or check the job directly.

<details>
<summary>More info</summary>

- This comment was created by {workflow_run_link}.

</details>
"""
        upsert_comment(github_session, repo, args.pull_number, timeout_comment_body)
        return

    # Get CircleCI job details and deploy to Netlify
    for _ in range(5):
        try:
            # Despite using a valid CircleCI token, the request occasionally fails with a 403 error.
            job = circle_session.get(f"https://circleci.com/api/v2/project/gh/{repo}/job/{job_id}")
            workflow_id = job["latest_workflow"]["id"]
            workflow = circle_session.get(f"https://circleci.com/api/v2/workflow/{workflow_id}/job")
            break
        except requests.HTTPError as e:
            print(
                f"Failed to get CircleCI job info: {e.response.status_code, e.response.text}, "
                f"retrying..."
            )
            time.sleep(1)
            continue
    else:
        upsert_comment(
            github_session,
            repo,
            args.pull_number,
            (
                f"Failed to find a documentation preview for {args.commit_sha}. "
                f"See {workflow_run_link} for what went wrong."
            ),
        )
        return

    build_doc_job = next(filter(lambda s: s["name"] == build_doc_job_name, workflow["items"]))
    build_doc_job_id = build_doc_job["id"]
    artifact_url = f"https://output.circle-artifacts.com/output/job/{build_doc_job_id}/artifacts/0/docs-html.zip"

    # Deploy to Netlify and update comment with preview URL
    print("Deploying to Netlify...")
    try:
        netlify_url = deploy_to_netlify(
            artifact_url,
            args.pull_number,
            args.netlify_site_name,
            args.action_url,
        )
        final_comment_body = f"""
Documentation preview for {args.commit_sha} is available at:

- {netlify_url}

<details>
<summary>More info</summary>

- Ignore this comment if this PR does not change the documentation.
- The preview is updated when a new commit is pushed to this PR.
- This comment was created by {workflow_run_link}.

</details>
"""
        upsert_comment(github_session, repo, args.pull_number, final_comment_body)
        print(f"Documentation preview deployed successfully: {netlify_url}")
    except Exception as e:
        print(f"Failed to deploy to Netlify: {e}")
        deployment_error_comment_body = f"""
Documentation preview for {args.commit_sha} failed to deploy.

The [CircleCI job]({job_url}) completed successfully, but deployment to Netlify failed: {str(e)}

<details>
<summary>More info</summary>

- This comment was created by {workflow_run_link}.

</details>
"""
        upsert_comment(github_session, repo, args.pull_number, deployment_error_comment_body)


if __name__ == "__main__":
    main()
