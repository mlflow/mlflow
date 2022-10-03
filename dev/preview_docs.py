import os
import requests
import argparse
import time
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--issue-number", required=True)
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"}
    session = requests.Session()
    session.headers.update(headers)

    # Get the ID of the build_doc job
    repo = "mlflow/mlflow"
    build_doc_job_name = "build_doc"
    job_id = None
    for _ in range(5):
        resp = session.get(f"https://api.github.com/repos/{repo}/commits/{args.ref}/status")
        resp.raise_for_status()
        build_doc_status = next(
            filter(lambda s: s["context"].endswith(build_doc_job_name), resp.json()["statuses"]),
            None,
        )
        if build_doc_status:
            job_id = urlparse(build_doc_status["target_url"]).path.split("/")[-1]
            break
        print(f"Waiting for {build_doc_job_name} job status to be available...")
        time.sleep(3)
    else:
        print(f"Could not find {build_doc_job_name} job status")
        return

    # Get the artifact URL of the top level index.html
    resp = session.get(f"https://circleci.com/api/v2/project/gh/{repo}/job/{job_id}")
    resp.raise_for_status()
    job_data = resp.json()
    job_url = job_data["web_url"]
    workflow_id = job_data["latest_workflow"]["id"]
    resp = session.get(f"https://circleci.com/api/v2/workflow/{workflow_id}/job")
    resp.raise_for_status()
    workflow_job = next(filter(lambda s: s["name"] == build_doc_job_name, resp.json()["items"]))
    workflow_job_id = workflow_job["id"]
    artifact_url = f"https://output.circle-artifacts.com/output/job/{workflow_job_id}/artifacts/0/docs/build/html/index.html"
    print(f"Artifact URL: {artifact_url}")

    # Post the artifact URL as a comment
    resp = session.get(f"https://api.github.com/repos/{repo}/issues/{args.issue_number}/comments")
    resp.raise_for_status()
    marker = "<!-- documentation preview -->"
    preview_docs_comment = next(filter(lambda c: marker in c["body"], resp.json()), None)
    comment_body = f"""
{marker}
### Documentation preview will be available [here]({artifact_url}).

<details>
<summary>Notes</summary>

- It takes a few minutes for the preview to be available.
- The preview is updated on every commit to the PR.
- Job URL: {job_url}

</details>
"""
    if preview_docs_comment is None:
        print("Creating comment")
        resp = session.post(
            f"https://api.github.com/repos/{repo}/issues/{args.issue_number}/comments",
            json={"body": comment_body},
        )
        resp.raise_for_status()
    else:
        print("Updating comment")
        resp = session.patch(preview_docs_comment["url"], json={"body": comment_body})
        resp.raise_for_status()


if __name__ == "__main__":
    main()
