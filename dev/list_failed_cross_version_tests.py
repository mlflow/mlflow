"""
A script to list failed jobs in the latest scheduled run of the `cross-version-tests` workflow
using the GitHub Actions API.

References:
- https://docs.github.com/en/rest/reference/actions#list-workflow-runs
- https://docs.github.com/en/rest/reference/actions#list-jobs-for-a-workflow-run
"""
import json
import os
import requests


def fetch_failed_jobs(session, run_id):
    failed_jobs = []
    per_page = 100
    page = 1
    while True:
        url = f"https://api.github.com/repos/mlflow/mlflow/actions/runs/{run_id}/jobs"
        resp = session.get(url, params={"per_page": per_page, "page": page})
        jobs = resp.json()["jobs"]
        failed_jobs.extend(j for j in jobs if j["conclusion"] == "failure")
        if len(jobs) < per_page:
            break
        page += 1

    return failed_jobs


class StrictSession(requests.Session):
    """
    A wrapper class for `requests.Session` to validate the request is successful.
    """

    def request(self, *args, **kwargs):
        resp = super().request(*args, **kwargs)
        resp.raise_for_status()
        return resp


def create_session():
    session = StrictSession()
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        session.headers.update({"Authorization": f"token {token}"})
    return session


def main():
    session = create_session()
    url = (
        "https://api.github.com/repos/mlflow/mlflow/actions/workflows/cross-version-tests.yml/runs"
    )
    resp = session.get(url, params={"event": "schedule"})
    latest_run_id = resp.json()["workflow_runs"][0]["id"]
    failed_jobs = fetch_failed_jobs(session, latest_run_id)
    print(json.dumps(failed_jobs, indent=2))


if __name__ == "__main__":
    main()
