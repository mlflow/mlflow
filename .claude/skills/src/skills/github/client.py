from collections.abc import AsyncIterator
from typing import Any, cast

import aiohttp
from typing_extensions import Self

from skills.github.types import Job, JobRun, PullRequest
from skills.github.utils import get_github_token


class GitHubClient:
    def __init__(self, token: str | None = None) -> None:
        self.token = token or get_github_token()
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> Self:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
        }
        self._session = aiohttp.ClientSession(
            base_url="https://api.github.com",
            headers=headers,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._session:
            await self._session.close()

    async def _get_json(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("GitHubClient must be used as async context manager")
        async with self._session.get(endpoint, params=params) as resp:
            resp.raise_for_status()
            return cast(dict[str, Any], await resp.json())

    async def _get_text(self, endpoint: str, accept: str) -> str:
        if self._session is None:
            raise RuntimeError("GitHubClient must be used as async context manager")
        headers = {"Accept": accept}
        async with self._session.get(endpoint, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.text()

    async def get_pr(self, owner: str, repo: str, pr_number: int) -> PullRequest:
        data = await self._get_json(f"/repos/{owner}/{repo}/pulls/{pr_number}")
        return PullRequest.model_validate(data)

    async def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        return await self._get_text(
            f"/repos/{owner}/{repo}/pulls/{pr_number}",
            accept="application/vnd.github.v3.diff",
        )

    async def get_compare_diff(self, owner: str, repo: str, base: str, head: str) -> str:
        return await self._get_text(
            f"/repos/{owner}/{repo}/compare/{base}...{head}",
            accept="application/vnd.github.v3.diff",
        )

    async def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("GitHubClient must be used as async context manager")
        payload = {"query": query, "variables": variables}
        async with self._session.post(
            "https://api.github.com/graphql",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            return cast(dict[str, Any], await resp.json())

    async def get_raw(self, endpoint: str) -> aiohttp.ClientResponse:
        """Get raw response for streaming."""
        if self._session is None:
            raise RuntimeError("GitHubClient must be used as async context manager")
        return await self._session.get(endpoint, allow_redirects=True)

    async def get_workflow_runs(
        self,
        owner: str,
        repo: str,
        head_sha: str | None = None,
        status: str | None = None,
    ) -> AsyncIterator[JobRun]:
        """Get workflow runs for a repository."""
        params: dict[str, Any] = {"per_page": 100}
        if head_sha:
            params["head_sha"] = head_sha
        if status:
            params["status"] = status

        page = 1
        while True:
            params["page"] = page
            data = await self._get_json(f"/repos/{owner}/{repo}/actions/runs", params)
            runs = data.get("workflow_runs", [])
            if not runs:
                break
            for run in runs:
                yield JobRun.model_validate(run)
            if len(runs) < 100:
                break
            page += 1

    async def get_jobs(self, owner: str, repo: str, run_id: int) -> AsyncIterator[Job]:
        """Get jobs for a workflow run."""
        page = 1
        while True:
            data = await self._get_json(
                f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
                {"per_page": 100, "page": page},
            )
            jobs = data.get("jobs", [])
            if not jobs:
                break
            for job in jobs:
                yield Job.model_validate(job)
            if len(jobs) < 100:
                break
            page += 1

    async def get_job(self, owner: str, repo: str, job_id: int) -> Job:
        """Get a specific job."""
        data = await self._get_json(f"/repos/{owner}/{repo}/actions/jobs/{job_id}")
        return Job.model_validate(data)

    async def get_job_run(self, owner: str, repo: str, run_id: int) -> JobRun:
        """Get a specific workflow run."""
        data = await self._get_json(f"/repos/{owner}/{repo}/actions/runs/{run_id}")
        return JobRun.model_validate(data)
