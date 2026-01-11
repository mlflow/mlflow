from typing import Any

import aiohttp
from typing_extensions import Self

from skills.github.types import PullRequest
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

    async def _get_json(self, endpoint: str) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("GitHubClient must be used as async context manager")
        async with self._session.get(endpoint) as resp:
            resp.raise_for_status()
            return await resp.json()

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
            return await resp.json()
