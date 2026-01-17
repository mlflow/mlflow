import asyncio
import json
import subprocess
import sys
import traceback
from collections.abc import Sequence

import aiohttp
from pydantic import BaseModel


class ReleaseFile(BaseModel):
    upload_time: str


class PyPIResponse(BaseModel):
    releases: dict[str, list[ReleaseFile]]


def get_distributions() -> list[tuple[str, str]]:
    res = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--format", "json"], text=True
    )
    return [(pkg["name"], pkg["version"]) for pkg in json.loads(res)]


def extract_upload_time(response: PyPIResponse, version: str) -> str | None:
    for f in response.releases.get(version, []):
        return f.upload_time.replace("T", " ")
    return None


async def get_release_date(
    session: aiohttp.ClientSession, package: str, version: str
) -> str | None:
    try:
        async with session.get(f"https://pypi.python.org/pypi/{package}/json", timeout=10) as resp:
            resp.raise_for_status()
            resp_json = await resp.json()
            response = PyPIResponse.model_validate(resp_json)
            return extract_upload_time(response, version)

    except Exception:
        traceback.print_exc()
        return None


def get_longest_string_length(array: Sequence[str]) -> int:
    return len(max(array, key=len))


async def main() -> None:
    distributions = get_distributions()
    async with aiohttp.ClientSession() as session:
        tasks = [get_release_date(session, pkg, ver) for pkg, ver in distributions]
        results = await asyncio.gather(*tasks)

    release_dates = [r or "" for r in results]
    packages, versions = list(zip(*distributions))
    package_length = get_longest_string_length(packages)
    version_length = get_longest_string_length(versions)
    release_timestamp_length = get_longest_string_length(release_dates)
    print(
        "Package".ljust(package_length),
        "Version".ljust(version_length),
        "Release Timestamp".ljust(release_timestamp_length),
    )
    print("-" * (package_length + version_length + release_timestamp_length + 2))
    for package, version, release_date in sorted(
        zip(packages, versions, release_dates),
        # Sort by release date in descending order
        key=lambda x: x[2],
        reverse=True,
    ):
        print(
            package.ljust(package_length),
            version.ljust(version_length),
            release_date.ljust(release_timestamp_length),
        )


if __name__ == "__main__":
    asyncio.run(main())
