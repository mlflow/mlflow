import asyncio
import json
import re
import subprocess
import sys
import traceback
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
from pydantic import BaseModel


def get_cooldown_days() -> int:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if m := re.search(r'^exclude-newer\s*=\s*"P(\d+)D"\s*$', pyproject.read_text(), re.MULTILINE):
        return int(m.group(1))
    raise RuntimeError(f'`exclude-newer = "P<N>D"` not found in {pyproject}')


class ReleaseFile(BaseModel):
    upload_time_iso_8601: str


class PyPIResponse(BaseModel):
    releases: dict[str, list[ReleaseFile]]


def get_distributions() -> list[tuple[str, str]]:
    res = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format", "json"])
    return [(pkg["name"], pkg["version"]) for pkg in json.loads(res)]


def extract_upload_time(response: PyPIResponse, version: str) -> datetime | None:
    for f in response.releases.get(version, []):
        return datetime.fromisoformat(f.upload_time_iso_8601.replace("Z", "+00:00"))
    return None


async def get_release_date(
    session: aiohttp.ClientSession, package: str, version: str
) -> datetime | None:
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


def format_age_past_cooldown(release_date: datetime | None, cooldown_days: int) -> str:
    if release_date is None:
        return ""
    delta = datetime.now(timezone.utc) - release_date - timedelta(days=cooldown_days)
    return f"{delta.total_seconds() / 86400:.1f}d"


async def main() -> None:
    cooldown_days = get_cooldown_days()
    distributions = get_distributions()
    async with aiohttp.ClientSession() as session:
        tasks = [get_release_date(session, pkg, ver) for pkg, ver in distributions]
        results = await asyncio.gather(*tasks)

    ages = [format_age_past_cooldown(r, cooldown_days) for r in results]
    packages, versions = list(zip(*distributions))
    age_header = f"Age past {cooldown_days}d cooldown"
    package_length = get_longest_string_length(packages)
    version_length = get_longest_string_length(versions)
    age_length = max(get_longest_string_length(ages), len(age_header))
    print(
        "Package".ljust(package_length),
        "Version".ljust(version_length),
        age_header.ljust(age_length),
    )
    print("-" * (package_length + version_length + age_length + 2))
    for package, version, age, _ in sorted(
        zip(packages, versions, ages, results),
        # Newest first; entries without a release date sort to the end.
        key=lambda x: (x[3] is None, -x[3].timestamp() if x[3] else 0),
    ):
        print(
            package.ljust(package_length),
            version.ljust(version_length),
            age.ljust(age_length),
        )


if __name__ == "__main__":
    asyncio.run(main())
