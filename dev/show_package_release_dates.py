import asyncio
import json
import subprocess
import sys
import traceback

import aiohttp


def get_distributions() -> list[tuple[str, str]]:
    res = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--format", "json"], text=True
    )
    return [(pkg["name"], pkg["version"]) for pkg in json.loads(res)]


async def get_release_date(session: aiohttp.ClientSession, package: str, version: str) -> str:
    try:
        async with session.get(f"https://pypi.python.org/pypi/{package}/json", timeout=10) as resp:
            if resp.status != 200:
                return ""

            resp_json = await resp.json()
            matched = [
                dist_files for ver, dist_files in resp_json["releases"].items() if ver == version
            ]
            if not matched or not matched[0]:
                return ""

            upload_time = matched[0][0]["upload_time"]
            return upload_time.replace("T", " ")  # return year-month-day hour:minute:second
    except Exception:
        traceback.print_exc()
        return ""


def get_longest_string_length(array: list[str]) -> int:
    return len(max(array, key=len))


async def main() -> None:
    distributions = get_distributions()
    async with aiohttp.ClientSession() as session:
        tasks = [get_release_date(session, pkg, ver) for pkg, ver in distributions]
        release_dates = await asyncio.gather(*tasks)

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
