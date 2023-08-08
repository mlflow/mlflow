import os
import json
import sys
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor
import traceback


def get_distributions():
    res = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--format", "json"], text=True
    )
    return [(pkg["name"], pkg["version"]) for pkg in json.loads(res)]


def get_release_date(package, version):
    resp = requests.get(f"https://pypi.python.org/pypi/{package}/json", timeout=10)
    if not resp.ok:
        return ""

    matched = [dist_files for ver, dist_files in resp.json()["releases"].items() if ver == version]
    if (not matched) or (not matched[0]):
        return ""

    upload_time = matched[0][0]["upload_time"]
    return upload_time.split("T")[0]  # return year-month-day


def get_longest_string_length(array):
    return len(max(array, key=len))


def safe_result(future, if_error=""):
    try:
        return future.result()
    except Exception:
        traceback.print_exc()
        return if_error


def main():
    distributions = get_distributions()
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
        futures = [executor.submit(get_release_date, pkg, ver) for pkg, ver in distributions]
        release_dates = [safe_result(f) for f in futures]

    packages, versions = list(zip(*distributions))
    package_legnth = get_longest_string_length(packages)
    version_length = get_longest_string_length(versions)
    release_date_length = len("Release Date")
    print("Package".ljust(package_legnth), "Version".ljust(version_length), "Release Date")
    print("-" * (package_legnth + version_length + release_date_length + 2))
    for package, version, release_date in sorted(
        zip(packages, versions, release_dates),
        # Sort by release date in descending order
        key=lambda x: x[2],
        reverse=True,
    ):
        print(
            package.ljust(package_legnth),
            version.ljust(version_length),
            release_date.ljust(release_date_length),
        )


if __name__ == "__main__":
    main()
