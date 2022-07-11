"""
A script to display the package release dates.
"""
import os
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor

import requests


def get_distributions():
    """
    A function that extracts the package and version
    """
    res = subprocess.run(["pip", "list"], stdout=subprocess.PIPE, check=True)
    pip_list_stdout = res.stdout.decode("utf-8")
    # `pip_list_stdout` looks like this:
    # ``````````````````````````````````````````````````````````
    # Package     Version             Location
    # ----------- ------------------- --------------------------
    # mlflow      1.21.1.dev0         /home/user/path/to/mlflow
    # tensorflow  2.6.0
    # ...
    # ``````````````````````````````````````````````````````````
    lines = pip_list_stdout.splitlines()[2:]  # `[2:]` removes the header
    return [
        # Extract package and version
        line.split()[:2]
        for line in lines
    ]


def get_release_date(package, version):
    """
    Fetches a release date for a specific package and a version number
    :param package: package to find release date for
    :param version: version of package to find release date for
    """
    resp = requests.get(f"https://pypi.python.org/pypi/{package}/json", timeout=10)
    if not resp.ok:
        return ""

    matched = [dist_files for ver, dist_files in resp.json()["releases"].items() if ver == version]
    if (not matched) or (not matched[0]):
        return ""

    upload_time = matched[0][0]["upload_time"]
    return upload_time.split("T")[0]  # return year-month-day


def get_longest_string_length(array):
    """
    Gets the length of longest string
    :param array: array to calculate longest string length on
    """
    return len(max(array, key=len))


def safe_result(future, if_error=""):
    """
    Returns the result object safely
    :param future: future object to get result from
    :param if_error: error string to return in case of exception
    """
    try:
        return future.result()
    except future.exception():
        traceback.print_exc()
        return if_error


def main():
    """
    A main function which orchestrates package version and release date display.
    """
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
