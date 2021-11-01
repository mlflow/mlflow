import subprocess
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def get_distributions():
    res = subprocess.run(["pip", "list"], stdout=subprocess.PIPE)
    pip_list_stdout = res.stdout.decode("utf-8")
    # `pip_list_stdout` looks like this:
    # ``````````````````````````````````````````````````````````
    # Package     Version             Location
    # ----------- ------------------- --------------------------
    # mlflow      1.21.1.dev0         /home/user/path/to/mlflow
    # tensorflow  2.6.0
    # ...
    # ```````````````````````````````````````````````````````````
    lines = pip_list_stdout.splitlines()[2:]  # `[2:]` removes the header
    return [
        # Extract package and version
        line.split()[:2]
        for line in lines
    ]


def get_release_date(distribution):
    package_name, version = distribution
    resp = requests.get(f"https://pypi.python.org/pypi/{package_name}/json")
    if not resp.ok:
        return None

    matched = [dist_files for ver, dist_files in resp.json()["releases"].items() if ver == version]
    if (not matched) or (not matched[0]):
        return None

    upload_time = matched[0][0]["upload_time"]
    return upload_time.split("T")[0]  # return year-month-day


def main():
    distributions = get_distributions()

    with ThreadPoolExecutor() as executor:
        release_dates = list(executor.map(get_release_date, distributions))

    print(
        pd.DataFrame(distributions, columns=["package", "version"])
        .assign(release_date=release_dates)
        .sort_values("release_date", ascending=False)
        .to_markdown(index=False)
    )


if __name__ == "__main__":
    main()

