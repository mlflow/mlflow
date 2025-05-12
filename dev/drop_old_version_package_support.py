"""
A script to update the minimum package versions in 'mlflow/ml-package-versions.yml'.
The flavor package versions those are older than 1.5 years are dropped.

# How to run (make sure you're in the repository root):
$ python dev/drop_old_version_package_support.py
"""

from datetime import datetime
import re
import requests
import yaml

from packaging.version import Version
from dateutil.parser import isoparse
from dateutil.relativedelta import relativedelta


def get_cut_version(package, cut_date):
    cut_date = cut_date.replace(tzinfo=None)
    url = f"https://pypi.org/pypi/{package}/json"
    resp = requests.get(url)
    data = resp.json()

    releases = data["releases"]

    min_verion = None
    min_upload_time = None
    for version, files in releases.items():
        if files:  # skip empty releases
            yanked = any(file.get("yanked", False) for file in files)
            pyver = Version(version)
            is_official = not (pyver.is_devrelease or pyver.is_prerelease or pyver.is_postrelease)
            upload_time = isoparse(files[0]["upload_time_iso_8601"]).replace(tzinfo=None)

            if is_official and not yanked and upload_time > cut_date:
                if min_upload_time is None or upload_time < min_upload_time:
                    min_verion = version
                    min_upload_time = upload_time

    return min_verion


def update_min_version(src, key, new_min_version, category):
    pattern = r"((^|\n){key}:.+?{category}:.+?minimum: )\".+?\"".format(
        key=re.escape(key), category=category
    )
    # Matches the following pattern:
    #
    # <key>:
    #   ...
    #   <category>:
    #     ...
    #     minimum: "1.2.3"
    return re.sub(pattern, rf'\g<1>"{new_min_version}"', src, flags=re.DOTALL)


def update_config(config_path, cut_date):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(config_path) as f:
        src = f.read()

    new_src = src
    for flavor in config:
        flavor_cfg = config[flavor]
        package = flavor_cfg["package_info"]["pip_release"]
        cut_version = get_cut_version(package, cut_date)
        for category in ["autologging", "models"]:
            if category in flavor_cfg:
                old_min_version = flavor_cfg[category]["minimum"]
                if cut_version is not None and Version(cut_version) > Version(old_min_version):
                    new_src = update_min_version(new_src, flavor, cut_version, category)
        print(f"Updated flavor {flavor}")

    with open(config_path, "w") as f:
        f.write(new_src)


def main():
    cut_date = datetime.now() - relativedelta(years=1, months=6)
    update_config(
        "mlflow/ml-package-versions.yml",
        cut_date
    )


if __name__ == "__main__":
    main()
