"""
A script to update the maximum package versions in 'mlflow/ml-package-versions.yml'.

# Prerequisites:
$ pip install packaging pyyaml

# How to run (make sure you're in the repository root):
$ python dev/update_ml_package_versions.py
"""

import argparse
import json
import re
import requests
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from dateutil.parser import isoparse
from dateutil.relativedelta import relativedelta
from pathlib import Path

import yaml
from packaging.version import Version


def read_file(path):
    with open(path) as f:
        return f.read()


def save_file(src, path):
    with open(path, "w") as f:
        f.write(src)


def uploaded_recently(dist) -> bool:
    if ut := dist.get("upload_time"):
        return (datetime.now() - datetime.fromisoformat(ut)).days < 1
    return False


def get_package_versions(package_name):
    url = f"https://pypi.python.org/pypi/{package_name}/json"
    for _ in range(5):  # Retry up to 5 times
        try:
            with urllib.request.urlopen(url) as res:
                data = json.load(res)
        except ConnectionResetError as e:
            sys.stderr.write(f"Retrying {url} due to {e}\n")
            time.sleep(1)
        else:
            break
    else:
        raise Exception(f"Failed to fetch {url}")

    def is_dev_or_pre_release(version_str):
        v = Version(version_str)
        return v.is_devrelease or v.is_prerelease

    return [
        version
        for version, dist_files in data["releases"].items()
        if (
            len(dist_files) > 0
            and not is_dev_or_pre_release(version)
            and not any(uploaded_recently(dist) for dist in dist_files)
        )
    ]


def get_latest_version(candidates):
    return sorted(candidates, key=Version, reverse=True)[0]


def update_version(src, key, new_version, category, update_max):
    """
    Examples
    ========
    >>> src = '''
    ... sklearn:
    ...   ...
    ...   models:
    ...     minimum: "0.0.0"
    ...     maximum: "0.0.0"
    ... xgboost:
    ...   ...
    ...   autologging:
    ...     minimum: "1.1.1"
    ...     maximum: "1.1.1"
    ... '''.strip()
    >>> new_src = update_version(src, "sklearn", "0.1.0", "models", update_max=True)
    >>> new_src = update_version(new_src, "xgboost", "1.2.1", "autologging", update_max=True)
    >>> print(new_src)
    sklearn:
      ...
      models:
        minimum: "0.0.0"
        maximum: "0.1.0"
    xgboost:
      ...
      autologging:
        minimum: "1.1.1"
        maximum: "1.2.1"
    """
    match = "maximum" if update_max else "minimum"
    pattern = r"((^|\n){key}:.+?{category}:.+?{match}: )\".+?\"".format(
        key=re.escape(key), category=category, match=match
    )
    # Matches the following pattern:
    #
    # <key>:
    #   ...
    #   <category>:
    #     ...
    #     maximum: "1.2.3"
    return re.sub(pattern, rf'\g<1>"{new_version}"', src, flags=re.DOTALL)


def extract_field(d, keys):
    for key in keys:
        if key in d:
            d = d[key]
        else:
            return None
    return d


def _get_autolog_flavor_module_map(config):
    """
    Parse _ML_PACKAGE_VERSIONS to get the mapping of flavor name to
    the module name to be imported for autologging.
    """
    autolog_flavor_module_map = {}
    for flavor, config in config.items():
        if "autologging" not in config:
            continue
        module_name = config["package_info"].get("module_name", flavor)
        autolog_flavor_module_map[flavor] = module_name

    # pyspark.ml is a special case of spark flavor
    autolog_flavor_module_map["pyspark.ml"] = "pyspark"
    return autolog_flavor_module_map


def update_ml_package_versions_py(config_path):
    with open(config_path) as f:
        config = {}
        for name, cfg in yaml.load(f, Loader=yaml.SafeLoader).items():
            # Extract required fields
            pip_release = extract_field(cfg, ("package_info", "pip_release"))
            module_name = extract_field(cfg, ("package_info", "module_name"))
            min_version = extract_field(cfg, ("models", "minimum"))
            max_version = extract_field(cfg, ("models", "maximum"))
            if min_version:
                config[name] = {
                    "package_info": {
                        "pip_release": pip_release,
                    },
                    "models": {
                        "minimum": min_version,
                        "maximum": max_version,
                    },
                }
            else:
                config[name] = {
                    "package_info": {
                        "pip_release": pip_release,
                    }
                }
            if module_name:
                config[name]["package_info"]["module_name"] = module_name
            min_version = extract_field(cfg, ("autologging", "minimum"))
            max_version = extract_field(cfg, ("autologging", "maximum"))
            if (pip_release, min_version, max_version).count(None) > 0:
                continue

            config[name].update(
                {
                    "autologging": {
                        "minimum": min_version,
                        "maximum": max_version,
                    }
                },
            )
        flavor_module_mapping = _get_autolog_flavor_module_map(config)

        # We have "langgraph" entry in ml-package-versions.yml so that we can run test
        # against multiple versions of langgraph. However, we don't have a flavor for
        # langgraph and it is a part of the langchain flavor.
        flavor_module_mapping.pop("langgraph", None)

        this_file = Path(__file__).name
        dst = Path("mlflow", "ml_package_versions.py")
        Path(dst).write_text(
            f"""\
# This file was auto-generated by {this_file}.
# Please do not edit it manually.

_ML_PACKAGE_VERSIONS = {json.dumps(config, indent=4)}

# A mapping of flavor name to the module name to be imported for autologging.
# This is used for checking version compatibility in autologging.
# DO NOT EDIT MANUALLY
FLAVOR_TO_MODULE_NAME = {json.dumps(flavor_module_mapping, indent=4)}
"""
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Update MLflow package versions")
    parser.add_argument(
        "--skip-yml", action="store_true", help="Skip updating ml-package-versions.yml"
    )
    return parser.parse_args()


def get_cut_version(package):
    """
    Get the minimum version that is released within the past two years
    """
    cut_date = datetime.now() - relativedelta(years=2, months=0)
    cut_date = cut_date.replace(tzinfo=None)
    url = f"https://pypi.org/pypi/{package}/json"
    resp = requests.get(url)
    data = resp.json()

    releases = data["releases"]

    min_version = None
    min_upload_time = None
    for version, files in releases.items():
        if files:  # skip empty releases
            yanked = any(file.get("yanked", False) for file in files)
            pyver = Version(version)
            is_official = not (pyver.is_devrelease or pyver.is_prerelease or pyver.is_postrelease)
            upload_time = isoparse(files[0]["upload_time_iso_8601"]).replace(tzinfo=None)

            if is_official and not yanked and upload_time > cut_date:
                if min_upload_time is None or upload_time < min_upload_time:
                    min_version = version
                    min_upload_time = upload_time

    return min_version


def update(skip_yml=False):
    yml_path = "mlflow/ml-package-versions.yml"

    if not skip_yml:
        old_src = read_file(yml_path)
        new_src = old_src
        config_dict = yaml.load(old_src, Loader=yaml.SafeLoader)
        for flavor_key, config in config_dict.items():
            package_name = config["package_info"]["pip_release"]
            cut_version = get_cut_version(package_name)

            for category in ["autologging", "models"]:
                print("Processing", flavor_key, category)

                if category in config:
                    old_min_version = config[category]["minimum"]
                    if cut_version is None:
                        # The latest release version is 2 years ago.
                        # set the min version to be the same with the max version.
                        max_ver = config[category]["maximum"]
                        new_src = update_version(
                            new_src, flavor_key, max_ver, category, update_max=False
                        )
                    elif Version(cut_version) > Version(old_min_version):
                        new_src = update_version(
                            new_src, flavor_key, cut_version, category, update_max=False
                        )

                if (category not in config) or config[category].get("pin_maximum", False):
                    continue

                max_ver = config[category]["maximum"]
                versions = get_package_versions(package_name)
                unsupported = config[category].get("unsupported", [])
                versions = set(versions).difference(unsupported)  # exclude unsupported versions
                latest_version = get_latest_version(versions)

                if Version(latest_version) <= Version(max_ver):
                    continue

                new_src = update_version(new_src, flavor_key, latest_version, category, update_max=True)

        save_file(new_src, yml_path)

    update_ml_package_versions_py(yml_path)


def main():
    args = parse_args()
    update(args.skip_yml)


if __name__ == "__main__":
    main()
