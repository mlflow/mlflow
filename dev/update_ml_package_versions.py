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
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
    if ut := dist.get("upload_time_iso_8601"):
        delta = datetime.now(timezone.utc) - datetime.fromisoformat(ut.replace("Z", "+00:00"))
        return delta.days < 1
    return False


@dataclass
class VersionInfo:
    version: str
    upload_time: datetime


def get_package_version_infos(package_name: str) -> list[VersionInfo]:
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
        VersionInfo(
            version=version,
            upload_time=datetime.fromisoformat(dist_files[0]["upload_time"]),
        )
        for version, dist_files in data["releases"].items()
        if (
            len(dist_files) > 0
            and not is_dev_or_pre_release(version)
            and not any(uploaded_recently(dist) for dist in dist_files)
            and not any(dist.get("yanked", False) for dist in dist_files)
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

    return autolog_flavor_module_map


def update_ml_package_versions_py(config_path):
    with open(config_path) as f:
        genai_config = {}
        non_genai_config = {}

        for name, cfg in yaml.load(f, Loader=yaml.SafeLoader).items():
            # Extract required fields
            pip_release = extract_field(cfg, ("package_info", "pip_release"))
            module_name = extract_field(cfg, ("package_info", "module_name"))
            min_version = extract_field(cfg, ("models", "minimum"))
            max_version = extract_field(cfg, ("models", "maximum"))
            genai = extract_field(cfg, ("package_info", "genai"))
            config_to_update = genai_config if genai else non_genai_config
            if min_version:
                config_to_update[name] = {
                    "package_info": {
                        "pip_release": pip_release,
                    },
                    "models": {
                        "minimum": min_version,
                        "maximum": max_version,
                    },
                }
            else:
                config_to_update[name] = {
                    "package_info": {
                        "pip_release": pip_release,
                    }
                }
            if module_name:
                config_to_update[name]["package_info"]["module_name"] = module_name

            # Check for autologging configuration
            autolog_min_version = extract_field(cfg, ("autologging", "minimum"))
            autolog_max_version = extract_field(cfg, ("autologging", "maximum"))
            if (pip_release, autolog_min_version, autolog_max_version).count(None) > 0:
                continue

            config_to_update[name].update(
                {
                    "autologging": {
                        "minimum": autolog_min_version,
                        "maximum": autolog_max_version,
                    }
                },
            )

        genai_flavor_module_mapping = _get_autolog_flavor_module_map(genai_config)
        # We have "langgraph" entry in ml-package-versions.yml so that we can run test
        # against multiple versions of langgraph. However, we don't have a flavor for
        # langgraph and it is a part of the langchain flavor.
        genai_flavor_module_mapping.pop("langgraph", None)

        non_genai_flavor_module_mapping = _get_autolog_flavor_module_map(non_genai_config)
        # Add special case for pyspark.ml (non-GenAI)
        non_genai_flavor_module_mapping["pyspark.ml"] = "pyspark"

        this_file = Path(__file__).name
        dst = Path("mlflow", "ml_package_versions.py")

        config_str = json.dumps(genai_config | non_genai_config, indent=4)

        Path(dst).write_text(
            f"""\
# This file was auto-generated by {this_file}.
# Please do not edit it manually.

_ML_PACKAGE_VERSIONS = {config_str}

# A mapping of flavor name to the module name to be imported for autologging.
# This is used for checking version compatibility in autologging.
# DO NOT EDIT MANUALLY

# GenAI packages
GENAI_FLAVOR_TO_MODULE_NAME = {json.dumps(genai_flavor_module_mapping, indent=4)}

# Non-GenAI packages
NON_GENAI_FLAVOR_TO_MODULE_NAME = {json.dumps(non_genai_flavor_module_mapping, indent=4)}

# Combined mapping for backward compatibility
FLAVOR_TO_MODULE_NAME = NON_GENAI_FLAVOR_TO_MODULE_NAME | GENAI_FLAVOR_TO_MODULE_NAME
"""
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Update MLflow package versions")
    parser.add_argument(
        "--skip-yml", action="store_true", help="Skip updating ml-package-versions.yml"
    )
    return parser.parse_args()


def get_min_supported_version(versions_infos: list[VersionInfo], genai: bool = False) -> str | None:
    """
    Get the minimum version that is released within the past two years
    """
    years = 1 if genai else 2
    min_support_date = datetime.now() - timedelta(days=years * 365)
    min_support_date = min_support_date.replace(tzinfo=None)

    # Extract versions that were released in the past two years
    recent_versions = [v for v in versions_infos if v.upload_time > min_support_date]

    if not recent_versions:
        return None

    # Get minimum version according to upload date
    return min(recent_versions, key=lambda v: v.upload_time).version


def update(skip_yml=False):
    yml_path = "mlflow/ml-package-versions.yml"

    if not skip_yml:
        old_src = read_file(yml_path)
        new_src = old_src
        config_dict = yaml.load(old_src, Loader=yaml.SafeLoader)
        for flavor_key, config in config_dict.items():
            # We currently don't have bandwidth to support newer versions of these flavors.
            if flavor_key in ["litellm"]:
                continue
            package_name = config["package_info"]["pip_release"]
            genai = config["package_info"].get("genai", False)
            versions_and_upload_times = get_package_version_infos(package_name)
            min_supported_version = get_min_supported_version(
                versions_and_upload_times, genai=genai
            )

            for category in ["autologging", "models"]:
                print("Processing", flavor_key, category)

                if category in config and "minimum" in config[category]:
                    old_min_version = config[category]["minimum"]
                    if flavor_key == "spark":
                        # We should support pyspark versions that are older than the cut off date.
                        pass
                    elif min_supported_version is None:
                        # The latest release version was 2 years ago.
                        # set the min version to be the same with the max version.
                        max_ver = config[category]["maximum"]
                        new_src = update_version(
                            new_src, flavor_key, max_ver, category, update_max=False
                        )
                    elif Version(min_supported_version) > Version(old_min_version):
                        new_src = update_version(
                            new_src, flavor_key, min_supported_version, category, update_max=False
                        )

                if (category not in config) or config[category].get("pin_maximum", False):
                    continue

                max_ver = config[category]["maximum"]
                versions = [v.version for v in versions_and_upload_times]
                unsupported = config[category].get("unsupported", [])
                versions = set(versions).difference(unsupported)  # exclude unsupported versions
                latest_version = get_latest_version(versions)

                if Version(latest_version) <= Version(max_ver):
                    continue

                new_src = update_version(
                    new_src, flavor_key, latest_version, category, update_max=True
                )

        save_file(new_src, yml_path)

    update_ml_package_versions_py(yml_path)


def main():
    args = parse_args()
    update(args.skip_yml)


if __name__ == "__main__":
    main()
