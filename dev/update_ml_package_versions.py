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
import urllib.request
import yaml
from packaging.version import Version


def read_file(path):
    """
    Reads a file from a path and returns its contents
    :param path: path to read file from
    """
    with open(path, encoding="utf-8") as file:
        return file.read()


def save_file(src, path):
    """
    Saves a file to a path
    :param src: source to get file from
    :param path: destination path to store file to
    """
    with open(path, "w", encoding="utf-8") as file:
        file.write(src)


def get_package_versions(package_name):
    """
    Get all the available package versions
    :param package_name: Name of the package
    """
    url = f"https://pypi.python.org/pypi/{package_name}/json"
    with urllib.request.urlopen(url) as res:
        data = json.load(res)

    def is_dev_or_pre_release(version_str):
        ver = Version(version_str)
        return ver.is_devrelease or ver.is_prerelease

    return [
        version
        for version, dist_files in data["releases"].items()
        if len(dist_files) > 0 and not is_dev_or_pre_release(version)
    ]


def get_latest_version(candidates):
    """
    Get latest version of a package from candidates
    :param candidates: Candidates to calculate latest version from
    """
    return sorted(candidates, key=Version, reverse=True)[0]


def update_max_version(src, key, new_max_version, category):
    """
    Updates to max version from a source.
    :param src: Source
    :param key: Key to which max version to update to
    :param new_max_version: New maximum version
    :param category: Categories to update version to

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
    >>> new_src = update_max_version(src, "sklearn", "0.1.0", "models")
    >>> new_src = update_max_version(new_src, "xgboost", "1.2.1", "autologging")
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
    pattern = rf"({re.escape(key)}:.+?{category}:.+?maximum: )\".+?\""
    # Matches the following pattern:
    #
    # <key>:
    #   ...
    #   <category>:
    #     ...
    #     maximum: "1.2.3"
    return re.sub(pattern, rf'\g<1>"{new_max_version}"', src, flags=re.DOTALL)


def parse_args(args):
    """
    A function to parse arguments
    :param args: arguments to parse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the ML package versions yaml (default: mlflow/ml-package-versions.yml)",
        default="mlflow/ml-package-versions.yml",
        required=False,
    )
    return parser.parse_args(args)


def main(args):
    """
    Main function to update ml package versions
    :param args: arguments to parse
    """
    args = parse_args(args)

    yml_path = args.path
    old_src = read_file(yml_path)
    new_src = old_src
    config_dict = yaml.load(old_src, Loader=yaml.SafeLoader)

    for flavor_key, config in config_dict.items():
        for category in ["autologging", "models"]:
            if (category not in config) or config[category].get("pin_maximum", False):
                continue
            print("Processing", flavor_key, category)

            package_name = config["package_info"]["pip_release"]
            max_ver = config[category]["maximum"]
            versions = get_package_versions(package_name)
            unsupported = config[category].get("unsupported", [])
            versions = set(versions).difference(unsupported)  # exclude unsupported versions
            latest_version = get_latest_version(versions)

            if max_ver == latest_version:
                continue

            new_src = update_max_version(new_src, flavor_key, latest_version, category)

    save_file(new_src, yml_path)


if __name__ == "__main__":
    main(sys.argv[1:])
