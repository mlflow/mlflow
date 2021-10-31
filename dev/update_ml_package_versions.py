"""
A script to update the maximum package versions in 'mlflow/ml-package-versions.yml'.

# Prerequisites:
$ pip install packaging pyyaml

# How to run (make sure you're in the repository root):
$ python dev/update_ml_package_versions.py
"""
import argparse
import json
from datetime import datetime
from packaging.version import Version
import re
import sys
import urllib.request
import yaml
from collections import namedtuple
import itertools


def read_file(path):
    with open(path) as f:
        return f.read()


def save_file(src, path):
    with open(path, "w") as f:
        f.write(src)


Release = namedtuple("Release", ["version", "release_date"])


def get_packages_releases(package_name):
    url = "https://pypi.python.org/pypi/{}/json".format(package_name)
    with urllib.request.urlopen(url) as res:
        data = json.load(res)

    def is_dev_or_pre_release(version_str):
        v = Version(version_str)
        return v.is_devrelease or v.is_prerelease

    return [
        Release(Version(version), datetime.fromisoformat(dist_files[0]["upload_time"]))
        for version, dist_files in data["releases"].items()
        if len(dist_files) > 0 and not is_dev_or_pre_release(version)
    ]


def days_between(d1, d2):
    return abs((d2 - d1).days)


def get_utc_now():
    return datetime.utcnow()


def drop_old_releases(releases, days_threshold):
    utcnow = get_utc_now()
    return [r for r in releases if days_between(utcnow, r.release_date) <= days_threshold]


def update_suppported_version(src, flavor, category, min_or_max, new_version):
    """
    Examples
    ========
    >>> src = '''
    ... sklearn:
    ...   ...
    ...   models:
    ...     minimum: "0.1.0"
    ...     maximum: "0.3.0"
    ... '''.strip()
    >>> print(update_suppported_version(src, "sklearn", "models", "minimum", "0.2.0"))
    sklearn:
      ...
      models:
        minimum: "0.2.0"
        maximum: "0.3.0"
    >>> print(update_suppported_version(src, "sklearn", "models", "maximum", "0.4.0"))
    sklearn:
      ...
      models:
        minimum: "0.1.0"
        maximum: "0.4.0"
    """
    assert min_or_max in ["minimum", "maximum"]

    pattern = r'({flavor}:.+?{category}:.+?{min_or_max}: )".+?"'.format(
        flavor=re.escape(flavor), category=category, min_or_max=min_or_max
    )
    # This matches the following pattern:
    #
    # <flavor>:
    #   ...
    #   <category>:
    #     ...
    #     <min_or_max>: "1.2.3"
    return re.sub(
        pattern.format(min_or_max="minimum"),
        r'\g<1>"{}"'.format(new_version),
        src,
        flags=re.DOTALL,
    )


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        help="Path to the ML package versions yaml (default: mlflow/ml-package-versions.yml)",
        default="mlflow/ml-package-versions.yml",
        required=False,
    )
    parser.add_argument(
        "--drop-old-versions",
        action="store_true",
        help="If specified, drop support for package versions released more than 2 years ago",
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    yml_path = args.path
    old_src = read_file(yml_path)
    new_src = old_src
    config_dict = yaml.load(old_src, Loader=yaml.SafeLoader)
    categories = ["autologging", "models"]

    for (flavor_key, config), category in itertools.product(config_dict.items(), categories):
        if (category not in config) or config[category].get("pin_maximum", False):
            continue
        print("Processing", flavor_key, category)

        package_name = config["package_info"]["pip_release"]
        releases = get_packages_releases(package_name)

        # Drop unsupported versions
        unsupported = config[category].get("unsupported", [])
        releases = [r for r in releases if str(r.version) not in unsupported]
        sorted_releases = sorted(releases, key=lambda r: r.version)

        # Update the maximum version
        latest_release = sorted_releases[-1]
        max_ver = config[category]["maximum"]
        if Version(max_ver) < latest_release.version:
            new_src = update_suppported_version(
                new_src, flavor_key, category, "maximum", latest_release.version
            )

        # Update the minimum version if `--drop-old-versions` is specified
        if args.drop_old_versions:
            oldest_release = drop_old_releases(sorted_releases, days_threshold=365 * 2)[0]
            min_ver = config[category]["minimum"]
            if Version(min_ver) < oldest_release.version:
                new_src = update_suppported_version(
                    new_src, flavor_key, category, "minimum", oldest_release.version
                )

    save_file(new_src, yml_path)


if __name__ == "__main__":
    main(sys.argv[1:])
