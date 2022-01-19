"""
A script to update the maximum package versions in 'mlflow/ml-package-versions.yml'.

# Prerequisites:
$ pip install packaging pyyaml

# How to run (make sure you're in the repository root):
$ python dev/update_ml_package_versions.py
"""
import argparse
import json
from packaging.version import Version
import re
import sys
import urllib.request
import yaml
from datetime import datetime
from dataclasses import dataclass


def read_file(path):
    with open(path) as f:
        return f.read()


def save_file(src, path):
    with open(path, "w") as f:
        f.write(src)


@dataclass
class Release:
    version: Version
    release_date: datetime


def iter_package_releases(package_name):
    url = "https://pypi.python.org/pypi/{}/json".format(package_name)
    with urllib.request.urlopen(url) as res:
        data = json.load(res)

    for version, dist_files in data["releases"].items():
        ver = Version(version)
        if len(dist_files) == 0 or ver.is_devrelease or ver.is_prerelease:
            continue

        upload_times = [f["upload_time"] for f in dist_files]
        release_date = datetime.fromisoformat(min(upload_times))
        yield Release(ver, release_date)


def update_versions(yml_string, flavor, category, *, min_release, max_release):
    # Construct regular expressions to capture this pattern:
    # ======================================================
    # {key}:
    #   ...
    #   {category}:
    #     ...
    #     {minimum | maximum}: "1.2.3"
    kwargs = dict(flavor=re.escape(flavor), category=category)
    min_pattern = r"({flavor}:.+?{category}:.+?minimum).+?\n".format(**kwargs)
    max_pattern = r"({flavor}:.+?{category}:.+?maximum).+?\n".format(**kwargs)
    #               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                              \g<1>
    repl_tmpl = r'\g<1>: "{version}"\n'

    # Update the minimum version
    res = re.sub(
        min_pattern,
        repl_tmpl.format(version=min_release.version),
        yml_string,
        flags=re.DOTALL,
    )
    # Update the maximum version
    res = re.sub(
        max_pattern,
        repl_tmpl.format(version=max_release.version),
        res,
        flags=re.DOTALL,
    )
    return res


def parse_args(args):
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
    args = parse_args(args)

    yml_path = args.path
    old_src = read_file(yml_path)
    new_src = old_src
    config_dict = yaml.load(old_src, Loader=yaml.SafeLoader)

    for flavor, config in config_dict.items():
        for category in ["autologging", "models"]:
            print("Processing", (flavor, category))

            version_config = config.get(category)
            if not version_config or version_config.get("pin_maximum", False):
                continue

            package_name = config["package_info"]["pip_release"]
            releases = iter_package_releases(package_name)
            # Filter versions
            utc_now = datetime.utcnow()
            existing_min_ver = Version(version_config["minimum"])
            unsupported = version_config.get("unsupported", [])
            releases = [
                r
                for r in releases
                if not (
                    # Released more than two years ago?
                    abs((r.release_date - utc_now).days) > 365 * 2
                    # Older than the existing minimum version?
                    or (r.version < existing_min_ver)
                    # Marked as unsupported?
                    or (str(r.version) in unsupported)
                )
            ]
            sorted_releases = sorted(releases, key=lambda r: r.version)
            min_release = sorted_releases[0]
            max_release = sorted_releases[-1]
            new_src = update_versions(
                new_src,
                flavor,
                category,
                min_release=min_release,
                max_release=max_release,
            )

    save_file(new_src, yml_path)


if __name__ == "__main__":
    main(sys.argv[1:])
