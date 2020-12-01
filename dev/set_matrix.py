"""
A script to set a matrix for the cross version tests for MLflow Models / autologging integrations.

# How to run:

python dev/set_matrix.py --diff-only \
    --ref-config <URL or local file path> \
    --changed-files mlflow/sklearn/__init__.py

# How to run doctests:

pytest dev/set_matrix.py --doctest-modules --verbose
"""

import argparse
from distutils.version import LooseVersion
import json
import operator
import os
import re
import shutil
import urllib.request

import yaml

CONFIG_PATH = "ml-package-versions.yml"


def read_yaml(location):
    """
    Reads a YAML file. `location` can be a URL or local file path.

    Examples
    --------
    >>> path = ".github/workflows/master.yml"
    >>> read_yaml(path)
    {...}
    >>> url = "https://raw.githubusercontent.com/mlflow/mlflow/master/" + path
    >>> read_yaml(url)
    {...}
    """
    if re.search("^https?://", location):
        with urllib.request.urlopen(location) as f:
            return yaml.load(f, Loader=yaml.SafeLoader)
    else:
        with open(location) as f:
            return yaml.load(f, Loader=yaml.SafeLoader)


def get_released_versions(package_name):
    """
    Fetches the released versions & datetimes of the specified Python package.

    Examples
    --------
    >>> get_released_versions("scikit-learn")
    {'0.10': '2012-01-11T14:42:25', '0.11': '2012-05-08T00:40:14', ...}
    """
    # Ref.: https://stackoverflow.com/a/27239645/6943581

    url = "https://pypi.python.org/pypi/{}/json".format(package_name)
    data = json.load(urllib.request.urlopen(url))

    versions = {
        ver: files[0]["upload_time"] for ver, files in data["releases"].items() if len(files) > 0
    }
    return versions


def is_final_release(ver):
    """
    Returns True if the given version matches PEP440's final release scheme.

    Examples
    --------
    >>> is_final_release("0.1")
    True
    >>> is_final_release("0.23.0")
    True
    >>> is_final_release("0.4.0a1")
    False
    >>> is_final_release("0.5.0rc")
    False
    """
    # Ref.: https://www.python.org/dev/peps/pep-0440/#final-releases
    return re.search(r"^\d+(\.\d+)+$", ver) is not None


def select_latest_micro_versions(versions):
    """
    Selects the latest micro version in each minor version.

    Examples
    --------
    >>> versions = {
    ...     "1.3.0": "2020-01-01T00:00:00",
    ...     "1.3.1": "2020-02-01T00:00:00",  # latest in 1.3
    ...     "1.4.0": "2020-03-01T00:00:00",
    ...     "1.4.1": "2020-04-01T00:00:00",
    ...     "1.4.2": "2020-05-01T00:00:00",  # latest in 1.4
    ... }
    >>> select_latest_micro_versions(versions)
    ['1.3.1', '1.4.2']
    """
    seen_minors = set()
    res = []

    for ver, _ in sorted(
        versions.items(),
        # sort by (minor_version, upload_time) in descending order
        key=lambda x: (LooseVersion(x[0]).version[:2], x[1]),
        reverse=True,
    ):
        minor_ver = tuple(LooseVersion(ver).version[:2])  # A set doesn't accept a list

        if minor_ver not in seen_minors:
            seen_minors.add(minor_ver)
            res.insert(0, ver)

    return res


def get_major_version(ver):
    return LooseVersion(ver).version[0]


def filter_versions(versions, min_ver, max_ver, excludes=None):
    """
    Filter versions that satisfy the following conditions:

    1. is newer than or equal to `min_ver`
    2. shares the same major version as `max_ver` or `min_ver`
    3. (Optional) is not in `excludes`

    Examples
    --------
    >>> versions = {
    ...     "0.1.0": "2020-01-01T00:00:01",
    ...     "0.2.0": "2020-01-01T00:00:02",
    ...     "1.0.0": "2020-01-01T00:00:00",
    ...     "1.1.0": "2020-01-01T00:01:00",
    ... }
    >>> filter_versions(versions, "0.1.0", "0.2.0")  # fetch up to the latest in 0.x.y
    {'0.1.0': ..., '0.2.0': ...}
    >>> filter_versions(versions, "0.1.0", "1.0.0")  # fetch up to the latest in 1.x.y
    {'0.1.0': ..., '0.2.0': ..., '1.0.0': ..., '1.1.0': ...}
    >>> filter_versions(versions, "0.1.0", "1.0.0", excludes=["0.2.0"])
    {'0.1.0': ..., '1.0.0': ..., '1.1.0': ...}
    """
    if excludes is None:
        excludes = []

    # prevent specifying non-existent versions
    assert all(v in versions for v in excludes)
    assert min_ver in versions
    assert max_ver in versions

    versions = {v: t for v, t in versions.items() if v not in excludes}
    versions = {v: t for v, t in versions.items() if is_final_release(v)}

    max_major = get_major_version(max_ver)
    versions = {v: t for v, t in versions.items() if get_major_version(v) <= max_major}
    versions = {v: t for v, t in versions.items() if LooseVersion(v) >= LooseVersion(min_ver)}

    return versions


def get_changed_flavors(changed_files, flavors):
    """
    Detects changed flavors from a list of changed files.

    Examples
    --------
    >>> flavors = ["pytorch", "xgboost"]
    >>> get_changed_flavors(["mlflow/pytorch/__init__.py", "mlflow/xgboost.py"], flavors)
    ['pytorch', 'xgboost']
    >>> get_changed_flavors(["mlflow/xgboost.py"], flavors)
    ['xgboost']
    >>> get_changed_flavors(["README.rst"], flavors)
    []
    >>> get_changed_flavors([], flavors)
    []
    """
    changed_flavors = []
    for f in changed_files:
        match = re.search(r"^mlflow/(.+?)(\.py|/)", f)

        if (match is not None) and (match.group(1) in flavors):
            changed_flavors.append(match.group(1))

    return changed_flavors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diff-only",
        action="store_true",
        help="If specified, ignore unchanged flavors or configs.",
    )
    # make `--ref-config` and `--diff-files` configurable to make it easier to test this script.
    parser.add_argument(
        "--ref-config",
        default="https://raw.githubusercontent.com/mlflow/mlflow/master/{}".format(CONFIG_PATH),
        help=(
            "URL or local file path of the reference config. "
            "Valid only when `--diff-only` is specified."
        ),
    )
    parser.add_argument(
        "--changed-files",
        type=lambda x: [] if x.strip() == "" else x.strip().split("\n"),
        default="",
        help=(
            "A string that represents a list of changed files in a pull request. "
            "Valid only when `--diff-only` is specified."
        ),
    )

    return parser.parse_args()


def divider(title):
    """
    Generates a divider (e.g. '=== title ===').
    """
    width = shutil.get_terminal_size(fallback=(80, 24))[0]
    rest = width - len(title) - 2
    left = rest // 2 if rest % 2 else (rest + 1) // 2
    return "\n{} {} {}".format("=" * left, title, "=" * (rest - left))


def str_to_operator(s):
    """
    Turns a string into the corresponding operator.
    """
    return {
        # https://docs.python.org/3/library/operator.html#mapping-operators-to-functions
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">": operator.gt,
    }[s]


def get_operator_and_version(ver_spec):
    """
    Converts a version specifier (e.g. "< 3") to a tuple of (operator, version).

    >>> get_operator_and_version("< 3")
    (<built-in function lt>, '3')
    >>> get_operator_and_version("!= dev")
    (<built-in function ne>, 'dev')
    """
    regexp = r"([<>=!]+)([\w\.]+)"
    m = re.search(regexp, ver_spec.replace(" ", ""))

    if m is None:
        raise ValueError(
            "Invalid value for `ver_spec`: '{}'. Must match this regular expression: '{}'".format(
                ver_spec, regexp,
            )
        )

    return str_to_operator(m.group(1)), m.group(2)


def process_requirements(requirements, version=None):
    """
    >>> process_requirements(None)
    []
    >>> process_requirements(["foo"])
    ['foo']
    >>> process_requirements({"== 0.1": ["foo"]}, "0.1")
    ['foo']
    >>> process_requirements({"< 0.2": ["foo"]}, "0.1")
    ['foo']
    >>> process_requirements({"> 0.1, != 0.2": ["foo"]}, "0.3")
    ['foo']
    >>> process_requirements({"== 0.1": ["foo"], "== 0.2": ["bar"]}, "0.2")
    ['bar']
    >>> process_requirements({"== dev": ["foo"]}, "0.1")
    []
    >>> process_requirements({"< dev": ["foo"]}, "0.1")
    ['foo']
    >>> process_requirements({"> 0.1": ["foo"]}, "dev")
    ['foo']
    >>> process_requirements({"== dev": ["foo"]}, "dev")
    ['foo']
    >>> process_requirements({"> 0.1, != dev": ["foo"]}, "dev")
    []
    """
    if requirements is None:
        return []

    if isinstance(requirements, list):
        return requirements

    dev_numeric = "9999.9999.9999"

    if version == "dev":
        version = dev_numeric

    if isinstance(requirements, dict):
        for ver_spec, packages in requirements.items():
            op_and_ver_pairs = map(get_operator_and_version, ver_spec.split(","))
            match_all = all(
                op(LooseVersion(version), LooseVersion(dev_numeric if ver == "dev" else ver))
                for op, ver in op_and_ver_pairs
            )
            if match_all:
                return packages
        else:
            return []

    raise TypeError("Should not reach here")


def remove_comments(s):
    return "\n".join(l for l in s.strip().split("\n") if not l.strip().startswith("#"))


def main():
    args = parse_args()

    print(divider("Parameters"))
    print(json.dumps(vars(args), indent=2))

    print(divider("Log"))
    config = read_yaml(CONFIG_PATH)
    try:
        config_ref = read_yaml(args.ref_config)
    except Exception as e:
        print("Failed to read '{}' due to: '{}'".format(args.ref_config, e))
        config_ref = {}

    flavors = set(x.split("-")[0] for x in config.keys())
    changed_flavors = get_changed_flavors(args.changed_files, flavors)

    job_names = []
    includes = []
    should_include_all_items = not args.diff_only

    for flavor, cfgs in config.items():
        package_info = cfgs.pop("package_info")

        should_include_all_items_in_this_flavor = (
            should_include_all_items
            or any(flavor.startswith(x) for x in changed_flavors)
            or (flavor not in config_ref)
            or (package_info != config_ref[flavor]["package_info"])
        )

        for key, cfg in cfgs.items():
            if (
                should_include_all_items_in_this_flavor
                or (key not in config_ref[flavor])
                or (cfg != config_ref[flavor][key])
            ):
                print("Processing {}.{}".format(flavor, key))

                # released versions
                all_versions = get_released_versions(package_info["pip_release"])
                versions = filter_versions(
                    all_versions, cfg["minimum"], cfg["maximum"], cfg.get("unsupported"),
                )
                versions = select_latest_micro_versions(versions)
                for ver in versions:
                    job_name = " / ".join([flavor, ver, key])
                    job_names.append(job_name)
                    requirements = process_requirements(cfg.get("requirements"), ver)
                    install = "pip install -U '{}=={}'".format(package_info["pip_release"], ver)
                    run = remove_comments(cfg["run"])
                    includes.append(
                        {
                            "job_name": job_name,
                            "requirements": requirements or None,
                            "install": install,
                            "run": run,
                        }
                    )

                # development version
                if "pip_dev" in package_info:
                    job_name = " / ".join([flavor, "dev", key])
                    job_names.append(job_name)
                    requirements = process_requirements(cfg.get("requirements"), "dev")
                    install = remove_comments(package_info["pip_dev"])
                    run = remove_comments(cfg["run"])
                    includes.append(
                        {
                            "job_name": job_name,
                            "requirements": requirements or None,
                            "install": install,
                            "run": run,
                        }
                    )

    matrix = {"job_name": job_names, "include": includes}
    print(divider("Result"))
    # specify `indent` to prettify the output
    print(json.dumps(matrix, indent=2))

    if "GITHUB_ACTIONS" in os.environ:
        # this line prints out nothing in the console on github actions
        print("::set-output name=matrix::{}".format(json.dumps(matrix)))


if __name__ == "__main__":
    main()
