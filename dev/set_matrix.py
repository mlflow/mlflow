"""
A script to set a matrix for the cross version tests for MLflow Models / autologging integrations.

# How to run:

```
# ===== Include all items =====

python dev/set_matrix.py

# ===== Include only `ml-package-versions.yml` updates =====

REF_VERSIONS_YAML="https://raw.githubusercontent.com/mlflow/mlflow/master/ml-package-versions.yml"
python dev/set_matrix.py --ref-versions-yaml $REF_VERSIONS_YAML

# ===== Include only flavor file updates =====

CHANGED_FILES="
mlflow/keras.py
mlflow/tensorlfow/__init__.py
"
python dev/set_matrix.py --changed-files $CHANGED_FILES

# ===== Include both `ml-package-versions.yml` & flavor file updates =====

python dev/set_matrix.py --ref-versions-yaml $REF_VERSIONS_YAML --changed-files $CHANGED_FILES
```

# How to run doctests:

```
pytest dev/set_matrix.py --doctest-modules --verbose
```
"""

import argparse
from packaging.version import Version
import json
import operator
import os
import re
import shutil
import urllib.request

import yaml

VERSIONS_YAML_PATH = "ml-package-versions.yml"
DEV_VERSION = "dev"


def read_yaml(location, if_error=None):
    """
    Reads a YAML file.

    Examples
    --------
    >>> read_yaml("https://raw.githubusercontent.com/mlflow/mlflow/master/.circleci/config.yml")
    {...}
    >>> read_yaml(".circleci/config.yml")
    {...}
    >>> read_yaml("non_existent.yml", if_error={})
    Failed to read ...
    {}
    """
    try:
        if re.search("^https?://", location):
            with urllib.request.urlopen(location) as f:
                return yaml.load(f, Loader=yaml.SafeLoader)
        else:
            with open(location) as f:
                return yaml.load(f, Loader=yaml.SafeLoader)
    except Exception as e:
        if if_error is not None:
            print("Failed to read '{}' due to: `{}`".format(location, e))
            return if_error

        raise


def get_released_versions(package_name):
    """
    Fetches the released versions & datetimes of the specified Python package.

    Examples
    --------
    >>> get_released_versions("scikit-learn")
    {'0.10': '2012-01-11T14:42:25', '0.11': '2012-05-08T00:40:14', ...}
    """
    url = "https://pypi.python.org/pypi/{}/json".format(package_name)
    data = json.load(urllib.request.urlopen(url))

    versions = {
        # We can actually select any element in `dist_files` because all the distribution files
        # should have almost the same upload time.
        version: dist_files[0]["upload_time"]
        for version, dist_files in data["releases"].items()
        # If len(dist_files) = 0, this release is unavailable.
        # Example: https://pypi.org/project/xgboost/0.7
        #
        # > pip install 'xgboost==0.7'
        # ERROR: Could not find a version that satisfies the requirement xgboost==0.7
        if len(dist_files) > 0
    }
    return versions


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
        # Sort by (minor_version, upload_time) in descending order
        key=lambda x: (Version(x[0]).release[:2], x[1]),
        reverse=True,
    ):
        minor_ver = Version(ver).release[:2]

        if minor_ver not in seen_minors:
            seen_minors.add(minor_ver)
            res.insert(0, ver)

    return res


def filter_versions(versions, min_ver, max_ver, excludes=None):
    """
    Filter versions that satisfy the following conditions:

    1. is a final or post release that PEP 440 defines
    2. is newer than or equal to `min_ver`
    3. shares the same major version as `max_ver` or `min_ver`
    4. (Optional) is not in `excludes`

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

    # Prevent specifying non-existent versions
    assert min_ver in versions
    assert max_ver in versions
    assert all(v in versions for v in excludes)

    versions = {Version(v): t for v, t in versions.items() if v not in excludes}

    def _is_final_or_post_release(v):
        # final release: https://www.python.org/dev/peps/pep-0440/#final-releases
        # post release: https://www.python.org/dev/peps/pep-0440/#post-releases
        return (v.base_version == v.public) or (v.is_postrelease)

    versions = {v: t for v, t in versions.items() if _is_final_or_post_release(v)}
    versions = {v: t for v, t in versions.items() if v.major <= Version(max_ver).major}
    versions = {str(v): t for v, t in versions.items() if v >= Version(min_ver)}

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
    >>> get_changed_flavors(["tests/xgboost/test_xxx.py"], flavors)
    ['xgboost']
    >>> get_changed_flavors(["tests/xgboost_autolog/test_xxx.py"], flavors)
    ['xgboost']
    >>> get_changed_flavors(["tests/xgboost_autologging/test_xxx.py"], flavors)
    ['xgboost']
    >>> get_changed_flavors(["README.rst"], flavors)
    []
    >>> get_changed_flavors([], flavors)
    []
    """
    changed_flavors = []
    for f in changed_files:
        pattern = r"^(mlflow|tests)/(.+?)(_autolog(ging)?)?(\.py|/)"
        #                           ~~~~~
        #                           # This group captures a flavor name
        match = re.search(pattern, f)

        if (match is not None) and (match.group(2) in flavors):
            changed_flavors.append(match.group(2))

    return changed_flavors


def str_to_operator(s):
    """
    Turns a string into the corresponding operator.

    Examples
    --------
    >>> str_to_operator("<")(1, 2)  # equivalent to '1 < 2'
    True
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

    Examples
    --------
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
    Examples
    --------
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

    if isinstance(requirements, dict):
        # The version "dev" should always compare as greater than any exisiting versions.
        dev_numeric = "9999.9999.9999"

        if version == DEV_VERSION:
            version = dev_numeric

        for ver_spec, packages in requirements.items():
            op_and_ver_pairs = map(get_operator_and_version, ver_spec.split(","))
            match_all = all(
                comp_op(
                    Version(version), Version(dev_numeric if req_ver == DEV_VERSION else req_ver),
                )
                for comp_op, req_ver in op_and_ver_pairs
            )
            if match_all:
                return packages
        else:
            return []

    raise TypeError("Invalid object type for `requirements`: '{}'".format(type(requirements)))


def remove_comments(s):
    """
    Examples
    --------
    >>> code = '''
    ... # comment 1
    ...  # comment 2
    ... echo foo
    ... '''
    >>> remove_comments(code)
    'echo foo'
    """
    return "\n".join(l for l in s.strip().split("\n") if not l.strip().startswith("#"))


def make_pip_install_command(packages):
    """
    Examples
    --------
    >>> make_pip_install_command(["foo", "bar"])
    "pip install 'foo' 'bar'"
    """
    return "pip install " + " ".join("'{}'".format(x) for x in packages)


def divider(title, length=None):
    r"""
    Examples
    --------
    >>> divider("1234", 20)
    '\n======= 1234 ======='
    """
    length = shutil.get_terminal_size(fallback=(80, 24))[0] if length is None else length
    rest = length - len(title) - 2
    left = rest // 2 if rest % 2 else (rest + 1) // 2
    return "\n{} {} {}".format("=" * left, title, "=" * (rest - left))


def parse_args():
    parser = argparse.ArgumentParser(description="Set a test matrix for the cross version tests")
    parser.add_argument(
        "--ref-versions-yaml",
        required=False,
        default=None,
        help=(
            "URL or local file path of the reference config which will be compared with the config "
            "on the branch where this script is running in order to identify version YAML updates"
        ),
    )
    parser.add_argument(
        "--changed-files",
        type=lambda x: [] if x.strip() == "" else x.strip().split("\n"),
        required=False,
        default=None,
        help=("A string that represents a list of changed files"),
    )

    return parser.parse_args()


class Hashabledict(dict):
    def __hash__(self):
        return hash(frozenset(self))


def expand_config(config):
    matrix = []
    for flavor_key, cfgs in config.items():
        flavor = flavor_key.split("-")[0]
        package_info = cfgs.pop("package_info")
        all_versions = get_released_versions(package_info["pip_release"])

        for key, cfg in cfgs.items():
            print("Processing", flavor_key, key)
            # Released versions
            versions = filter_versions(
                all_versions, cfg["minimum"], cfg["maximum"], cfg.get("unsupported"),
            )
            versions = select_latest_micro_versions(versions)

            # Explicitly include the minimum supported version
            if cfg["minimum"] not in versions:
                versions.append(cfg["minimum"])

            for ver in versions:
                job_name = " / ".join([flavor_key, ver, key])
                requirements = ["{}=={}".format(package_info["pip_release"], ver)]
                requirements.extend(process_requirements(cfg.get("requirements"), ver))
                install = make_pip_install_command(requirements)
                run = remove_comments(cfg["run"])

                matrix.append(
                    Hashabledict(flavor=flavor, job_name=job_name, install=install, run=run,)
                )

            # Development version
            if "install_dev" in package_info:
                job_name = " / ".join([flavor_key, DEV_VERSION, key])
                requirements = process_requirements(cfg.get("requirements"), DEV_VERSION)
                install = (
                    make_pip_install_command(requirements) + "\n" if requirements else ""
                ) + remove_comments(package_info["install_dev"])
                run = remove_comments(cfg["run"])

                matrix.append(
                    Hashabledict(flavor=flavor, job_name=job_name, install=install, run=run,)
                )
    return matrix


def main():
    args = parse_args()

    print(divider("Parameters"))
    print(json.dumps(vars(args), indent=2))

    print(divider("Logs"))
    changed_files = [] if (args.changed_files is None) else args.changed_files
    config = read_yaml(VERSIONS_YAML_PATH)
    config_ref = (
        {} if args.ref_versions_yaml is None else read_yaml(args.ref_versions_yaml, if_error={})
    )

    # Assuming that the top-level keys in `ml-package-versions.yml` have the format:
    # <flavor name>(-<suffix>) (e.g. sklearn, tensorflow-1.x, keras-tf1.x)
    flavors = set(x.split("-")[0] for x in config.keys())
    changed_flavors = get_changed_flavors(changed_files, flavors)

    matrix = set(expand_config(config))
    matrix_ref = set(expand_config(config_ref))

    diff_config = (
        set()
        if (args.changed_files is not None and args.ref_versions_yaml is None)
        else matrix.difference(matrix_ref)
    )
    diff_flavor = set(filter(lambda x: x["flavor"] in changed_flavors, matrix))

    # If this file contains changes, re-run all the tests, otherwise re-run the affected tests.
    include = matrix if (__file__ in changed_files) else diff_config.union(diff_flavor)
    include = sorted(include, key=lambda x: x["job_name"])
    job_names = [x["job_name"] for x in include]

    matrix = {"job_name": job_names, "include": include}
    print(divider("Result"))
    print(json.dumps(matrix, indent=2))

    if "GITHUB_ACTIONS" in os.environ:
        # `::set-output` is a special syntax for GitHub Actions to set an action's output parameter.
        # https://docs.github.com/en/free-pro-team@latest/actions/reference/workflow-commands-for-github-actions#setting-an-output-parameter # noqa
        # Note that this actually doesn't print anything to the console.
        print("::set-output name=matrix::{}".format(json.dumps(matrix)))

        # Set a flag that indicates whether or not the matrix is empty. If this flag is 'true',
        # skip the subsequent jobs.
        print("::set-output name=is_matrix_empty::{}".format("false" if job_names else "true"))


if __name__ == "__main__":
    main()
