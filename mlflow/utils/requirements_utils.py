"""
This module provides a set of utilities for interpreting and creating requirements files
(e.g. pip's `requirements.txt`), which is useful for managing ML software environments.
"""
import os
from itertools import filterfalse
from collections import namedtuple

from packaging.version import Version


def _is_comment(line):
    return line.startswith("#")


def _is_empty(line):
    return line == ""


def _strip_inline_comment(line):
    return line[: line.find(" #")].rstrip() if " #" in line else line


def _is_requirements_file(line):
    return line.startswith("-r ") or line.startswith("--requirement ")


def _is_constraints_file(line):
    return line.startswith("-c ") or line.startswith("--constraint ")


def _join_continued_lines(lines):
    """
    Joins lines ending with '\\'.

    >>> _join_continued_lines["a\\", "b\\", "c"]
    >>> 'abc'
    """
    continued_lines = []

    for line in lines:
        if line.endswith("\\"):
            continued_lines.append(line.rstrip("\\"))
        else:
            continued_lines.append(line)
            yield "".join(continued_lines)
            continued_lines.clear()

    # The last line ends with '\'
    if continued_lines:
        yield "".join(continued_lines)


# Represents a pip requirement.
#
# :param req_str: A requirement string (e.g. "scikit-learn == 0.24.2").
# :param is_constraint: A boolean indicating whether this requirement is a constraint.
_Requirement = namedtuple("_Requirement", ["req_str", "is_constraint"])


def _parse_requirements(requirements_file, is_constraint):
    """
    A simplified version of `pip._internal.req.parse_requirements` which performs the following
    operations on the given requirements file and yields the parsed requirements.

    - Remove comments and blank lines
    - Join continued lines
    - Resolve requirements file references (e.g. '-r requirements.txt')
    - Resolve constraints file references (e.g. '-c constraints.txt')

    :param requirements_file: A string path to a requirements file on the local filesystem.
    :param is_constraint: Indicates the parsed requirements file is a constraint file.
    :return: A list of ``_Requirement`` instances.

    References:
    - `pip._internal.req.parse_requirements`:
      https://github.com/pypa/pip/blob/7a77484a492c8f1e1f5ef24eaf71a43df9ea47eb/src/pip/_internal/req/req_file.py#L118
    - Requirements File Format:
      https://pip.pypa.io/en/stable/cli/pip_install/#requirements-file-format
    - Constraints Files:
      https://pip.pypa.io/en/stable/user_guide/#constraints-files
    """
    with open(requirements_file) as f:
        lines = f.read().splitlines()

    lines = map(str.strip, lines)
    lines = map(_strip_inline_comment, lines)
    lines = _join_continued_lines(lines)
    lines = filterfalse(_is_comment, lines)
    lines = filterfalse(_is_empty, lines)

    for line in lines:
        if _is_requirements_file(line):
            req_file = line.split(maxsplit=1)[1]
            # If `req_file` is an absolute path, `os.path.join` returns `req_file`:
            # https://docs.python.org/3/library/os.path.html#os.path.join
            abs_path = os.path.join(os.path.dirname(requirements_file), req_file)
            yield from _parse_requirements(abs_path, is_constraint=False)
        elif _is_constraints_file(line):
            req_file = line.split(maxsplit=1)[1]
            abs_path = os.path.join(os.path.dirname(requirements_file), req_file)
            yield from _parse_requirements(abs_path, is_constraint=True)
        else:
            yield _Requirement(line, is_constraint)


def _strip_local_version_identifier(version):
    """
    Strips a local version identifer in `version`.

    Local version identifiers:
    https://www.python.org/dev/peps/pep-0440/#local-version-identifiers

    :param version: A version string to strip.
    """

    class IgnoreLocal(Version):
        @property
        def local(self):
            return None

    return str(IgnoreLocal(version))


def _get_installed_version(module):
    """
    Returns the installed version of the specified module.

    :param module: The name of the module.
    """
    return __import__(module).__version__


def _get_pinned_requirement(package, version=None, module=None):
    """
    Returns a string representing a pinned pip requirement to install the specified package and
    version (e.g. 'mlflow==1.2.3').

    :param package: The name of the package.
    :param version: The version of the package. If None, defaults to the installed version.
    :param module: The name of the top-level module provided by the package . For example,
                   if `package` is 'scikit-learn', `module` should be 'sklearn'. If None, defaults
                   to `package`.
    """
    module = module or package
    version = version or _get_installed_version(module)
    version = _strip_local_version_identifier(version)
    return f"{package}=={version}"
