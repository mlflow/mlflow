import operator
import importlib.metadata
from packaging.version import Version

operators = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def compare_package_version(package: str, version: str, cmp: str) -> bool:
    """
    Compare the version of a package with a given version.
    :param package: The name of the package.
    :param version: The version to compare with.
    :param cmp: The comparison operator.
    :return: True if the version of the package satisfies the comparison.
    """

    if cmp not in operators:
        raise ValueError(f"Invalid comparison operator: {cmp}")
    op = operators[cmp]
    return op(Version(importlib.metadata.version(package)), Version(version))
