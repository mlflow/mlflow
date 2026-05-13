import ast

from packaging.version import InvalidVersion, Version

from clint.resolver import Resolver
from clint.rules.base import Rule


def _is_valid_version(version: str) -> bool:
    try:
        v = Version(version)
        return not (v.is_devrelease or v.is_prerelease or v.is_postrelease)
    except InvalidVersion:
        return False


class InvalidExperimentalDecorator(Rule):
    def _message(self) -> str:
        return (
            "Invalid usage of `@experimental` decorator. It must be used with a `version` "
            "argument that is a valid semantic version string."
        )

    @staticmethod
    def check(node: ast.expr, resolver: Resolver) -> bool:
        """
        Returns True if the `@experimental` decorator from mlflow.utils.annotations is used
        incorrectly.
        """
        resolved = resolver.resolve(node)
        if not resolved:
            return False

        if resolved != ["mlflow", "utils", "annotations", "experimental"]:
            return False

        if not isinstance(node, ast.Call):
            return True

        version = next((k.value for k in node.keywords if k.arg == "version"), None)
        if version is None:
            # No `version` argument, invalid usage
            return True

        if not isinstance(version, ast.Constant) or not isinstance(version.value, str):
            # `version` is not a string literal, invalid usage
            return True

        if not _is_valid_version(version.value):
            # `version` is not a valid semantic version, # invalid usage
            return True

        return False
