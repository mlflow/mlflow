import ast
from typing import TYPE_CHECKING

from clint.rules.base import Rule

if TYPE_CHECKING:
    from clint.resolver import Resolver


class UnsafeVersionParse(Rule):
    # Names/attributes that hold a raw Databricks runtime (DBR) version string. These are NOT
    # PEP 440 (e.g. "18.x-aarch64-photon-scala2") and crash `Version(...)` with `InvalidVersion`.
    _DBR_VERSION_NAMES = frozenset({
        "runtime_version",
        "dbr_version",
        "databricks_runtime",
        "databricks_runtime_version",
    })

    def _message(self) -> str:
        return (
            "Do not pass an untrusted version string directly to `packaging.version.Version(...)`. "
            "It raises on missing/non-PEP440 input, which crashes on Databricks Serverless. "
            "For an installed distribution's version, use "
            "`mlflow.utils.get_installed_version(...)`. For a Databricks runtime (DBR) version "
            "string, use `mlflow.utils.databricks_utils.parse_dbr_runtime_major_minor(...)`."
        )

    @staticmethod
    def check(node: ast.Call, resolver: "Resolver") -> bool:
        if not UnsafeVersionParse._is_version_call(node, resolver):
            return False

        match node.args:
            case [arg]:
                return UnsafeVersionParse._is_metadata_version_call(
                    arg, resolver
                ) or UnsafeVersionParse._is_dbr_version_name(arg)

        return False

    @staticmethod
    def _is_version_call(node: ast.Call, resolver: "Resolver") -> bool:
        if resolved := resolver.resolve(node.func):
            return resolved == ["packaging", "version", "Version"]
        return False

    @staticmethod
    def _is_metadata_version_call(arg: ast.expr, resolver: "Resolver") -> bool:
        if not isinstance(arg, ast.Call):
            return False
        if resolved := resolver.resolve(arg.func):
            return resolved in (
                ["importlib", "metadata", "version"],
                ["importlib_metadata", "version"],
            )
        return False

    @staticmethod
    def _is_dbr_version_name(arg: ast.expr) -> bool:
        # Match both a bare name (`runtime_version`) and an attribute access
        # (`udf_sandbox_info.runtime_version`).
        if isinstance(arg, ast.Name):
            return arg.id in UnsafeVersionParse._DBR_VERSION_NAMES
        if isinstance(arg, ast.Attribute):
            return arg.attr in UnsafeVersionParse._DBR_VERSION_NAMES
        return False
