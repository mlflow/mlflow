import ast
from typing import TYPE_CHECKING

from clint.rules.base import Rule
from clint.utils import resolve_expr

if TYPE_CHECKING:
    from clint.index import SymbolIndex


class LogModelArtifactPath(Rule):
    def _message(self) -> str:
        return "`artifact_path` parameter of `log_model` is deprecated. Use `name` instead."

    @staticmethod
    def check(node: ast.Call, index: "SymbolIndex") -> bool:
        """
        Returns True if the call looks like `mlflow.<flavor>.log_model(...)` and
        the `artifact_path` argument is specified.
        """
        parts = resolve_expr(node.func)
        if not parts or len(parts) != 3:
            return False

        first, second, third = parts
        if not (first == "mlflow" and third == "log_model"):
            return False

        # TODO: Remove this once spark flavor supports logging models as logged model artifacts
        if second == "spark":
            return False

        function_name = f"{first}.{second}.log_model"
        artifact_path_idx = LogModelArtifactPath._find_artifact_path_index(index, function_name)
        if artifact_path_idx is None:
            return False

        if len(node.args) > artifact_path_idx:
            return True
        else:
            return any(kw.arg and kw.arg == "artifact_path" for kw in node.keywords)

    @staticmethod
    def _find_artifact_path_index(index: "SymbolIndex", function_name: str) -> int | None:
        """
        Finds the index of the `artifact_path` argument in the function signature of `log_model`
        using the SymbolIndex.
        """
        if f := index.resolve(function_name):
            try:
                return f.all_args.index("artifact_path")
            except ValueError:
                return None
        return None
