from clint.builtin import BUILTIN_MODULES
from clint.rules.base import Rule

# Third-party packages that are always available as core dependencies of mlflow-tracing
# (the smallest installable unit of MLflow). Lazy imports of these packages are flagged
# the same way as stdlib lazy imports.
_ALWAYS_AVAILABLE_MODULES = {
    "cachetools",
    "packaging",
    "pydantic",
}

_LAZY_IMPORT_MODULES = BUILTIN_MODULES | _ALWAYS_AVAILABLE_MODULES


class LazyImport(Rule):
    def _message(self) -> str:
        return "This module must be imported at the top level."

    @staticmethod
    def check(module: str | None) -> bool:
        """Check if importing the given module lazily should be flagged."""
        if module is None:
            return False
        root_module = module.split(".", 1)[0]
        return root_module in _LAZY_IMPORT_MODULES
