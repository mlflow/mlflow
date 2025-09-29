"""Symbol indexing for MLflow codebase.

This module provides efficient indexing and lookup of Python symbols (functions, classes)
across the MLflow codebase using AST parsing and parallel processing.

Key components:
- FunctionInfo: Lightweight function signature information
- ModuleSymbolExtractor: AST visitor for extracting symbols from modules
- SymbolIndex: Main index class for symbol resolution and lookup

Example usage:

```python
# Build an index of all MLflow symbols
index = SymbolIndex.build()

# Look up function signature information
func_info = index.resolve("mlflow.log_metric")
print(f"Arguments: {func_info.args}")  # -> ['key, 'value', 'step', ...]
```
"""

import ast
import multiprocessing
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Self

from clint.utils import get_repo_root


@dataclass
class FunctionInfo:
    """Lightweight function signature information for efficient serialization."""

    has_vararg: bool  # *args
    has_kwarg: bool  # **kwargs
    args: list[str] = field(default_factory=list)  # Regular arguments
    kwonlyargs: list[str] = field(default_factory=list)  # Keyword-only arguments
    posonlyargs: list[str] = field(default_factory=list)  # Positional-only arguments

    @classmethod
    def from_func_def(
        cls, node: ast.FunctionDef | ast.AsyncFunctionDef, skip_self: bool = False
    ) -> Self:
        """Create FunctionInfo from an AST function definition node."""
        args = node.args.args
        if skip_self and args:
            args = args[1:]  # Skip 'self' for methods

        return cls(
            has_vararg=node.args.vararg is not None,
            has_kwarg=node.args.kwarg is not None,
            args=[arg.arg for arg in args],
            kwonlyargs=[arg.arg for arg in node.args.kwonlyargs],
            posonlyargs=[arg.arg for arg in node.args.posonlyargs],
        )

    @property
    def all_args(self) -> list[str]:
        return self.posonlyargs + self.args + self.kwonlyargs


class ModuleSymbolExtractor(ast.NodeVisitor):
    """Extracts function definitions and import mappings from a Python module."""

    def __init__(self, mod: str) -> None:
        self.mod = mod
        self.import_mapping: dict[str, str] = {}
        self.func_mapping: dict[str, FunctionInfo] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not alias.name.startswith("mlflow."):
                continue
            if alias.asname:
                self.import_mapping[f"{self.mod}.{alias.asname}"] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None or not node.module.startswith("mlflow."):
            return
        for alias in node.names:
            if alias.name.startswith("_"):
                continue
            if alias.asname:
                self.import_mapping[f"{self.mod}.{alias.asname}"] = f"{node.module}.{alias.name}"
            else:
                self.import_mapping[f"{self.mod}.{alias.name}"] = f"{node.module}.{alias.name}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("_"):
            return
        self.func_mapping[f"{self.mod}.{node.name}"] = FunctionInfo.from_func_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name.startswith("_"):
            return
        self.func_mapping[f"{self.mod}.{node.name}"] = FunctionInfo.from_func_def(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                if stmt.name == "__init__":
                    info = FunctionInfo.from_func_def(stmt, skip_self=True)
                    self.func_mapping[f"{self.mod}.{node.name}"] = info
                elif any(
                    isinstance(deco, ast.Name) and deco.id in ("classmethod", "staticmethod")
                    for deco in stmt.decorator_list
                ):
                    info = FunctionInfo.from_func_def(stmt, skip_self=True)
                    self.func_mapping[f"{self.mod}.{node.name}.{stmt.name}"] = info
        else:
            # If no __init__ found, still add the class with *args and **kwargs
            self.func_mapping[f"{self.mod}.{node.name}"] = FunctionInfo(
                has_vararg=True, has_kwarg=True
            )


def extract_symbols_from_file(
    rel_path: str, content: str
) -> tuple[dict[str, str], dict[str, FunctionInfo]] | None:
    """Extract function definitions and import mappings from a Python file."""
    p = Path(rel_path)
    if not p.parts or p.parts[0] != "mlflow":
        return None

    try:
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError):
        return None

    mod_name = (
        ".".join(p.parts[:-1]) if p.name == "__init__.py" else ".".join([*p.parts[:-1], p.stem])
    )

    extractor = ModuleSymbolExtractor(mod_name)
    extractor.visit(tree)
    return extractor.import_mapping, extractor.func_mapping


class SymbolIndex:
    """Index of all symbols (functions, classes) in the MLflow codebase."""

    def __init__(
        self,
        import_mapping: dict[str, str],
        func_mapping: dict[str, FunctionInfo],
    ) -> None:
        self.import_mapping = import_mapping
        self.func_mapping = func_mapping

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump((self.import_mapping, self.func_mapping), f)

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as f:
            import_mapping, func_mapping = pickle.load(f)
        return cls(import_mapping, func_mapping)

    @classmethod
    def build(cls) -> Self:
        repo_root = get_repo_root()
        py_files = subprocess.check_output(
            ["git", "-C", repo_root, "ls-files", "mlflow/*.py"], text=True
        ).splitlines()

        mapping: dict[str, str] = {}
        func_mapping: dict[str, FunctionInfo] = {}

        # Ensure at least 1 worker to avoid ProcessPoolExecutor ValueError
        max_workers = max(1, min(multiprocessing.cpu_count(), len(py_files)))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for py_file in py_files:
                abs_file_path = repo_root / py_file
                if not abs_file_path.exists():
                    continue
                content = abs_file_path.read_text()
                f = executor.submit(extract_symbols_from_file, py_file, content)
                futures[f] = py_file

            for future in as_completed(futures):
                if result := future.result():
                    file_imports, file_functions = result
                    mapping.update(file_imports)
                    func_mapping.update(file_functions)

        return cls(mapping, func_mapping)

    def _resolve_import(self, target: str) -> str:
        resolved = target
        while v := self.import_mapping.get(resolved):
            resolved = v
        return resolved

    def resolve(self, target: str) -> FunctionInfo | None:
        """Resolve a symbol to its actual definition, following import chains."""
        if f := self.func_mapping.get(target):
            return f

        resolved = self._resolve_import(target)
        if f := self.func_mapping.get(resolved):
            return f

        target, tail = target.rsplit(".", 1)
        resolved = self._resolve_import(target)
        if f := self.func_mapping.get(f"{resolved}.{tail}"):
            return f

        return None
