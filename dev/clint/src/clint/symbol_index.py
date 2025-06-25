from __future__ import annotations

import ast
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


class ModuleSymbolExtractor(ast.NodeVisitor):
    """Extracts function definitions and import mappings from a Python module."""

    def __init__(self, mod: str) -> None:
        self.mod = mod
        self.import_mapping: dict[str, str] = {}
        self.func_mapping: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

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
        self.func_mapping[f"{self.mod}.{node.name}"] = node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name.startswith("_"):
            return
        self.func_mapping[f"{self.mod}.{node.name}"] = node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                self.func_mapping[f"{self.mod}.{node.name}"] = item


def extract_symbols_from_file(
    py_file: Path,
) -> tuple[dict[str, str], dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] | None:
    """Extract function definitions and import mappings from a Python file."""
    p = Path(py_file)
    if not p.parts or p.parts[0] != "mlflow":
        return None

    try:
        tree = ast.parse(p.read_text())
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
        func_mapping: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> None:
        self.import_mapping = import_mapping
        self.func_mapping = func_mapping

    @classmethod
    def build(cls) -> SymbolIndex:
        mapping: dict[str, str] = {}
        func_mapping: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

        py_files = subprocess.check_output(
            ["git", "ls-files", "mlflow/*.py"], text=True
        ).splitlines()

        max_workers = min(multiprocessing.cpu_count(), len(py_files))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(extract_symbols_from_file, f): f for f in map(Path, py_files)
            }
            for future in as_completed(futures):
                if result := future.result():
                    file_imports, file_funcs = result
                    mapping.update(file_imports)
                    func_mapping.update(file_funcs)

        return cls(mapping, func_mapping)

    def resolve_symbol(self, target: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """Resolve a symbol to its actual definition, following import chains."""
        if f := self.func_mapping.get(target):
            return f

        while v := self.import_mapping.get(target):
            target = v

        if target and (f := self.func_mapping.get(target)):
            return f

        return None


def main():
    symbol_index = SymbolIndex.build()
    target = "mlflow.MlflowClient"
    if func := symbol_index.resolve_symbol(target):
        print(f"Function {target} found at line {func.lineno}")


if __name__ == "__main__":
    main()
