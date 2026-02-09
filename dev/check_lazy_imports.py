from __future__ import annotations

import argparse
import ast
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


def is_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


@dataclass
class Warning:
    file_path: Path
    line: int
    column: int
    message: str

    def format(self, github: bool = False) -> str:
        path = self.file_path.as_posix()
        if github:
            return f"::warning file={path},line={self.line},col={self.column}::{self.message}"
        return f"{path}:{self.line}:{self.column}: {self.message}"


@dataclass
class LazyImport:
    func_name: str
    module: str
    line: int
    column: int


def _is_type_checking_guard(test: ast.expr) -> bool:
    # Match `if TYPE_CHECKING:` and `if typing.TYPE_CHECKING:`
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id in {"typing", "typing_extensions"}
    ):
        return True
    return False


def _get_module_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = "." * node.level + (node.module or "")
    return [module]


class LazyImportExtractor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.lazy_imports: dict[tuple[str, str], LazyImport] = {}
        self._scope_stack: list[str] = []
        self._func_depth: int = 0

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._scope_stack.append(node.name)
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1
        self._scope_stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        if self._func_depth > 0 and _is_type_checking_guard(node.test):
            # Skip the TYPE_CHECKING body (those imports are fine),
            # but still visit the else branch
            for child in node.orelse:
                self.visit(child)
        else:
            self.generic_visit(node)

    def _record_import(self, node: ast.Import | ast.ImportFrom) -> None:
        if self._func_depth == 0:
            return
        func_name = ".".join(self._scope_stack)
        for module in _get_module_names(node):
            key = (func_name, module)
            if key not in self.lazy_imports:
                self.lazy_imports[key] = LazyImport(
                    func_name=func_name,
                    module=module,
                    line=node.lineno,
                    column=node.col_offset,
                )

    def visit_Import(self, node: ast.Import) -> None:
        self._record_import(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record_import(node)


def extract_lazy_imports(content: str) -> dict[tuple[str, str], LazyImport]:
    tree = ast.parse(content)
    extractor = LazyImportExtractor()
    extractor.visit(tree)
    return extractor.lazy_imports


def get_changed_python_files(base_branch: str) -> list[Path]:
    if is_github_actions():
        subprocess.check_call(
            ["git", "fetch", "origin", f"{base_branch}:{base_branch}"],
        )

    result = subprocess.check_output(
        ["git", "diff", "--name-only", f"{base_branch}...HEAD"], text=True
    )
    files = [s.strip() for s in result.splitlines()]
    return [Path(f) for f in files if f]


def get_file_content_at_revision(file_path: Path, revision: str) -> str | None:
    try:
        return subprocess.check_output(["git", "show", f"{revision}:{file_path}"], text=True)
    except subprocess.CalledProcessError:
        return None


def compare_lazy_imports(
    file_path: Path,
    base_content: str | None,
    current_content: str,
) -> list[Warning]:
    current_imports = extract_lazy_imports(current_content)
    base_imports = extract_lazy_imports(base_content) if base_content else {}

    warnings: list[Warning] = []
    new_keys = set(current_imports) - set(base_imports)
    for key in sorted(
        new_keys,
        key=lambda k: (current_imports[k].line, current_imports[k].module),
    ):
        lazy = current_imports[key]
        warnings.append(
            Warning(
                file_path=file_path,
                line=lazy.line,
                column=lazy.column + 1,
                message=(
                    f"[Non-blocking] Lazy import '{lazy.module}'. Consider moving to top-level."
                ),
            )
        )
    return warnings


def check_lazy_imports(base_branch: str = "master") -> list[Warning]:
    warnings: list[Warning] = []
    for file_path in get_changed_python_files(base_branch):
        if file_path.suffix != ".py":
            continue

        if file_path.parts[0] != "mlflow":
            continue

        if not file_path.exists():
            continue

        current_content = file_path.read_text()
        base_content = get_file_content_at_revision(file_path, base_branch)
        warnings.extend(compare_lazy_imports(file_path, base_content, current_content))

    return warnings


@dataclass
class Args:
    base_branch: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Check for newly introduced lazy imports")
    parser.add_argument("--base-branch", default=os.environ.get("GITHUB_BASE_REF", "master"))
    args = parser.parse_args()
    return Args(base_branch=args.base_branch)


def main() -> None:
    args = parse_args()
    warnings = check_lazy_imports(args.base_branch)
    for warning in warnings:
        print(warning.format(github=is_github_actions()))


if __name__ == "__main__":
    main()
