from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def is_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


@dataclass
class Error:
    file_path: Path
    line: int
    column: int
    lines: list[str]

    def format(self, github: bool = False) -> str:
        message = " ".join(self.lines)
        if github:
            return f"::warning file={self.file_path},line={self.line},col={self.column}::{message}"
        else:
            return f"{self.file_path}:{self.line}:{self.column}: {message}"


@dataclass
class Parameter:
    name: str
    position: int | None  # None for keyword-only
    is_required: bool
    is_positional_only: bool
    is_keyword_only: bool
    lineno: int
    col_offset: int


@dataclass
class Signature:
    positional: list[Parameter]  # Includes positional-only and regular positional
    keyword_only: list[Parameter]
    has_var_positional: bool  # *args
    has_var_keyword: bool  # **kwargs


@dataclass
class ParameterError:
    message: str
    param_name: str
    lineno: int
    col_offset: int


def parse_signature(args: ast.arguments) -> Signature:
    """Convert ast.arguments to a Signature dataclass for easier processing."""
    parameters_positional: list[Parameter] = []
    parameters_keyword_only: list[Parameter] = []

    # Process positional-only parameters
    for i, arg in enumerate(args.posonlyargs):
        parameters_positional.append(
            Parameter(
                name=arg.arg,
                position=i,
                is_required=True,  # All positional-only are required
                is_positional_only=True,
                is_keyword_only=False,
                lineno=arg.lineno,
                col_offset=arg.col_offset,
            )
        )

    # Process regular positional parameters
    offset = len(args.posonlyargs)
    first_optional_idx = len(args.posonlyargs + args.args) - len(args.defaults)

    for i, arg in enumerate(args.args):
        pos = offset + i
        parameters_positional.append(
            Parameter(
                name=arg.arg,
                position=pos,
                is_required=pos < first_optional_idx,
                is_positional_only=False,
                is_keyword_only=False,
                lineno=arg.lineno,
                col_offset=arg.col_offset,
            )
        )

    # Process keyword-only parameters
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters_keyword_only.append(
            Parameter(
                name=arg.arg,
                position=None,
                is_required=default is None,
                is_positional_only=False,
                is_keyword_only=True,
                lineno=arg.lineno,
                col_offset=arg.col_offset,
            )
        )

    return Signature(
        positional=parameters_positional,
        keyword_only=parameters_keyword_only,
        has_var_positional=args.vararg is not None,
        has_var_keyword=args.kwarg is not None,
    )


def check_signature_compatibility(
    old_fn: ast.FunctionDef | ast.AsyncFunctionDef,
    new_fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ParameterError]:
    """
    Return list of error messages when *new_fn* is not backward-compatible with *old_fn*,
    or None if compatible.

    Compatibility rules
    -------------------
    • Positional / positional-only parameters
        - Cannot be reordered, renamed, or removed.
        - Adding **required** ones is breaking.
        - Adding **optional** ones is allowed only at the end.
        - Making an optional parameter required is breaking.

    • Keyword-only parameters (order does not matter)
        - Cannot be renamed or removed.
        - Making an optional parameter required is breaking.
        - Adding a required parameter is breaking; adding an optional parameter is fine.
    """
    old_sig = parse_signature(old_fn.args)
    new_sig = parse_signature(new_fn.args)
    errors: list[ParameterError] = []

    # ------------------------------------------------------------------ #
    # 1. Positional / pos-only parameters
    # ------------------------------------------------------------------ #

    # (a) existing parameters must line up
    for idx, old_param in enumerate(old_sig.positional):
        if idx >= len(new_sig.positional):
            errors.append(
                ParameterError(
                    message=f"Positional param '{old_param.name}' was removed.",
                    param_name=old_param.name,
                    lineno=old_param.lineno,
                    col_offset=old_param.col_offset,
                )
            )
            continue

        new_param = new_sig.positional[idx]
        if old_param.name != new_param.name:
            errors.append(
                ParameterError(
                    message=(
                        f"Positional param order/name changed: "
                        f"'{old_param.name}' -> '{new_param.name}'."
                    ),
                    param_name=new_param.name,
                    lineno=new_param.lineno,
                    col_offset=new_param.col_offset,
                )
            )
            # Stop checking further positional params after first order/name mismatch
            break

        if (not old_param.is_required) and new_param.is_required:
            errors.append(
                ParameterError(
                    message=f"Optional positional param '{old_param.name}' became required.",
                    param_name=new_param.name,
                    lineno=new_param.lineno,
                    col_offset=new_param.col_offset,
                )
            )

    # (b) any extra new positional params must be optional and appended
    if len(new_sig.positional) > len(old_sig.positional):
        for idx in range(len(old_sig.positional), len(new_sig.positional)):
            new_param = new_sig.positional[idx]
            if new_param.is_required:
                errors.append(
                    ParameterError(
                        message=f"New required positional param '{new_param.name}' added.",
                        param_name=new_param.name,
                        lineno=new_param.lineno,
                        col_offset=new_param.col_offset,
                    )
                )

    # ------------------------------------------------------------------ #
    # 2. Keyword-only parameters (order-agnostic)
    # ------------------------------------------------------------------ #
    old_kw_names = {p.name for p in old_sig.keyword_only}
    new_kw_names = {p.name for p in new_sig.keyword_only}

    # Build mappings for easier lookup
    old_kw_by_name = {p.name: p for p in old_sig.keyword_only}
    new_kw_by_name = {p.name: p for p in new_sig.keyword_only}

    # removed or renamed
    for name in old_kw_names - new_kw_names:
        old_param = old_kw_by_name[name]
        errors.append(
            ParameterError(
                message=f"Keyword-only param '{name}' was removed.",
                param_name=name,
                lineno=old_param.lineno,
                col_offset=old_param.col_offset,
            )
        )

    # optional -> required upgrades
    for name in old_kw_names & new_kw_names:
        if not old_kw_by_name[name].is_required and new_kw_by_name[name].is_required:
            new_param = new_kw_by_name[name]
            errors.append(
                ParameterError(
                    message=f"Keyword-only param '{name}' became required.",
                    param_name=name,
                    lineno=new_param.lineno,
                    col_offset=new_param.col_offset,
                )
            )

    # new required keyword-only params
    for param in new_sig.keyword_only:
        if param.is_required and param.name not in old_kw_names:
            errors.append(
                ParameterError(
                    message=f"New required keyword-only param '{param.name}' added.",
                    param_name=param.name,
                    lineno=param.lineno,
                    col_offset=param.col_offset,
                )
            )

    return errors


def _is_private(n: str) -> bool:
    return n.startswith("_") and not n.startswith("__") and not n.endswith("__")


class FunctionSignatureExtractor(ast.NodeVisitor):
    def __init__(self):
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.stack: list[ast.ClassDef] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Is this a private function or a function in a private class?
        # If so, skip it.
        if _is_private(node.name) or (self.stack and _is_private(self.stack[-1].name)):
            return

        names = [*(c.name for c in self.stack), node.name]
        self.functions[".".join(names)] = node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if _is_private(node.name) or (self.stack and _is_private(self.stack[-1].name)):
            return

        names = [*(c.name for c in self.stack), node.name]
        self.functions[".".join(names)] = node


def get_changed_python_files(base_branch: str = "master") -> list[Path]:
    # In GitHub Actions PR context, we need to fetch the base branch first
    if is_github_actions():
        # Fetch the base branch to ensure we have it locally
        subprocess.check_call(
            ["git", "fetch", "origin", f"{base_branch}:{base_branch}"],
        )

    result = subprocess.check_output(
        ["git", "diff", "--name-only", f"{base_branch}...HEAD"], text=True
    )
    files = [s.strip() for s in result.splitlines()]
    return [Path(f) for f in files if f]


def parse_functions(content: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(content)
    extractor = FunctionSignatureExtractor()
    extractor.visit(tree)
    return extractor.functions


def get_file_content_at_revision(file_path: Path, revision: str) -> str | None:
    try:
        return subprocess.check_output(["git", "show", f"{revision}:{file_path}"], text=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to get file content at revision: {e}", file=sys.stderr)
        return None


def compare_signatures(base_branch: str = "master") -> list[Error]:
    errors: list[Error] = []
    for file_path in get_changed_python_files(base_branch):
        # Ignore non-Python files
        if not file_path.suffix == ".py":
            continue

        # Ignore files not in the mlflow directory
        if file_path.parts[0] != "mlflow":
            continue

        # Ignore private modules
        if any(part.startswith("_") for part in file_path.parts):
            continue

        base_content = get_file_content_at_revision(file_path, base_branch)
        if base_content is None:
            # Find not found in the base branch, likely added in the current branch
            continue

        if not file_path.exists():
            # File not found, likely deleted in the current branch
            continue

        current_content = file_path.read_text()
        base_functions = parse_functions(base_content)
        current_functions = parse_functions(current_content)
        for func_name in set(base_functions.keys()) & set(current_functions.keys()):
            base_func = base_functions[func_name]
            current_func = current_functions[func_name]
            if param_errors := check_signature_compatibility(base_func, current_func):
                # Create individual errors for each problematic parameter
                for param_error in param_errors:
                    errors.append(
                        Error(
                            file_path=file_path,
                            line=param_error.lineno,
                            column=param_error.col_offset + 1,
                            lines=[
                                "[Non-blocking | Ignore if not public API]",
                                param_error.message,
                                f"This change will break existing `{func_name}` calls.",
                                "If this is not intended, please fix it.",
                            ],
                        )
                    )

    return errors


@dataclass
class Args:
    base_branch: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Check for breaking changes in Python function signatures"
    )
    parser.add_argument("--base-branch", default=os.environ.get("GITHUB_BASE_REF", "master"))
    args = parser.parse_args()
    return Args(base_branch=args.base_branch)


def main():
    args = parse_args()
    errors = compare_signatures(args.base_branch)
    for error in errors:
        print(error.format(github=is_github_actions()))


if __name__ == "__main__":
    main()
