"""
Standalone script to extract code blocks marked with :test: from Python docstrings.
This script no longer depends on Sphinx and can be run independently.
Uses a similar approach to clint for parsing code blocks.
"""

import argparse
import functools
import importlib
import inspect
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any


def get_obj_and_module(obj_path: str) -> tuple[Any, Any]:
    """Import a module and get an object from it by dotted path."""
    splits = obj_path.split(".")
    for i in reversed(range(1, len(splits) + 1)):
        try:
            maybe_module = ".".join(splits[:i])
            mod = importlib.import_module(maybe_module)
        except ImportError:
            continue
        return mod, functools.reduce(getattr, splits[i:], mod)

    raise Exception(f"Could not import {obj_path}")


def get_code_block_line(mod_file: Path, obj_line: int, lineno_in_docstring: int) -> int:
    """Calculate the actual line number of a code block in the source file."""
    with mod_file.open() as f:
        lines = f.readlines()[obj_line:]
        for offset, line in enumerate(lines):
            if line.lstrip().startswith('"""'):
                extra_offset = 0
                while re.search(r"[^\"\s]", lines[offset + extra_offset]) is None:
                    extra_offset += 1
                return obj_line + offset + extra_offset + lineno_in_docstring


# fmt: off
# This function helps understand what each variable represents in `get_code_block_line`.
def _func():           # <- obj_line
    """                  <- obj_line + offset

    Docstring            <- obj_line + offset + extra_offset

    .. code-block::      <- obj_line + offset + extra_offset + lineno_in_docstring
        :test:
        ...
    """
# fmt: on


def get_code_block_location(obj_path: str, lineno_in_docstring: int, repo_root: Path) -> str:
    """Get the file location of a code block."""
    mod, obj = get_obj_and_module(obj_path)
    abs_mod_file = Path(mod.__file__)
    rel_mod_file = abs_mod_file.relative_to(repo_root)
    obj_line = inspect.getsourcelines(obj)[1]
    code_block_line = get_code_block_line(abs_mod_file, obj_line, lineno_in_docstring)
    return f"{rel_mod_file}:{code_block_line}"


def _get_indent(s: str) -> int:
    """Get indentation level of a string."""
    return len(s) - len(s.lstrip())


_CODE_BLOCK_HEADER_REGEX = re.compile(r"^\.\.\s+code-block::\s*py(thon)?")
_CODE_BLOCK_OPTION_REGEX = re.compile(r"^:\w+:")


def _get_header_indent(s: str) -> int | None:
    """Check if line is a code-block header and return its indent level."""
    if _CODE_BLOCK_HEADER_REGEX.match(s.lstrip()):
        return _get_indent(s)
    return None


def extract_code_blocks_from_docstring(docstring: str | None) -> list[tuple[int, str]]:
    """
    Extract all code blocks marked with :test: from a docstring.
    Uses the same approach as clint for parsing code blocks.

    Returns a list of tuples: (line_number, code_content)
    """
    if not docstring:
        return []

    blocks = []
    header_indent: int | None = None
    code_lines: list[str] = []
    has_test_option = False
    code_block_lineno = 0

    line_iter = enumerate(docstring.splitlines())
    while t := next(line_iter, None):
        idx, line = t

        if code_lines:
            # We're inside a code block
            indent = _get_indent(line)
            # If we encounter a non-blank line with an indent less than or equal to the header
            # we are done parsing the code block
            if line.strip() and (header_indent is not None) and indent <= header_indent:
                if has_test_option:
                    code = textwrap.dedent("\n".join(code_lines))
                    blocks.append((code_block_lineno, code))

                # Reset state
                code_lines.clear()
                has_test_option = False
                # It's possible that another code block follows the current one
                header_indent = _get_header_indent(line)
                continue

            code_lines.append(line)

        elif header_indent is not None:
            # We found a code-block header, now advance to the code body
            # Skip options like :test:, :caption:, etc.
            while True:
                stripped = line.lstrip()
                if stripped.startswith(":test:") or stripped == ":test:":
                    has_test_option = True

                # Check if this is still an option line or blank
                if stripped and not _CODE_BLOCK_OPTION_REGEX.match(stripped):
                    # We are at the first line of the code block
                    code_lines.append(line)
                    code_block_lineno = idx + 1  # Line number in docstring (1-indexed)
                    break

                if next_line := next(line_iter, None):
                    idx, line = next_line
                else:
                    break

        else:
            # Look for code-block headers
            header_indent = _get_header_indent(line)

    # The docstring ends with a code block
    if code_lines and has_test_option:
        code = textwrap.dedent("\n".join(code_lines))
        blocks.append((code_block_lineno, code))

    return blocks


def find_all_objects_with_docstrings(module_name: str) -> list[tuple[str, int, str]]:
    """
    Find all functions and classes in a module that have docstrings with test blocks.

    Returns a list of tuples: (obj_path, line_number, code_block)
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Warning: Could not import {module_name}: {e}", file=sys.stderr)
        return []

    results = []

    # Get all members of the module
    for name, obj in inspect.getmembers(module):
        # Skip private members
        if name.startswith("_"):
            continue

        # Only process functions and classes
        if not (inspect.isfunction(obj) or inspect.isclass(obj) or inspect.ismethod(obj)):
            continue

        # Get the object's module - only process if it's from the target module
        obj_module = inspect.getmodule(obj)
        if obj_module is None or not obj_module.__name__.startswith(module_name):
            continue

        obj_path = f"{obj_module.__name__}.{name}"
        docstring = inspect.getdoc(obj)

        if docstring and (blocks := extract_code_blocks_from_docstring(docstring)):
            for line_num, code in blocks:
                results.append((obj_path, line_num, code))

        # For classes, also check their methods
        if inspect.isclass(obj):
            for method_name, method in inspect.getmembers(obj):
                if method_name.startswith("_") and method_name not in ("__init__", "__call__"):
                    continue

                if not (inspect.isfunction(method) or inspect.ismethod(method)):
                    continue

                method_module = inspect.getmodule(method)
                if method_module is None or not method_module.__name__.startswith(module_name):
                    continue

                method_path = f"{obj_path}.{method_name}"
                method_docstring = inspect.getdoc(method)

                if method_docstring and (
                    blocks := extract_code_blocks_from_docstring(method_docstring)
                ):
                    for line_num, code in blocks:
                        results.append((method_path, line_num, code))

    return results


def generate_test_file(
    obj_path: str, line_num: int, code: str, repo_root: Path, output_dir: Path
) -> Path:
    """Generate a pytest test file for a code block."""
    try:
        code_block_location = get_code_block_location(obj_path, line_num, repo_root)
    except Exception as e:
        print(f"Warning: Could not get location for {obj_path}:{line_num}: {e}", file=sys.stderr)
        code_block_location = f"{obj_path}:{line_num}"

    name = re.sub(r"[\._]+", "_", obj_path).strip("_")
    filename = f"test_{name}_{line_num}.py"
    content = textwrap.indent(code, " " * 4)

    test_code = "\n".join(
        [
            f"# Location: {code_block_location}",
            "import pytest",
            "",
            "",
            # Show the code block location in the test report.
            f"@pytest.mark.parametrize('_', [' {code_block_location} '])",
            "def test(_):",
            content,
            "",
            "",
            'if __name__ == "__main__":',
            "    test()",
            "",
        ]
    )

    output_path = output_dir / filename
    output_path.write_text(test_code)
    return output_path


def extract_examples(
    mlflow_modules: list[str], output_dir: str | Path, repo_root: str | Path
) -> int:
    """
    Extract test examples from mlflow modules and generate test files.

    Args:
        mlflow_modules: List of module names to scan (e.g., ['mlflow', 'mlflow.tracking'])
        output_dir: Directory to write test files to
        repo_root: Root of the repository

    Returns:
        Number of tests generated
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    repo_root = Path(repo_root)

    # Clean up old test files
    for old_file in output_dir.glob("test_*.py"):
        old_file.unlink()

    total_tests = 0

    for module_name in mlflow_modules:
        print(f"Processing module: {module_name}")
        results = find_all_objects_with_docstrings(module_name)

        for obj_path, line_num, code in results:
            try:
                output_path = generate_test_file(obj_path, line_num, code, repo_root, output_dir)
                print(f"  Generated: {output_path.name}")
                total_tests += 1
            except Exception as e:
                print(f"  Error generating test for {obj_path}:{line_num}: {e}", file=sys.stderr)

    print(f"\nTotal tests generated: {total_tests}")
    return total_tests


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Extract test code blocks from Python docstrings")
    parser.add_argument(
        "--output-dir",
        default=".examples",
        help="Directory to write test files (default: .examples)",
    )
    parser.add_argument(
        "--repo-root", help="Repository root directory (default: auto-detect using git)"
    )
    parser.add_argument(
        "modules", nargs="*", default=["mlflow"], help="Module names to scan (default: mlflow)"
    )

    args = parser.parse_args()

    # Determine repository root
    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
            )
            repo_root = Path(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            repo_root = Path.cwd().parent.parent

    # Ensure mlflow is importable
    mlflow_root = repo_root
    if str(mlflow_root) not in sys.path:
        sys.path.insert(0, str(mlflow_root))

    extract_examples(args.modules, args.output_dir, repo_root)


if __name__ == "__main__":
    main()
