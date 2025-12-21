"""
Standalone script to extract code blocks marked with :test: from Python docstrings.
Uses AST to parse Python files and extract docstrings with test code blocks.
"""

import ast
import re
import subprocess
import textwrap
from pathlib import Path

_CODE_BLOCK_HEADER_REGEX = re.compile(r"^\.\.\s+code-block::\s*py(thon)?")
_CODE_BLOCK_OPTION_REGEX = re.compile(r"^:\w+:")


def _get_indent(s: str) -> int:
    return len(s) - len(s.lstrip())


def _get_header_indent(s: str) -> int | None:
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


def extract_code_blocks_from_file(filepath: Path, repo_root: Path) -> list[tuple[str, int, str]]:
    """
    Extract all code blocks marked with :test: from a Python file.

    Args:
        filepath: Path to the Python file
        repo_root: Root of the repository

    Returns:
        List of tuples: (location_string, line_number, code_content)
    """
    source = filepath.read_text()
    tree = ast.parse(source)

    results = []
    rel_path = filepath.relative_to(repo_root)

    for node in ast.walk(tree):
        # Check functions and classes for docstrings
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node)
            if not docstring:
                continue

            blocks = extract_code_blocks_from_docstring(docstring)
            for lineno_in_docstring, code in blocks:
                # Calculate the actual line number in the file
                # The docstring starts at node.lineno, and lineno_in_docstring is relative to that
                actual_line = node.lineno + lineno_in_docstring
                location = f"{rel_path}:{actual_line}"
                results.append((location, lineno_in_docstring, code))

    return results


def find_python_files(directory: Path, repo_root: Path) -> list[Path]:
    """Find all Python files tracked by git in a directory."""
    # Get relative path from repo root
    rel_dir = directory.relative_to(repo_root)

    # Run git ls-files from repo root with the directory as a pattern
    output = subprocess.check_output(
        ["git", "ls-files", f"{rel_dir}/*.py"],
        cwd=repo_root,
        text=True,
    )
    files = [repo_root / line for line in output.strip().split("\n") if line]
    return sorted(files)


def generate_test_file(location: str, line_num: int, code: str, output_dir: Path) -> Path:
    """Generate a pytest test file for a code block."""
    # Create a unique filename based on location
    safe_name = re.sub(r"[/\\:.]", "_", location)
    filename = f"test_{safe_name}_{line_num}.py"
    content = textwrap.indent(code, " " * 4)

    test_code = "\n".join(
        [
            f"# Location: {location}",
            "import pytest",
            "",
            "",
            # Show the code block location in the test report.
            f"@pytest.mark.parametrize('_', [' {location} '])",
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


def extract_examples(mlflow_dir: Path, output_dir: Path, repo_root: Path) -> None:
    """
    Extract test examples from Python files and generate test files.

    Args:
        mlflow_dir: Directory containing Python files to scan
        output_dir: Directory to write test files to
        repo_root: Root of the repository
    """
    output_dir.mkdir(exist_ok=True)

    # Clean up old test files
    for old_file in output_dir.glob("test_*.py"):
        old_file.unlink()

    print(f"Scanning Python files in: {mlflow_dir}")
    python_files = find_python_files(mlflow_dir, repo_root)
    print(f"Found {len(python_files)} Python files")

    for filepath in python_files:
        results = extract_code_blocks_from_file(filepath, repo_root)

        for location, line_num, code in results:
            output_path = generate_test_file(location, line_num, code, output_dir)
            print(f"  Generated: {output_path.name}")


def main() -> None:
    output = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True)
    repo_root = Path(output.strip())
    scan_dir = repo_root / "mlflow"
    output_dir = Path(".examples")
    extract_examples(scan_dir, output_dir, repo_root)


if __name__ == "__main__":
    main()
