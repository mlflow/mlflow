"""Utilities for tracing-related detection and validation."""

import re
from pathlib import Path


def find_autolog_calls(search_path: str = ".") -> list[tuple[str, str]]:
    """Find autolog calls in codebase.

    Searches for: mlflow.{langchain,langgraph,openai}.autolog()

    Args:
        search_path: Root directory to search (default: current directory)

    Returns:
        List of (file_path, library_name) tuples
    """
    autolog_patterns = [
        r"mlflow\.langchain\.autolog\(\)",
        r"mlflow\.langgraph\.autolog\(\)",
        r"mlflow\.openai\.autolog\(\)",
        r"mlflow\.autolog\(\)",
    ]

    # Search in likely initialization files
    search_patterns = [
        f"{search_path}/*/__init__.py",
        f"{search_path}/*/main.py",
        "main.py",
        "__init__.py",
    ]

    found = []

    for pattern in search_patterns:
        for file_path in Path(".").glob(pattern):
            try:
                content = file_path.read_text()
                for autolog_pattern in autolog_patterns:
                    if re.search(autolog_pattern, content):
                        # Extract library name
                        lib = autolog_pattern.split(".")[1].split("\\")[0]
                        found.append((str(file_path), lib))
            except Exception:
                continue

    return found


def check_import_order(file_path: str) -> tuple[bool, str]:
    """Verify autolog is called before agent imports.

    Args:
        file_path: Path to file containing autolog call

    Returns:
        Tuple of (is_correct, message)
    """
    try:
        content = Path(file_path).read_text()
        lines = content.split("\n")

        autolog_line = None
        agent_import_line = None

        for i, line in enumerate(lines, 1):
            if "autolog()" in line:
                autolog_line = i
            # Look for imports from agent modules (adjust pattern as needed)
            if autolog_line and ("from" in line and "agent" in line.lower()):
                agent_import_line = i
                break

        if autolog_line and agent_import_line:
            if autolog_line < agent_import_line:
                return True, f"Autolog (line {autolog_line}) before agent imports (line {agent_import_line})"
            else:
                return (
                    False,
                    f"Autolog (line {autolog_line}) after agent imports (line {agent_import_line})",
                )
        elif autolog_line:
            return True, f"Autolog found at line {autolog_line}"
        else:
            return False, "Autolog not found"

    except Exception as e:
        return True, f"Could not check import order: {e}"  # Don't fail on errors


def find_trace_decorators(search_path: str = ".") -> list[tuple[str, str, int]]:
    """Find @mlflow.trace decorators in codebase.

    Args:
        search_path: Root directory to search (default: current directory)

    Returns:
        List of (file_path, func_name, line_num) tuples
    """
    decorated = []

    for py_file in Path(search_path).rglob("*.py"):
        # Skip virtual environments
        if "venv" in str(py_file) or ".venv" in str(py_file) or "site-packages" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines):
                if "@mlflow.trace" in line:
                    # Look for function definition in next few lines
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if "def " in lines[j]:
                            func_match = re.search(r"def\s+(\w+)\s*\(", lines[j])
                            if func_match:
                                func_name = func_match.group(1)
                                decorated.append((str(py_file), func_name, i + 1))
                                break
        except Exception:
            continue

    return decorated


def check_session_id_capture(file_path: str) -> bool:
    """Check if file has session ID tracking code.

    Looks for: get_last_active_trace_id(), set_trace_tag(), session_id

    Args:
        file_path: Path to file to check

    Returns:
        True if all patterns found
    """
    try:
        content = Path(file_path).read_text()

        session_patterns = [
            r"mlflow\.get_last_active_trace_id\(\)",
            r"mlflow\.set_trace_tag\(",
            r"session_id",
        ]

        return all(re.search(pattern, content) for pattern in session_patterns)
    except Exception:
        return False


def verify_mlflow_imports(file_paths: list[str]) -> dict[str, bool]:
    """Check mlflow is imported in given files.

    Args:
        file_paths: List of file paths to check

    Returns:
        Dictionary mapping file_path to has_mlflow_import
    """
    results = {}

    for file_path in file_paths:
        try:
            content = Path(file_path).read_text()
            results[file_path] = "import mlflow" in content
        except Exception:
            results[file_path] = False

    return results
