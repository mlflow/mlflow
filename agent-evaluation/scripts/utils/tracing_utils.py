"""Utilities for tracing-related validation.

The coding agent should use Grep tool for discovery:
- Find autolog calls: grep -r "mlflow.*autolog" . --include="*.py"
- Find trace decorators: grep -r "@mlflow.trace" . --include="*.py"
- Find MLflow imports: grep -r "import mlflow" . --include="*.py"

This module provides validation helpers used by validation scripts.
"""

import re
from pathlib import Path


def check_import_order(file_path: str, import_pattern: str = None) -> tuple[bool, str]:
    """Verify autolog is called before library/module imports.

    Args:
        file_path: Path to file containing autolog call
        import_pattern: Optional regex pattern to match imports (e.g., r"from .* import")
                       If None, checks for any "from ... import" after autolog

    Returns:
        Tuple of (is_correct, message)
    """
    try:
        content = Path(file_path).read_text()
        lines = content.split("\n")

        autolog_line = None
        first_import_line = None

        for i, line in enumerate(lines, 1):
            if "autolog()" in line:
                autolog_line = i
            # After finding autolog, look for any imports (customizable via pattern)
            if autolog_line and "from" in line and "import" in line:
                if import_pattern:
                    if re.search(import_pattern, line):
                        first_import_line = i
                        break
                else:
                    first_import_line = i
                    break

        if autolog_line and first_import_line:
            if autolog_line < first_import_line:
                return True, f"Autolog (line {autolog_line}) before imports (line {first_import_line})"
            else:
                return (
                    False,
                    f"Autolog (line {autolog_line}) after imports (line {first_import_line})",
                )
        elif autolog_line:
            return True, f"Autolog found at line {autolog_line}"
        else:
            return False, "Autolog not found"

    except Exception as e:
        return True, f"Could not check import order: {e}"  # Don't fail on errors




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
