"""Utilities for discovering agent modules and entry points."""

import importlib
import re
from pathlib import Path


def find_agent_module() -> str | None:
    """Find agent module using common glob patterns.

    Searches for agent modules in standard locations:
    - src/*/agent/__init__.py
    - src/*/agent.py
    - */agent/__init__.py
    - */agent.py
    - agent/__init__.py
    - agent.py

    Returns:
        Module name (e.g., "my_agent.agent") or None if not found
    """
    candidates = [
        "src/*/agent/__init__.py",
        "src/*/agent.py",
        "*/agent/__init__.py",
        "*/agent.py",
        "agent/__init__.py",
        "agent.py",
    ]

    for pattern in candidates:
        matches = list(Path(".").glob(pattern))
        if matches:
            # Convert path to module name
            path = matches[0]
            parts = path.parts
            if parts[0] == "src":
                parts = parts[1:]
            module_parts = [p for p in parts if p != "__init__.py" and not p.endswith(".py")]
            if path.name != "__init__.py":
                module_parts.append(path.stem)

            return ".".join(module_parts)

    return None


def find_decorated_functions() -> list[tuple[str, str]]:
    """Find all functions decorated with @mlflow.trace.

    Searches recursively through Python files for @mlflow.trace decorators.
    Excludes virtual environment directories.

    Returns:
        List of (file_path, function_name) tuples
    """
    decorated_functions = []

    for py_file in Path(".").rglob("*.py"):
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
                                decorated_functions.append((str(py_file), func_name))
                                break
        except Exception:
            continue

    return decorated_functions


def find_entry_points_by_pattern() -> list[tuple[str, str, bool]]:
    """Find entry points by common naming patterns.

    Searches for functions with common entry point names:
    - run_agent, stream_agent, handle_request, process_query
    - chat, query, process, execute, handle, invoke

    Returns:
        List of (file_path, func_name, has_decorator) tuples
    """
    patterns = [
        "run_agent",
        "stream_agent",
        "handle_request",
        "process_query",
        "chat",
        "query",
        "process",
        "execute",
        "handle",
        "invoke",
    ]

    found = []

    for py_file in Path(".").rglob("*.py"):
        if "venv" in str(py_file) or ".venv" in str(py_file) or "site-packages" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            for func_name in patterns:
                if f"def {func_name}(" in content:
                    # Check for @mlflow.trace decorator
                    has_decorator = "@mlflow.trace" in content
                    found.append((str(py_file), func_name, has_decorator))
        except Exception:
            continue

    return found


def get_public_functions(module_name: str) -> list[str]:
    """Get all public functions from a module via introspection.

    Args:
        module_name: Full module name (e.g., "my_agent.agent")

    Returns:
        List of function names (excludes _private functions)
    """
    try:
        module = importlib.import_module(module_name)
        functions = []

        for name in dir(module):
            if not name.startswith("_"):  # Public functions only
                obj = getattr(module, name)
                if callable(obj):
                    # Check if it's defined in this module (not imported)
                    if hasattr(obj, "__module__") and obj.__module__ == module_name:
                        functions.append(name)

        return functions
    except Exception:
        return []


def select_entry_point(
    module_name: str, specified: str | None = None, prefer_decorated: bool = True
) -> str | None:
    """Select entry point with fallback chain.

    Priority:
    1. Use specified entry point if provided
    2. Auto-select single @mlflow.trace decorated function
    3. Auto-select first decorated function matching common patterns
    4. Auto-select first common pattern match
    5. Return None (fail with clear error)

    Non-interactive: Always auto-selects, never prompts.

    Args:
        module_name: Agent module name
        specified: Explicitly specified entry point (overrides auto-detection)
        prefer_decorated: Prefer functions with @mlflow.trace decorator

    Returns:
        Entry point function name or None if not found
    """
    # Method 1: Use specified entry point
    if specified:
        return specified

    # Method 2: Search for @mlflow.trace decorated functions
    decorated = find_decorated_functions()
    if decorated:
        if len(decorated) == 1:
            # Single decorated function - use it
            return decorated[0][1]
        else:
            # Multiple decorated functions - return first one
            return decorated[0][1]

    # Method 3: Search by common patterns (prefer decorated if any)
    pattern_matches = find_entry_points_by_pattern()
    if pattern_matches:
        # Prefer decorated functions
        if prefer_decorated:
            decorated_matches = [m for m in pattern_matches if m[2]]
            if decorated_matches:
                return decorated_matches[0][1]

        # Fall back to first match
        return pattern_matches[0][1]

    # Method 4: Try to get public functions from module
    functions = get_public_functions(module_name)
    if functions:
        # Return first public function
        return functions[0]

    # No entry point found
    return None
