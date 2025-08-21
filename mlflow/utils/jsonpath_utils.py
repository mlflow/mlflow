"""
JSONPath utilities for navigating and manipulating nested JSON structures.

This module provides a simplified JSONPath-like implementation without adding
external dependencies to MLflow. Instead of using a full JSONPath library,
we implement a lightweight subset focused on trace data navigation using
dot notation with wildcard support.

The implementation supports:
- Dot notation path traversal (e.g., "info.trace_id")
- Wildcard expansion (e.g., "info.assessments.*")
- Array/list navigation with numeric indices
- Structure-preserving filtering
- Path validation with helpful error messages

This approach keeps MLflow dependencies minimal while providing the essential
functionality needed for trace field selection and data manipulation.

Note: This is NOT a complete JSONPath implementation. It's a custom solution
tailored specifically for MLflow trace data structures.
"""

from typing import Any


def split_path_respecting_backticks(path: str) -> list[str]:
    """
    Split path on dots, but keep backticked segments intact.

    Args:
        path: Path string like 'info.tags.`mlflow.traceName`'

    Returns:
        List of path segments, e.g., ['info', 'tags', 'mlflow.traceName']
    """
    parts = []
    i = 0
    current = ""

    while i < len(path):
        if i < len(path) and path[i] == "`":
            # Start of backticked segment - read until closing backtick
            i += 1  # Skip opening backtick
            while i < len(path) and path[i] != "`":
                current += path[i]
                i += 1
            if i < len(path):
                i += 1  # Skip closing backtick
        elif path[i] == ".":
            if current:
                parts.append(current)
                current = ""
            i += 1
        else:
            current += path[i]
            i += 1

    if current:
        parts.append(current)

    return parts


def jsonpath_extract_values(obj: dict[str, Any], path: str) -> list[Any]:
    """
    Extract values from nested dict using JSONPath-like dot notation with * wildcard support.

    Supports backtick escaping for field names containing dots:
        'info.tags.`mlflow.traceName`' - treats 'mlflow.traceName' as a single field

    Args:
        obj: The dictionary/object to traverse
        path: Dot-separated path like 'info.trace_id' or 'data.spans.*.name'
              Can use backticks for fields with dots: 'info.tags.`mlflow.traceName`'

    Returns:
        List of values found at the path. Returns empty list if path not found.

    Examples:
        >>> data = {"info": {"trace_id": "tr-123", "status": "OK"}}
        >>> jsonpath_extract_values(data, "info.trace_id")
        ['tr-123']
        >>> jsonpath_extract_values(data, "info.*")
        ['tr-123', 'OK']
        >>> data = {"tags": {"mlflow.traceName": "test"}}
        >>> jsonpath_extract_values(data, "tags.`mlflow.traceName`")
        ['test']
    """
    parts = split_path_respecting_backticks(path)

    def traverse(current, parts_remaining):
        if not parts_remaining:
            return [current]

        part = parts_remaining[0]
        rest = parts_remaining[1:]

        if part == "*":
            # Wildcard - expand all keys at this level
            if isinstance(current, dict):
                results = []
                for key, value in current.items():
                    results.extend(traverse(value, rest))
                return results
            elif isinstance(current, list):
                results = []
                for item in current:
                    results.extend(traverse(item, rest))
                return results
            else:
                return []
        else:
            # Regular key
            if isinstance(current, dict) and part in current:
                return traverse(current[part], rest)
            else:
                return []

    return traverse(obj, parts)


def filter_json_by_fields(data: dict[str, Any], field_paths: list[str]) -> dict[str, Any]:
    """
    Filter a JSON dict to only include fields specified by the field paths.
    Expands wildcards but preserves original JSON structure.

    Args:
        data: Original JSON dictionary
        field_paths: List of dot-notation paths like ['info.trace_id', 'info.assessments.*']

    Returns:
        Filtered dictionary with original structure preserved
    """
    result = {}

    # Collect all actual paths by expanding wildcards
    expanded_paths = set()
    for field_path in field_paths:
        if "*" in field_path:
            # Find all actual paths that match this wildcard pattern
            matching_paths = find_matching_paths(data, field_path)
            expanded_paths.update(matching_paths)
        else:
            # Direct path
            expanded_paths.add(field_path)

    # Build the result by including only the specified paths
    for path in expanded_paths:
        parts = split_path_respecting_backticks(path)
        set_nested_value(result, parts, get_nested_value_safe(data, parts))

    return result


def find_matching_paths(data: dict[str, Any], wildcard_path: str) -> list[str]:
    """Find all actual paths in data that match a wildcard pattern."""
    parts = split_path_respecting_backticks(wildcard_path)

    def find_paths(current_data, current_parts, current_path=""):
        if not current_parts:
            return [current_path.lstrip(".")]

        part = current_parts[0]
        remaining = current_parts[1:]

        if part == "*":
            paths = []
            if isinstance(current_data, dict):
                for key in current_data.keys():
                    new_path = f"{current_path}.{key}"
                    paths.extend(find_paths(current_data[key], remaining, new_path))
            elif isinstance(current_data, list):
                for i, item in enumerate(current_data):
                    new_path = f"{current_path}.{i}"
                    paths.extend(find_paths(item, remaining, new_path))
            return paths
        else:
            if isinstance(current_data, dict) and part in current_data:
                new_path = f"{current_path}.{part}"
                return find_paths(current_data[part], remaining, new_path)
            return []

    return find_paths(data, parts)


def get_nested_value_safe(data: dict[str, Any], parts: list[str]) -> Any | None:
    """Safely get nested value, returning None if path doesn't exist."""
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def set_nested_value(data: dict[str, Any], parts: list[str], value: Any) -> None:
    """Set a nested value in a dictionary, creating intermediate dicts/lists as needed."""
    if value is None:
        return

    current = data
    for i, part in enumerate(parts[:-1]):
        if part.isdigit() and isinstance(current, list):
            # Handle array index
            idx = int(part)
            while len(current) <= idx:
                current.append({})
            current = current[idx]
        else:
            # Handle object key
            if not isinstance(current, dict):
                return  # Can't set object key on non-dict
            if part not in current:
                # Look ahead to see if next part is a number (array index)
                next_part = parts[i + 1] if i + 1 < len(parts) else None
                if next_part and next_part.isdigit():
                    current[part] = []
                else:
                    current[part] = {}
            current = current[part]

    if parts:
        final_part = parts[-1]
        if final_part.isdigit() and isinstance(current, list):
            # Extend list if needed
            idx = int(final_part)
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        elif isinstance(current, dict):
            current[final_part] = value


def validate_field_paths(
    field_paths: list[str], sample_data: dict[str, Any], verbose: bool = False
) -> None:
    """Validate that field paths exist in the data structure.

    Args:
        field_paths: List of field paths to validate
        sample_data: Sample data to validate against
        verbose: If True, show all available fields instead of truncated list
    """
    invalid_paths = []

    for path in field_paths:
        # Skip validation for paths with wildcards - they'll be expanded later
        if "*" in path:
            continue

        # Test if the path exists by trying to extract values
        values = jsonpath_extract_values(sample_data, path)
        if not values:  # Empty list means path doesn't exist
            invalid_paths.append(path)

    if invalid_paths:
        available_fields = get_available_field_suggestions(sample_data)

        # Create a nice error message
        error_msg = "❌ Invalid field path(s):\n"
        for path in invalid_paths:
            error_msg += f"   • {path}\n"

        error_msg += "\n💡 Use dot notation to specify nested fields:"
        error_msg += "\n   Examples: info.trace_id, info.state, info.assessments.*"

        if available_fields:
            error_msg += "\n\n📋 Available fields in this data:\n"

            if verbose:
                # In verbose mode, show ALL available fields organized by category
                info_fields = [f for f in available_fields if f.startswith("info.")]
                data_fields = [f for f in available_fields if f.startswith("data.")]

                if info_fields:
                    error_msg += "   Info fields:\n"
                    for field in sorted(info_fields):
                        error_msg += f"     • {field}\n"

                if data_fields:
                    error_msg += "   Data fields:\n"
                    for field in sorted(data_fields):
                        error_msg += f"     • {field}\n"
            else:
                # Non-verbose mode: show truncated list
                # Group by top-level key for better readability
                info_fields = [f for f in available_fields if f.startswith("info.")]
                data_fields = [f for f in available_fields if f.startswith("data.")]

                if info_fields:
                    error_msg += f"   info.*: {', '.join(info_fields[:8])}"
                    if len(info_fields) > 8:
                        error_msg += f", ... (+{len(info_fields) - 8} more)"
                    error_msg += "\n"

                if data_fields:
                    error_msg += f"   data.*: {', '.join(data_fields[:5])}"
                    if len(data_fields) > 5:
                        error_msg += f", ... (+{len(data_fields) - 5} more)"
                    error_msg += "\n"

                error_msg += "\n💡 Tip: Use --verbose flag to see all available fields"

        raise ValueError(error_msg)


def get_available_field_suggestions(data: dict[str, Any], prefix: str = "") -> list[str]:
    """Get a list of available field paths for suggestions."""
    paths = []

    def collect_paths(obj, current_path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{current_path}.{key}" if current_path else key
                paths.append(path)
                # Only go 2 levels deep for suggestions to keep it manageable
                if current_path.count(".") < 2:
                    collect_paths(value, path)
        elif isinstance(obj, list) and obj:
            # Show array notation but don't expand all indices
            path = f"{current_path}.*" if current_path else "*"
            if path not in paths:
                paths.append(path)
            # Sample first item if it's an object
            if isinstance(obj[0], dict):
                collect_paths(obj[0], f"{current_path}.*" if current_path else "*")

    collect_paths(data, prefix)
    return sorted(set(paths))
