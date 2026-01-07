"""
Validate MLflow tracing setup through static code analysis.

This script checks that tracing is properly integrated without requiring
authentication or actually running the agent.

Checks:
- Autolog call present in initialization code
- Autolog called before agent imports
- @mlflow.trace decorators on entry points
- Session ID capture code present (if applicable)

Usage:
    python scripts/validate_tracing_static.py
"""

import re
import sys
from pathlib import Path


def find_autolog_calls(src_path="src"):
    """Find autolog calls in the codebase."""
    print("Checking for autolog...")

    autolog_patterns = [
        r"mlflow\.langchain\.autolog\(\)",
        r"mlflow\.langgraph\.autolog\(\)",
        r"mlflow\.openai\.autolog\(\)",
        r"mlflow\.autolog\(\)",
    ]

    # Search in likely initialization files
    search_patterns = [
        f"{src_path}/*/__init__.py",
        f"{src_path}/*/main.py",
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
                        lib = autolog_pattern.split(".")[1]
                        found.append((str(file_path), lib))
                        print(f"  ✓ Found mlflow.{lib}.autolog() in {file_path}")
            except Exception:
                continue

    if not found:
        print("  ✗ No autolog call found")
        print("    Searched for: mlflow.{langchain,langgraph,openai}.autolog()")
        return []

    return found


def check_import_order(file_path):
    """Check if autolog is called before agent imports."""
    print(f"\nChecking import order in {file_path}...")

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
                print(
                    f"  ✓ Autolog (line {autolog_line}) before agent imports (line {agent_import_line})"
                )
                return True
            else:
                print(
                    f"  ✗ Autolog (line {autolog_line}) after agent imports (line {agent_import_line})"
                )
                print("    Move autolog call before agent imports")
                return False
        elif autolog_line:
            print(f"  ✓ Autolog found at line {autolog_line}")
            return True
        else:
            return True

    except Exception as e:
        print(f"  ⚠ Could not check import order: {e}")
        return True  # Don't fail on this


def find_decorated_functions(src_path="src"):
    """Find functions decorated with @mlflow.trace."""
    print("\nChecking for @mlflow.trace decorators...")

    decorated = []

    for py_file in Path(".").rglob(f"{src_path}/**/*.py"):
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
                                print(f"  ✓ {py_file}:{func_name} (line {i + 1})")
                                break
        except Exception:
            continue

    if not decorated:
        print("  ✗ No @mlflow.trace decorators found")
        print("    Add @mlflow.trace to entry point functions")
        return []

    return decorated


def check_mlflow_imports(decorated_functions):
    """Check that mlflow is imported in files with decorators."""
    print("\nChecking mlflow imports...")

    all_good = True

    checked_files = set()
    for file_path, func_name, _ in decorated_functions:
        if file_path in checked_files:
            continue

        checked_files.add(file_path)

        try:
            content = Path(file_path).read_text()
            if "import mlflow" in content:
                print(f"  ✓ mlflow imported in {file_path}")
            else:
                print(f"  ✗ mlflow NOT imported in {file_path}")
                print("    Add: import mlflow")
                all_good = False
        except Exception as e:
            print(f"  ⚠ Could not check {file_path}: {e}")

    return all_good


def check_session_id_capture(decorated_functions):
    """Check if session_id is captured in traces."""
    print("\nChecking session ID capture...")

    session_patterns = [
        r"mlflow\.get_last_active_trace_id\(\)",
        r"mlflow\.set_trace_tag\(",
        r"session_id",
    ]

    found_session_tracking = []

    for file_path, func_name, line_num in decorated_functions:
        try:
            content = Path(file_path).read_text()

            # Look for session tracking patterns
            has_all_patterns = all(re.search(pattern, content) for pattern in session_patterns)

            if has_all_patterns:
                found_session_tracking.append((file_path, func_name))
                print(f"  ✓ Session ID capture found in {func_name} ({file_path})")

        except Exception:
            continue

    if not found_session_tracking:
        print("  ⚠ No session ID capture detected")
        print("    This is optional but recommended for conversation tracking")
        print("    See references/tracing-integration.md for implementation")

    return found_session_tracking


def main():
    """Main validation workflow."""
    print("=" * 60)
    print("Static Tracing Validation")
    print("=" * 60)
    print()

    issues = []

    # Check 1: Find autolog calls
    autolog_files = find_autolog_calls()
    if not autolog_files:
        issues.append("No autolog call found")
    else:
        # Check import order for each file
        for file_path, lib in autolog_files:
            if not check_import_order(file_path):
                issues.append(f"Autolog not called before agent imports in {file_path}")

    # Check 2: Find decorated functions
    decorated = find_decorated_functions()
    if not decorated:
        issues.append("No @mlflow.trace decorators found")

    # Check 3: Verify mlflow imports
    if decorated and not check_mlflow_imports(decorated):
        issues.append("mlflow not imported in files with decorators")

    # Check 4: Check session ID (optional - warning only)
    check_session_id_capture(decorated)

    # Summary
    print()
    print("=" * 60)
    print("Validation Report")
    print("=" * 60)
    print()

    if not issues:
        print("✓ STATIC VALIDATION PASSED")
        print()
        print("Code structure looks good!")
        print()
        print("Next steps:")
        print("  1. Verify authentication: python scripts/validate_auth.py")
        print("  2. Run runtime validation: python scripts/validate_tracing_runtime.py")
        print()
    else:
        print(f"✗ Found {len(issues)} issue(s):")
        print()
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        print("=" * 60)
        print("Next Steps")
        print("=" * 60)
        print()
        print("1. Fix the issues listed above")
        print("2. Refer to references/tracing-integration.md for detailed guidance")
        print("3. Run this script again to verify fixes")
        print()
        print("DO NOT proceed with evaluation until all issues are resolved.")
        print()
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
