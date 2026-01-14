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

import sys
from pathlib import Path

from utils import (
    check_import_order,
    check_session_id_capture,
    find_autolog_calls,
    find_trace_decorators,
    verify_mlflow_imports,
)


def main():
    """Main validation workflow."""
    print("=" * 60)
    print("Static Tracing Validation")
    print("=" * 60)
    print()

    issues = []

    # Check 1: Find autolog calls
    print("Checking for autolog...")
    autolog_files = find_autolog_calls()
    if not autolog_files:
        print("  ✗ No autolog call found")
        print("    Searched for: mlflow.{langchain,langgraph,openai}.autolog()")
        issues.append("No autolog call found")
    else:
        for file_path, lib in autolog_files:
            print(f"  ✓ Found mlflow.{lib}.autolog() in {file_path}")
            # Check import order for each file
            is_correct, message = check_import_order(file_path)
            if is_correct:
                print(f"\n  ✓ {message}")
            else:
                print(f"\n  ✗ {message}")
                print("    Move autolog call before agent imports")
                issues.append(f"Autolog not called before agent imports in {file_path}")

    # Check 2: Find decorated functions
    print("\nChecking for @mlflow.trace decorators...")
    decorated = find_trace_decorators()
    if not decorated:
        print("  ✗ No @mlflow.trace decorators found")
        print("    Add @mlflow.trace to entry point functions")
        issues.append("No @mlflow.trace decorators found")
    else:
        for file_path, func_name, line_num in decorated:
            print(f"  ✓ {file_path}:{func_name} (line {line_num})")

    # Check 3: Verify mlflow imports
    if decorated:
        print("\nChecking mlflow imports...")
        file_paths = list(set(fp for fp, _, _ in decorated))
        import_results = verify_mlflow_imports(file_paths)

        all_good = True
        for file_path, has_import in import_results.items():
            if has_import:
                print(f"  ✓ mlflow imported in {file_path}")
            else:
                print(f"  ✗ mlflow NOT imported in {file_path}")
                print("    Add: import mlflow")
                all_good = False

        if not all_good:
            issues.append("mlflow not imported in files with decorators")

    # Check 4: Check session ID (optional - warning only)
    if decorated:
        print("\nChecking session ID capture...")
        found_session_tracking = False

        for file_path, func_name, _ in decorated:
            if check_session_id_capture(file_path):
                print(f"  ✓ Session ID capture found in {func_name} ({file_path})")
                found_session_tracking = True

        if not found_session_tracking:
            print("  ⚠ No session ID capture detected")
            print("    This is optional but recommended for conversation tracking")
            print("    See references/tracing-integration.md for implementation")

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
