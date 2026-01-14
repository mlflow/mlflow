"""
Validate MLflow tracing by running the agent (RUNTIME VALIDATION).

CRITICAL: This script REQUIRES valid authentication and LLM access.
If this validation fails, the evaluation workflow MUST STOP until auth issues are resolved.

This is Stage 2 validation - run AFTER validate_tracing_static.py passes.

This script verifies by actually running the agent:
1. Traces are captured successfully
2. Complete trace hierarchy is present (decorator + autolog spans)
3. Session ID is tagged (if applicable)
4. Agent execution completes without errors

Usage:
    python validate_tracing_runtime.py                                    # Auto-detect everything
    python validate_tracing_runtime.py --module my_agent.agent            # Specify module
    python validate_tracing_runtime.py --entry-point process              # Specify entry point
    python validate_tracing_runtime.py --module my_agent --entry-point process  # Specify both
"""

import argparse
import importlib
import os
import sys
from pathlib import Path


def check_environment():
    """Check that required environment variables are set."""
    print("Checking environment...")
    errors = []

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        errors.append("MLFLOW_TRACKING_URI is not set")
    else:
        print(f"  ✓ MLFLOW_TRACKING_URI={tracking_uri}")

    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
    if not experiment_id:
        errors.append("MLFLOW_EXPERIMENT_ID is not set")
    else:
        print(f"  ✓ MLFLOW_EXPERIMENT_ID={experiment_id}")

    return errors


def find_agent_module():
    """Find the agent module in the project."""
    # Common patterns
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


def check_autolog(module_name: str) -> tuple[bool, str]:
    """Check if autolog is enabled."""
    print("\nChecking autolog...")

    # Look for autolog calls in __init__.py or main entry point
    search_paths = [
        "src/*/__init__.py",
        "*/__init__.py",
        "__init__.py",
        "main.py",
        "app.py",
    ]

    autolog_patterns = [
        "mlflow.langchain.autolog",
        "mlflow.langgraph.autolog",
        "mlflow.openai.autolog",
        "mlflow.autolog",
    ]

    for pattern in search_paths:
        for file_path in Path(".").glob(pattern):
            try:
                content = file_path.read_text()
                for autolog_pattern in autolog_patterns:
                    if autolog_pattern in content:
                        lib = autolog_pattern.split(".")[1]
                        print(f"  ✓ Found {autolog_pattern}() in {file_path}")
                        return True, lib
            except Exception:
                continue

    print("  ✗ No autolog call found")
    print("    Searched for: mlflow.{langchain,langgraph,openai}.autolog()")
    return False, ""


def find_decorated_functions() -> list[tuple[str, str]]:
    """Find all functions decorated with @mlflow.trace (most reliable method)."""
    print("\nSearching for @mlflow.trace decorated functions...")

    decorated_functions = []

    for py_file in Path(".").rglob("*.py"):
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
                            func_name = lines[j].split("def ")[1].split("(")[0].strip()
                            decorated_functions.append((str(py_file), func_name))
                            print(f"  ✓ {py_file}:{func_name}")
                            break
        except Exception:
            continue

    return decorated_functions


def find_entry_points_by_pattern() -> list[tuple[str, str, bool]]:
    """Find entry points by common naming patterns (fallback method)."""
    print("\nSearching for common entry point patterns...")

    # Common entry point patterns
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

    # Search in Python files
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
                    status = "✓" if has_decorator else "✗"
                    print(f"  {status} {py_file}:{func_name} (@mlflow.trace: {has_decorator})")
        except Exception:
            continue

    return found


def find_all_public_functions(module_name: str) -> list[str]:
    """Find all public functions in a module."""
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
    except Exception as e:
        print(f"  ✗ Could not introspect module: {e}")
        return []


def select_entry_point(module_name: str, specified_entry_point: str | None = None) -> str | None:
    """Select entry point through various methods."""

    # Method 1: Use specified entry point
    if specified_entry_point:
        print(f"\nUsing specified entry point: {specified_entry_point}")
        return specified_entry_point

    # Method 2: Search for @mlflow.trace decorated functions (most reliable)
    decorated = find_decorated_functions()
    if decorated:
        if len(decorated) == 1:
            file_path, func_name = decorated[0]
            print(f"\n✓ Found single decorated function: {func_name}")
            return func_name
        else:
            print(f"\n✓ Found {len(decorated)} decorated functions:")
            for i, (file_path, func_name) in enumerate(decorated, 1):
                print(f"  {i}. {func_name} ({file_path})")

            try:
                choice = int(input("\nSelect entry point (number): "))
                if 1 <= choice <= len(decorated):
                    return decorated[choice - 1][1]
            except (ValueError, KeyboardInterrupt):
                pass

    # Method 3: Search by common patterns
    pattern_matches = find_entry_points_by_pattern()
    if pattern_matches:
        # Prefer decorated functions
        decorated_matches = [m for m in pattern_matches if m[2]]
        if decorated_matches:
            if len(decorated_matches) == 1:
                return decorated_matches[0][1]
            else:
                print(f"\n✓ Found {len(decorated_matches)} decorated functions matching patterns:")
                for i, (file_path, func_name, _) in enumerate(decorated_matches, 1):
                    print(f"  {i}. {func_name}")

                try:
                    choice = int(input("\nSelect entry point (number): "))
                    if 1 <= choice <= len(decorated_matches):
                        return decorated_matches[choice - 1][1]
                except (ValueError, KeyboardInterrupt):
                    pass
        else:
            # Show all matches even if not decorated
            print(f"\n⚠ Found {len(pattern_matches)} functions but none are decorated:")
            for file_path, func_name, _ in pattern_matches[:5]:
                print(f"  - {func_name}")

    # Method 4: List all public functions in the module
    print(f"\nListing all public functions in {module_name}...")
    functions = find_all_public_functions(module_name)

    if functions:
        print(f"\n✓ Found {len(functions)} public functions:")
        for i, func_name in enumerate(functions, 1):
            print(f"  {i}. {func_name}")

        try:
            choice = int(input(f"\nSelect entry point (1-{len(functions)}): "))
            if 1 <= choice <= len(functions):
                return functions[choice - 1]
        except (ValueError, KeyboardInterrupt):
            pass

    # Method 5: Manual input
    print("\n✗ Could not auto-detect entry point")
    entry_point = input("Enter entry point function name manually: ").strip()
    return entry_point or None


def run_test_query(
    module_name: str,
    entry_point_name: str,
    test_query: str = "What is MLflow?",
    test_session_id: str = "test-session-123",
):
    """Run a test query and verify trace capture."""
    print("\nRunning test query...")
    print(f"  Module: {module_name}")
    print(f"  Entry point: {entry_point_name}")
    print(f"  Query: {test_query}")
    print(f"  Session ID: {test_session_id}")

    try:
        # Import mlflow first
        import mlflow
        from mlflow import MlflowClient

        # Try to import the agent module
        try:
            agent_module = importlib.import_module(module_name)
        except ImportError as e:
            print(f"  ✗ Could not import module '{module_name}': {e}")
            print("    Try: pip install -e . (from project root)")
            return None

        # Get the entry point function
        if not hasattr(agent_module, entry_point_name):
            print(f"  ✗ Function '{entry_point_name}' not found in {module_name}")
            return None

        entry_point = getattr(agent_module, entry_point_name)
        print(f"  ✓ Found entry point: {entry_point_name}")

        # Try to call the entry point (be flexible with signatures)
        print("\n  Executing agent...")
        try:
            # Try different call signatures
            try:
                entry_point(test_query, session_id=test_session_id)
            except TypeError:
                try:
                    entry_point(test_query)
                except TypeError:
                    # Might need LLM provider or other args
                    print(f"  ⚠ Could not call {entry_point_name} with simple args")
                    print(
                        "    You may need to run this validation manually with proper configuration"
                    )
                    return None

            print("  ✓ Agent executed successfully")

            # Get trace
            trace_id = mlflow.get_last_active_trace_id()
            if not trace_id:
                print("  ✗ No trace ID captured!")
                return None

            print(f"  ✓ Trace captured: {trace_id}")

            # Get trace details
            client = MlflowClient()
            return client.get_trace(trace_id)

        except Exception as e:
            print(f"  ✗ Error executing agent: {e}")
            import traceback

            traceback.print_exc()
            return None

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def verify_trace_structure(trace) -> tuple[bool, list[str]]:
    """Verify the trace has the expected structure."""
    print("\nVerifying trace structure...")

    issues = []

    # Check for top-level span (from @mlflow.trace decorator)
    if not trace.data.spans:
        issues.append("No spans found in trace")
        return False, issues

    top_span = trace.data.spans[0]
    print(f"  ✓ Top-level span: {top_span.name} ({top_span.span_type})")

    # Check for library spans (from autolog)
    def count_spans(spans):
        count = len(spans)
        for span in spans:
            if hasattr(span, "spans") and span.spans:
                count += count_spans(span.spans)
        return count

    total_spans = count_spans(trace.data.spans)
    print(f"  ✓ Total spans in hierarchy: {total_spans}")

    if total_spans < 2:
        issues.append("Only one span found - autolog may not be working")
    else:
        print("  ✓ Multiple spans detected - autolog appears to be working")

    # Print hierarchy
    def print_hierarchy(spans, indent=0):
        for span in spans:
            prefix = "    " + "  " * indent
            print(f"{prefix}- {span.name} ({span.span_type})")
            if hasattr(span, "spans") and span.spans:
                print_hierarchy(span.spans, indent + 1)

    print("\n  Trace hierarchy:")
    print_hierarchy(trace.data.spans)

    return len(issues) == 0, issues


def verify_session_id(trace, expected_session_id: str) -> tuple[bool, str]:
    """Verify session ID is captured in trace."""
    print("\nVerifying session ID capture...")

    if "session_id" not in trace.info.tags:
        print("  ✗ Session ID not found in trace tags")
        return False, "Session ID not captured"

    actual_session_id = trace.info.tags["session_id"]
    print(f"  ✓ Session ID found: {actual_session_id}")

    if actual_session_id == expected_session_id:
        print("  ✓ Session ID matches expected value")
        return True, ""
    else:
        print("  ✗ Session ID mismatch!")
        print(f"    Expected: {expected_session_id}")
        print(f"    Got: {actual_session_id}")
        return (
            False,
            f"Session ID mismatch: expected {expected_session_id}, got {actual_session_id}",
        )


def main():
    """Main validation workflow."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Validate MLflow tracing integration with an agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_tracing.py                                    # Auto-detect everything
  python validate_tracing.py --module my_agent.agent            # Specify module
  python validate_tracing.py --entry-point process              # Specify entry point
  python validate_tracing.py --module my_agent --entry-point process  # Both
        """,
    )
    parser.add_argument("--module", help='Agent module name (e.g., "mlflow_agent.agent")')
    parser.add_argument("--entry-point", help='Entry point function name (e.g., "run_agent")')
    args = parser.parse_args()

    print("=" * 60)
    print("MLflow Tracing Validation")
    print("=" * 60)
    print()

    # Track issues
    all_issues = []

    # Step 1: Check environment
    env_errors = check_environment()
    if env_errors:
        print("\n✗ Environment issues:")
        for error in env_errors:
            print(f"  - {error}")
        all_issues.extend(env_errors)

    # Step 2: Find agent module
    module_name = args.module
    if not module_name:
        module_name = find_agent_module()
        if not module_name:
            print("\n✗ Could not find agent module automatically")
            module_name = input("Enter agent module name (e.g., 'mlflow_agent.agent'): ").strip()
            if not module_name:
                print("✗ No module specified")
                sys.exit(1)
        else:
            print(f"\n✓ Found agent module: {module_name}")
    else:
        print(f"\n✓ Using specified module: {module_name}")

    # Step 3: Check autolog
    autolog_enabled, library = check_autolog(module_name)
    if not autolog_enabled:
        all_issues.append("Autolog not enabled")

    # Step 4: Select entry point (improved with multiple methods)
    entry_point_name = select_entry_point(module_name, args.entry_point)

    if not entry_point_name:
        print("\n✗ No entry point specified")
        all_issues.append("No entry point found or specified")
    else:
        print(f"\n✓ Using entry point: {entry_point_name}")

    # Step 5: Run test query (if we have an entry point)
    trace = None
    if entry_point_name:
        trace = run_test_query(module_name, entry_point_name)
        if not trace:
            all_issues.append("Could not capture test trace")
        else:
            # Step 6: Verify trace structure
            structure_ok, structure_issues = verify_trace_structure(trace)
            if not structure_ok:
                all_issues.extend(structure_issues)

            # Step 7: Verify session ID (optional)
            session_ok, session_issue = verify_session_id(trace, "test-session-123")
            if not session_ok:
                # Session ID is optional, so just warn
                print(f"\n⚠ Note: {session_issue}")
                print("  Session ID tracking is optional. Skip if not needed.")

    # Final report
    print("\n" + "=" * 60)
    print("Validation Report")
    print("=" * 60)

    if not all_issues:
        print("\n✓ ALL CHECKS PASSED!")
        print("\nYour agent is properly integrated with MLflow tracing.")
        print("You can proceed with evaluation.")
    else:
        print(f"\n✗ Found {len(all_issues)} issue(s):")
        for i, issue in enumerate(all_issues, 1):
            print(f"\n{i}. {issue}")

        print("\n" + "=" * 60)
        print("Next Steps")
        print("=" * 60)
        print("\n1. Fix the issues listed above")
        print("2. Refer to references/tracing-integration.md for detailed guidance")
        print("3. Run this script again to verify fixes")
        print("\nDO NOT proceed with evaluation until all issues are resolved.")

    print("=" * 60)

    sys.exit(0 if not all_issues else 1)


if __name__ == "__main__":
    main()
