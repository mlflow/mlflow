# -*- coding: utf-8 -*-
"""
Validate MLflow tracing by running the agent (RUNTIME VALIDATION).

CRITICAL: This script REQUIRES valid authentication and LLM access.
If this validation fails, the evaluation workflow MUST STOP until auth issues are resolved.

The coding agent should discover module/entry-point/autolog using Grep first,
then pass the discovered information to this script for runtime validation.

This script verifies by actually running the agent:
1. Traces are captured successfully
2. Complete trace hierarchy is present (decorator + autolog spans)
3. Session ID is tagged (if applicable)
4. Agent execution completes without errors

Usage:
    python validate_tracing_runtime.py \
        --module my_agent.agent \
        --entry-point run_agent \
        --autolog-file src/agent/__init__.py
"""

import argparse
import importlib
import sys

from utils import validate_env_vars


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
            available = [name for name in dir(agent_module) if not name.startswith("_")]
            if available:
                print(f"    Available functions: {', '.join(available[:5])}")
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
  python validate_tracing_runtime.py                                    # Auto-detect everything
  python validate_tracing_runtime.py --module my_agent.agent            # Specify module
  python validate_tracing_runtime.py --entry-point process              # Specify entry point
  python validate_tracing_runtime.py --module my_agent --entry-point process  # Both
        """,
    )
    parser.add_argument("--module", help='Agent module name (e.g., "mlflow_agent.agent")')
    parser.add_argument("--entry-point", help='Entry point function name (e.g., "run_agent")')
    parser.add_argument(
        "--autolog-file", help='File containing autolog() call (e.g., "src/agent/__init__.py")'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MLflow Tracing Validation")
    print("=" * 60)
    print()

    # Track issues
    all_issues = []

    # Step 1: Check environment
    print("Checking environment...")
    env_errors = validate_env_vars()
    if env_errors:
        print()
        print("✗ Environment issues:")
        for error in env_errors:
            print(f"  - {error}")
        all_issues.extend(env_errors)
    else:
        import os

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
        print(f"  ✓ MLFLOW_TRACKING_URI={tracking_uri}")
        print(f"  ✓ MLFLOW_EXPERIMENT_ID={experiment_id}")

    # Step 2: Get agent module (must be specified manually)
    module_name = args.module
    if not module_name:
        print("\n✗ Agent module not specified")
        print("  Use --module to specify your agent module")
        print("  Example: --module my_agent.agent")
        print("\n  To find your agent module:")
        print("    grep -r 'def.*agent' . --include='*.py'")
        sys.exit(1)
    else:
        print(f"\n✓ Using specified module: {module_name}")

    # Step 3: Check autolog (optional - for informational purposes)
    print("\nChecking autolog configuration...")
    if args.autolog_file:
        from pathlib import Path

        if Path(args.autolog_file).exists():
            print(f"  ✓ Autolog file specified: {args.autolog_file}")
        else:
            print(f"  ✗ Autolog file not found: {args.autolog_file}")
            all_issues.append(f"Autolog file not found: {args.autolog_file}")
    else:
        print("  ⚠ No autolog file specified (use --autolog-file)")
        print("  This is optional but recommended for full validation")
        print("\n  To find autolog calls:")
        print("    grep -r 'mlflow.*autolog' . --include='*.py'")

    # Step 4: Get entry point (must be specified manually)
    print("\nChecking entry point...")
    entry_point_name = args.entry_point

    if not entry_point_name:
        print("  ✗ Entry point not specified")
        print("  Use --entry-point to specify your agent's main function")
        print("  Example: --entry-point run_agent")
        print("\n  To find entry points with @mlflow.trace:")
        print("    grep -r '@mlflow.trace' . --include='*.py'")
        all_issues.append("No entry point specified")
        sys.exit(1)
    else:
        print(f"  ✓ Using specified entry point: {entry_point_name}")

    # Step 5: Run test query
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
