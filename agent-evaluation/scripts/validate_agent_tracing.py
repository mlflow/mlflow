#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate MLflow tracing for your agent.

This is a template script. Fill in the TODO sections before running:
1. Update the import statement with your agent's module and function
2. Configure any dependencies (LLM providers, config, etc.)
3. Adjust the function call to match your agent's signature
4. Verify environment variables are set correctly
"""

import os
import sys
import mlflow
from mlflow import MlflowClient

# TODO: Update these imports with your agent's module and entry point
# Example: from my_agent.agent import run_agent
from YOUR_MODULE import YOUR_ENTRY_POINT  # <-- UPDATE THIS LINE

# Configuration
TEST_QUERY = "What is MLflow?"
TEST_SESSION_ID = "test-session-123"

# Verify environment variables
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

if not tracking_uri or not experiment_id:
    print("✗ Missing required environment variables:")
    print("  MLFLOW_TRACKING_URI:", tracking_uri or "(not set)")
    print("  MLFLOW_EXPERIMENT_ID:", experiment_id or "(not set)")
    print("\nRun scripts/setup_mlflow.py first")
    sys.exit(1)

print("=" * 60)
print("MLflow Tracing Validation")
print("=" * 60)
print()
print(f"Tracking URI: {tracking_uri}")
print(f"Experiment ID: {experiment_id}")
print()

# TODO: Configure your agent's dependencies here
# IMPORTANT: Add any required setup before calling your agent
# Examples:
# from your_agent.llm import LLMConfig, LLMProvider
# llm_config = LLMConfig(model="gpt-4", temperature=0.0)
# llm_provider = LLMProvider(config=llm_config)
#
# from your_agent.config import AgentConfig
# agent_config = AgentConfig.from_env()

print("Running test query...")
print(f"  Query: {TEST_QUERY}")
print(f"  Session ID: {TEST_SESSION_ID}")
print()

try:
    # TODO: Update this function call to match your agent's signature
    # Examples:
    #   response = YOUR_ENTRY_POINT(TEST_QUERY, llm_provider)
    #   response = YOUR_ENTRY_POINT(TEST_QUERY, session_id=TEST_SESSION_ID)
    #   response = YOUR_ENTRY_POINT(TEST_QUERY, config=agent_config)

    response = YOUR_ENTRY_POINT(TEST_QUERY)  # <-- UPDATE THIS LINE

    print("✓ Agent executed successfully")
    print()

    # Capture trace
    trace_id = mlflow.get_last_active_trace_id()
    if not trace_id:
        print("✗ FAILED: No trace ID captured!")
        print("  Check that mlflow.autolog() is called before agent execution")
        sys.exit(1)

    print(f"✓ Trace captured: {trace_id}")

    # Get trace details
    client = MlflowClient()
    trace = client.get_trace(trace_id)

    # Verify trace structure
    print()
    print("Verifying trace structure...")

    if not trace.data.spans:
        print("✗ FAILED: No spans found in trace")
        sys.exit(1)

    print(f"✓ Top-level span: {trace.data.spans[0].name} ({trace.data.spans[0].span_type})")

    # Count total spans (including nested)
    def count_spans(spans):
        count = len(spans)
        for span in spans:
            if hasattr(span, 'spans') and span.spans:
                count += count_spans(span.spans)
        return count

    total_spans = count_spans(trace.data.spans)
    print(f"✓ Total spans: {total_spans}")

    if total_spans < 2:
        print("⚠  WARNING: Only 1 span found - autolog may not be working")
        print("  Expected: @mlflow.trace decorator span + autolog library spans")
    else:
        print("✓ Multiple spans detected - autolog appears to be working")

    # Print trace hierarchy
    def print_hierarchy(spans, indent=0):
        for span in spans:
            prefix = "    " + "  " * indent
            print(f"{prefix}- {span.name} ({span.span_type})")
            if hasattr(span, 'spans') and span.spans:
                print_hierarchy(span.spans, indent + 1)

    print()
    print("  Trace hierarchy:")
    print_hierarchy(trace.data.spans)

    # Check session ID (optional)
    if "session_id" in trace.info.tags:
        actual_session_id = trace.info.tags["session_id"]
        print()
        if actual_session_id == TEST_SESSION_ID:
            print(f"✓ Session ID tagged: {actual_session_id}")
        else:
            print(f"⚠  Session ID mismatch: expected {TEST_SESSION_ID}, got {actual_session_id}")
    else:
        print()
        print("  ℹ  Note: No session_id tag found (optional for single-turn agents)")

    print()
    print("=" * 60)
    print("✓ VALIDATION PASSED")
    print("=" * 60)
    print()
    print("Your agent is properly integrated with MLflow tracing!")
    print()

except Exception as e:
    print(f"✗ FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
