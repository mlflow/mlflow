#!/usr/bin/env python3
"""
Stop hook for Claude Code with MLflow tracing.

This hook is called when a Claude conversation ends (user exits or single query completes).
It reads the full conversation transcript and creates a comprehensive MLflow trace
with nested spans for LLM calls and tool usage.

Creates MLflow trace structure:
- Root span (CHAIN) - entire conversation
  - LLM spans - each Claude API call
  - Tool spans - each tool execution with inputs/outputs

Can also be run directly for testing:
  python stop_hook.py /path/to/transcript.jsonl
"""

import argparse
import os
import sys

# Add mlflow to path for importing claude_code_tracing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mlflow.claudecode.claude_code_tracing import get_logger, process_transcript, stop_hook_handler


def main():
    """Main entry point handling both hook mode and CLI mode."""
    # Check if we have command line arguments
    if len(sys.argv) > 1:
        # CLI mode - parse arguments
        parser = argparse.ArgumentParser(
            description="Process Claude Code transcript and create MLflow trace"
        )
        parser.add_argument("transcript_path", help="Path to the transcript.jsonl file")

        args = parser.parse_args()

        # Process the transcript
        get_logger().info("Processing transcript from CLI: %s", args.transcript_path)
        trace = process_transcript(args.transcript_path, None)

        if trace:
            get_logger().info("Successfully processed transcript and created MLflow trace")
            get_logger().info("Trace ID: %s", trace.info.trace_id)
            sys.exit(0)
        get_logger().error("Failed to process transcript")
        sys.exit(1)

    # Hook mode - read from stdin
    stop_hook_handler()


if __name__ == "__main__":
    main()
