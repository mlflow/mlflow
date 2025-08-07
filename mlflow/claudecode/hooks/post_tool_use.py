#!/usr/bin/env python3
"""
PostToolUse hook for Claude Code with MLflow tracing.

This hook is called after each tool execution during a Claude conversation.
It logs tool usage when MLflow tracing is enabled.
"""

import os
import sys

# Add mlflow to path for importing claude_code_tracing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mlflow.claudecode.claude_code_tracing import post_tool_use_handler

if __name__ == "__main__":
    post_tool_use_handler()
