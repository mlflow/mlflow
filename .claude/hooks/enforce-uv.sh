#!/usr/bin/env bash

# Hook to enforce using 'uv run' instead of 'python' or 'python3' directly.
# This ensures consistent virtual environment usage across the project.

set -euo pipefail

# Read hook input from stdin
input=$(cat)

# Skip if jq is unavailable
if ! command -v jq &>/dev/null; then
  exit 0
fi

# Extract tool name and command
tool_name=$(echo "$input" | jq -r '.tool_name // empty')
command=$(echo "$input" | jq -r '.tool_input.command // empty')

# Only process Bash tool calls
if [[ "$tool_name" != "Bash" ]]; then
  exit 0
fi

# Skip if uv is unavailable
if ! command -v uv &>/dev/null; then
  exit 0
fi

# Block direct python/python3 commands, regardless of their full path
if echo "$command" | grep -qE '^([^[:space:]]*/)?python3?[[:space:]]'; then
  echo '{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "Direct python/python3 execution detected. Use `uv run` instead."
  }
}'
fi

exit 0
