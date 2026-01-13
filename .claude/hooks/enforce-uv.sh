#!/usr/bin/env bash

# Hook to enforce using 'uv run' instead of 'python' or 'python3' directly.
# This ensures consistent virtual environment usage across the project.

set -euo pipefail

# Read hook input from stdin
input=$(cat)

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

# Block direct python/python3 commands
if echo "$command" | grep -qE '^(python3?|/usr/bin/python3?|/usr/local/bin/python3?)\s'; then
  echo '{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "Use `uv run` instead."
  }
}'
fi

exit 0
