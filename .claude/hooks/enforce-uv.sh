#!/usr/bin/env bash

# Hook to enforce using 'uv' instead of 'python', 'python3', 'pip', or 'pip3' directly.
# This ensures consistent virtual environment usage across the project.

set -euo pipefail

# Skip if jq is unavailable
if ! command -v jq &>/dev/null; then
  exit 0
fi

# Skip if uv is unavailable
if ! command -v uv &>/dev/null; then
  exit 0
fi

# Read hook input from stdin
input=$(cat)

# Extract tool name and command
tool_name=$(echo "$input" | jq -r '.tool_name // empty')
command=$(echo "$input" | jq -r '.tool_input.command // empty')

# Only process Bash tool calls
if [[ "$tool_name" != "Bash" ]]; then
  exit 0
fi

# Block direct python/python3 commands, regardless of their full path
deny_reason=""

if echo "$command" | head -1 | grep -qE '^([^[:space:]]*/)?python3?[[:space:]]'; then
  deny_reason="Direct python/python3 execution detected. Use 'uv run' instead."
fi

# Block direct pip/pip3 commands
if echo "$command" | head -1 | grep -qE '^([^[:space:]]*/)?pip3?[[:space:]]'; then
  deny_reason="Direct pip/pip3 execution detected. Use 'uv pip' or 'uv run --with <package>' (for one-off usage without permanent install) instead."
fi

# Emit deny decision if a reason was set
if [[ -n "$deny_reason" ]]; then
  echo "{
    \"hookSpecificOutput\": {
      \"hookEventName\": \"PreToolUse\",
      \"permissionDecision\": \"deny\",
      \"permissionDecisionReason\": \"$deny_reason\"
    }
  }"
fi

exit 0
