#!/usr/bin/env bash
set -euo pipefail

jq '
  .tool_input
  | select(.run_in_background != true and (.timeout // 120000) > 120000)
  | {
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: "Foreground Bash timeout must be at most 120000 ms. Retry with a shorter timeout, or use run_in_background: true for longer commands."
      }
    }
'
