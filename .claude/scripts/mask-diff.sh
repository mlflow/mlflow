#!/usr/bin/env bash
# Replace the hunks of auto-generated files in a diff with a one-line message.
# Usage: git diff | .claude/scripts/mask-diff.sh
#
# Example output:
#   diff --git a/uv.lock b/uv.lock
#   index 1111111..2222222 100644
#   --- a/uv.lock
#   +++ b/uv.lock
#   Auto-generated file, diff omitted
set -euo pipefail

awk '
function generated(path) {
  if (path ~ /^mlflow\/protos\/.*\.pyi?$/) return 1
  if (path ~ /(^|\/)(uv\.lock|yarn\.lock|package-lock\.json)$/) return 1
  if (path ~ /^mlflow\/java\/client\/src\/main\/java\/.*\/(proto|scalapb_interface)\/.*\.java$/) return 1
  return 0
}
/^diff --git / {
  path = $0; sub(/.* b\//, "", path)
  mask = generated(path)
  header = 1
}
/^@@/ { if (mask && header) print "Auto-generated file, diff omitted"; header = 0 }
header || !mask
'
