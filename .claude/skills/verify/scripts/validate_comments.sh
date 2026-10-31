#!/usr/bin/env bash
set -euo pipefail

paths=$(jq -ce '.changed_files | arrays' "$2")
errors=$(jq -r --argjson paths "$paths" '
  .comments | to_entries[] | .key as $index | .value |
  if (.path as $path | $paths | index($path)) == null then
    "Comment \($index + 1): \(.path) is not in the diff"
  else empty end,
  if (.start_line // 0) >= .line then
    "Comment \($index + 1): start_line must be before line"
  else empty end
' "$1")
if [ -n "$errors" ]; then
  printf '%s\n' "$errors" >&2
  exit 1
fi
echo 'Inline comments are valid.'
