#!/usr/bin/env bash

command -v jq >/dev/null || exit 0

input=$(cat)

cwd=$(echo "$input" | jq -r '.workspace.current_dir')
dir=$(basename "$cwd")
model=$(echo "$input" | jq -r '.model.display_name // empty')

cd "$cwd" 2>/dev/null || cd /
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
dirty=""
if [ -n "$branch" ]; then
  # Check for staged or unstaged changes to tracked files
  git diff --quiet 2>/dev/null && git diff --cached --quiet 2>/dev/null || dirty="*"
  # Check for untracked files (if not already dirty)
  [ -z "$dirty" ] && [ -n "$(git ls-files --others --exclude-standard 2>/dev/null | head -1)" ] && dirty="*"
fi

used=$(echo "$input" | jq -r '.context_window.used_percentage // empty')
five=$(echo "$input" | jq -r '.rate_limits.five_hour.used_percentage // empty')

output="$dir"
[ -n "$branch" ] && output="$output ($branch$dirty)"
[ -n "$model" ] && output="$output | $model"
[ -n "$used" ] && output="$output | ctx $(printf "%.0f" "$used")%"
[ -n "$five" ] && output="$output | 5h $(printf "%.0f" "$five")%"

echo "$output"
