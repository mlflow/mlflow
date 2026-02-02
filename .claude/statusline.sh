#!/usr/bin/env bash

command -v jq >/dev/null || exit 0

input=$(cat)

cwd=$(echo "$input" | jq -r '.workspace.current_dir')
dir=$(basename "$cwd")
model=$(echo "$input" | jq -r '.model.display_name // empty')

cd "$cwd" 2>/dev/null || cd /
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)

used=$(echo "$input" | jq -r '.context_window.used_percentage // empty')

output="$dir"
[ -n "$branch" ] && output="$output ($branch)"
[ -n "$model" ] && output="$output | $model"
[ -n "$used" ] && output="$output | $(printf "%.1f" "$used")%"

echo "$output"
