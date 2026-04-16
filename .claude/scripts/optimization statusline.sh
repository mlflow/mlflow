#!/usr/bin/env bash

# Exit if jq not installed
command -v jq >/dev/null || exit 0

input=$(cat)

# Parse JSON once (major speed improvement)
read -r cwd model used five <<EOF
$(jq -r '
  .workspace.current_dir,
  (.model.display_name // ""),
  (.context_window.used_percentage // ""),
  (.rate_limits.five_hour.used_percentage // "")
' <<< "$input")
EOF

dir=${cwd##*/}

# --- Git info (single git call) ---
branch=""
dirty=""

if git -C "$cwd" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  branch=$(git -C "$cwd" rev-parse --abbrev-ref HEAD 2>/dev/null)

  # Check repo status in ONE call (huge speed gain)
  if [ -n "$(git -C "$cwd" status --porcelain 2>/dev/null)" ]; then
    dirty="*"
  fi
fi

# --- Build output ---
output="$dir"

[ -n "$branch" ] && output+=" ($branch$dirty)"
[ -n "$model" ] && output+=" | $model"
[ -n "$used" ]  && output+=" | ctx $(printf "%.0f" "$used")%"
[ -n "$five" ]  && output+=" | 5h $(printf "%.0f" "$five")%"

echo "$output"
