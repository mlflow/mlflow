#!/usr/bin/env bash
# Usage: SECRETS=$'<secret>\n<secret>...' .claude/scripts/redact.sh <file>
# Replaces each non-empty line of $SECRETS in <file> with ***. Artifacts aren't masked like logs.
set -euo pipefail

file=$1
if [ -L "$file" ]; then
  echo "$file is a symlink" >&2
  exit 1
fi
[ -f "$file" ] || exit 0

text=$(<"$file")
while IFS= read -r secret; do
  # The quoted pattern matches literally, so regex/glob characters in a secret are safe.
  if [ -n "$secret" ]; then
    text=${text//"$secret"/***}
  fi
done <<<"${SECRETS:-}"

tmp=$(mktemp "$file.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT
printf '%s\n' "$text" >"$tmp"
mv -f -- "$tmp" "$file"
