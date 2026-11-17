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
perl -pi -e 'BEGIN { @s = grep { length } split /\n/, $ENV{SECRETS} // "" } for my $x (@s) { s/\Q$x\E/***/g }' "$file"
