#!/usr/bin/env bash
# Prefix each hunk line of a diff with its old (`L`) and new (`R`) line numbers.
# Usage: git diff | .claude/scripts/number-diff.sh
#
# Example output:
#   @@ -3,3 +3,4 @@ def main():
#       L3     R3 |       a = 1
#       L4        | -     b = 2
#              R4 | +     b = 3
#              R5 | +     c = 4
#       L5     R6 |       return a
set -euo pipefail

awk '
/^diff --git / { header = 1 }
/^@@/ {
  header = 0
  split($2, old_range, /[-,]/); split($3, new_range, /[+,]/)
  old = old_range[2]; new = new_range[2]
  print; next
}
header { print; next }
# "\ No newline at end of file" is not a line of either file.
/^\\/ { printf "%6s %6s | %s\n", "", "", $0; next }
# Separate the +/- marker from the code with a space.
{ line = substr($0, 1, 1) " " substr($0, 2) }
/^-/ { printf "%6s %6s | %s\n", "L" old++, "", line; next }
/^\+/ { printf "%6s %6s | %s\n", "", "R" new++, line; next }
{ printf "%6s %6s | %s\n", "L" old++, "R" new++, line }
'
