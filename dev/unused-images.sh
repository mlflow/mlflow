#!/usr/bin/env bash
set -e

temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT

# Image files
git ls-files docs/source | grep -E "\.(png|gif|jpg|jpeg)$" > "$temp_dir/image_files.txt"
# Files where images are referenced
git ls-files '*.rst' '*.py' '*.html' '*.md' > "$temp_dir/tracked_files.txt"

EXIT_CODE=0
while IFS= read -r file; do
  filename=$(basename "$file")
  if ! xargs grep -q "$filename" < "$temp_dir/tracked_files.txt"; then
    commit_info=$(git log --diff-filter=A --follow --format="https://github.com/mlflow/mlflow/commit/%H %ad" -1 --date=short -- "$file")
    echo "$file ($commit_info)"
    EXIT_CODE=1
  fi
done < "$temp_dir/image_files.txt"

exit $EXIT_CODE
