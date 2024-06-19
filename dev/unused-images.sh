#!/usr/bin/env bash
set -e

temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT

# Image files
git ls-files docs/source | grep -E "\.(png|gif|jpg|jpeg|svg)$" > "$temp_dir/image_files.txt"
# Files where images are referenced
git ls-files | grep -E "\.(py|html|rst|md|Rmd|css|js)$" > "$temp_dir/tracked_files.txt"

unused_images=()
exit_code=0
while IFS= read -r file; do
  filename=$(basename "$file")
  if ! xargs grep -q "$filename" < "$temp_dir/tracked_files.txt"; then
    commit_info=$(git log --diff-filter=A --follow --format="https://github.com/mlflow/mlflow/commit/%H %ad" -1 --date=short -- "$file")
    echo "$file ($commit_info)"
    unused_images+=("$file")
    exit_code=1
  fi
done < "$temp_dir/image_files.txt"

if [ $exit_code -eq 1 ]; then
  echo "----- Command to remove unused images -----"
  echo "git rm ${unused_images[@]}"
fi

exit $exit_code
