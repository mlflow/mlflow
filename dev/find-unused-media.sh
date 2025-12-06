#!/usr/bin/env bash
set -euo pipefail

# Find unused images and videos under docs/ (basename matching).
# Exits with 1 if unused images are found.
#
# Requires: git, ripgrep (rg)

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -x "$repo_root/bin/rg" ]]; then
  rg="$repo_root/bin/rg"
elif command -v rg &> /dev/null; then
  rg="rg"
else
  echo "Error: ripgrep (rg) is not installed. Run 'python bin/install.py' first." >&2
  exit 1
fi

tmp_images="$(mktemp)"
tmp_image_map="$(mktemp)"
tmp_used="$(mktemp)"
trap 'rm -f "$tmp_images" "$tmp_image_map" "$tmp_used"' EXIT

# 1) List tracked files under docs/, then filter image extensions via grep
git ls-files docs/ \
  | grep -Ei '\.(png|jpe?g|gif|webp|ico|avif|mp4)$' \
  > "$tmp_images"

if [[ ! -s "$tmp_images" ]]; then
  echo "No tracked images under docs/."
  exit 0
fi

# basename<TAB>path
awk -F/ '{print $NF "\t" $0}' "$tmp_images" > "$tmp_image_map"

# 2) Extract used basenames from entire repo
"$rg" -o --no-heading --no-line-number \
  '[^"'\''[:space:]()]+\.(png|jpe?g|gif|webp|ico|avif|mp4)\b' \
  . \
  | sed 's#.*/##' \
  | sort -u \
  > "$tmp_used" || true

# 3) Compute unused (join by basename)
sort -k1,1 "$tmp_image_map" -o "$tmp_image_map"
sort "$tmp_used" -o "$tmp_used"

unused_paths="$(join -t $'\t' -v 1 "$tmp_image_map" "$tmp_used" | cut -f2)"

if [[ -z "$unused_paths" ]]; then
  echo "No unused media files" >&2
  exit 0
fi

echo "$unused_paths"
echo >&2
echo 'Unused media files found. Run `./dev/find-unused-media.sh | xargs rm` to remove them.' >&2
exit 1
