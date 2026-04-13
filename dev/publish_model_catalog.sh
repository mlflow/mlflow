#!/usr/bin/env bash
# Publish model catalog JSON files as GitHub Release assets.
#
# Usage:
#   bash dev/publish_model_catalog.sh [--repo OWNER/REPO] [--tag TAG] [--catalog-dir DIR]
#
# Defaults:
#   --repo       mlflow/mlflow
#   --tag        model-catalog/latest
#   --catalog-dir mlflow/utils/model_catalog
#
# The script creates the release if it doesn't exist, then uploads
# every *.json file in the catalog directory as a release asset,
# overwriting any existing asset with the same name.

set -euo pipefail

REPO="mlflow/mlflow"
TAG="model-catalog/latest"
CATALOG_DIR="mlflow/utils/model_catalog"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --catalog-dir) CATALOG_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "Repo:        $REPO"
echo "Tag:         $TAG"
echo "Catalog dir: $CATALOG_DIR"

# Count JSON files
file_count=$(find "$CATALOG_DIR" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')
if [[ "$file_count" -eq 0 ]]; then
  echo "Error: no JSON files found in $CATALOG_DIR" >&2
  exit 1
fi
echo "Files:       $file_count"

# Create release if it doesn't exist
if ! gh release view "$TAG" --repo "$REPO" &>/dev/null; then
  echo "Creating release $TAG ..."
  gh release create "$TAG" \
    --repo "$REPO" \
    --title "Model Catalog" \
    --notes "Per-provider model catalog files. Updated weekly by CI." \
    --latest=false
fi

# Upload all JSON files as release assets.
# --clobber overwrites existing assets (delete + re-upload per file).
# There is a brief window where each asset is missing during the overwrite,
# but clients fall back to the bundled catalog so this is safe.
echo "Uploading $file_count files to release $TAG ..."
gh release upload "$TAG" \
  --repo "$REPO" \
  --clobber \
  "$CATALOG_DIR"/*.json

echo "Done. Assets available at:"
echo "  https://github.com/$REPO/releases/tag/$(echo "$TAG" | sed 's|/|%2F|g')"
