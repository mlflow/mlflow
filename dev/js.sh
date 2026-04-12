#!/usr/bin/env bash
# Entrypoint for JS commands.
#
# Usage: dev/js.sh <command> [files...]
#
# Commands:
#   fmt    Format files

set -euo pipefail

cmd="${1:-}"
shift || true

if [ -z "$cmd" ]; then
  echo "Usage: dev/js.sh <command> [files...]" >&2
  exit 1
fi

cd mlflow/server/js

npx_version=$(npx --version)
if [ "$(printf '%s\n' "11.10.0" "$npx_version" | sort -V | head -n1)" != "11.10.0" ]; then
  echo "Error: npx >= 11.10.0 is required (found $npx_version)" >&2
  exit 1
fi

# Convert paths from repo root to relative paths
files=()
for f in "$@"; do
  files+=("${f#mlflow/server/js/}")
done

case "$cmd" in
  fmt)
    # Use npx to avoid slow `yarn install --immutable`
    [ ${#files[@]} -gt 0 ] && npx --min-release-age=14 "prettier@$(jq -r '.devDependencies.prettier' package.json)" --write --ignore-unknown "${files[@]}"
    ;;
  # TODO: Add eslint, i18n, type-check commands if needed
  *)
    echo "Unknown command: $cmd" >&2
    exit 1
    ;;
esac
