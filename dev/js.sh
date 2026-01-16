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

# Convert paths from repo root to relative paths
files=()
for f in "$@"; do
  files+=("${f#mlflow/server/js/}")
done

case "$cmd" in
  fmt)
    # Use npx to avoid slow `yarn install --immutable`
    [ ${#files[@]} -gt 0 ] && npx "prettier@$(jq -r '.devDependencies.prettier' package.json)" --write "${files[@]}"
    ;;
  # TODO: Add eslint, i18n, type-check commands if needed
  *)
    echo "Unknown command: $cmd" >&2
    exit 1
    ;;
esac
