#!/usr/bin/env bash
#
# Validates TypeScript SDK release inputs and checks for new commits since the last release tag.
#
# Usage:
#   bash dev/validate-ts-release.sh --packages <comma-separated|all>
#
# Outputs JSON to stdout:
#   {"core": {"publish": true, "version": "0.2.0"}, "openai": {"publish": false, "version": "0.2.0"}, ...}
#
# Requirements: node, git (with full history and tags)

set -euo pipefail

PACKAGES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --packages)
      PACKAGES="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: bash dev/validate-ts-release.sh --packages <comma-separated|all>"
      echo ""
      echo "Valid package names: all, core, openai, anthropic, gemini"
      exit 0
      ;;
    *)
      echo "Error: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PACKAGES" ]]; then
  echo "Error: --packages is required" >&2
  exit 1
fi

# Parse and validate package selection
SELECTED=()

if [[ "$PACKAGES" == "all" ]]; then
  SELECTED=(core openai anthropic gemini)
else
  IFS=',' read -ra PARTS <<< "$PACKAGES"
  for pkg in "${PARTS[@]}"; do
    pkg=$(echo "$pkg" | xargs)
    case "$pkg" in
      core|openai|anthropic|gemini)
        SELECTED+=("$pkg")
        ;;
      *)
        echo "Error: Unknown package: '$pkg'. Valid values: all, core, openai, anthropic, gemini" >&2
        exit 1
        ;;
    esac
  done
fi

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  echo "Error: No packages selected." >&2
  exit 1
fi

echo "Selected packages: ${SELECTED[*]}" >&2

# Map package names to directories and npm names
declare -A PKG_DIRS=(
  [core]="libs/typescript/core"
  [openai]="libs/typescript/integrations/openai"
  [anthropic]="libs/typescript/integrations/anthropic"
  [gemini]="libs/typescript/integrations/gemini"
)

declare -A PKG_NPM_NAMES=(
  [core]="@mlflow/core"
  [openai]="@mlflow/openai"
  [anthropic]="@mlflow/anthropic"
  [gemini]="@mlflow/gemini"
)

# Build JSON output
JSON="{"
FIRST=true

for pkg in core openai anthropic gemini; do
  version=$(node -p "require('./${PKG_DIRS[$pkg]}/package.json').version")
  publish="false"

  # Check if this package was selected
  selected=false
  for s in "${SELECTED[@]}"; do
    if [[ "$s" == "$pkg" ]]; then
      selected=true
      break
    fi
  done

  if [[ "$selected" == "true" ]]; then
    npm_name="${PKG_NPM_NAMES[$pkg]}"
    tag="${npm_name}@${version}"
    pkg_dir="${PKG_DIRS[$pkg]}"

    if git rev-parse "refs/tags/${tag}" >/dev/null 2>&1; then
      commit_count=$(git log "${tag}..HEAD" --oneline -- "${pkg_dir}" | wc -l | xargs)
      if [[ "$commit_count" -eq 0 ]]; then
        echo "Skipping ${npm_name}@${version}: no commits since tag ${tag}" >&2
      else
        echo "${npm_name}: ${commit_count} commit(s) since ${tag}" >&2
        publish="true"
      fi
    else
      echo "${npm_name}: no existing tag '${tag}' found (first release of this version)" >&2
      publish="true"
    fi
  fi

  if [[ "$FIRST" == "true" ]]; then
    FIRST=false
  else
    JSON+=","
  fi
  JSON+="\"${pkg}\":{\"publish\":${publish},\"version\":\"${version}\"}"
done

JSON+="}"
echo "$JSON"
