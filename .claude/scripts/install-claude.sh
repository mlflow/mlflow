#!/usr/bin/env bash
# Downloads the Claude Code binary from the official distribution URL and verifies it against the
# checksum published in the per-version manifest, avoiding `curl | bash` which
# pipes an unverified script with access to CI secrets.
# Ref: https://github.com/dagster-io/erk/blob/61ecee08754717959bb2f9cb6e7079df81ba80ea/.github/actions/setup-claude-code/action.yml
set -euo pipefail

if [ "${CI:-}" != "true" ]; then
  echo "Error: This script is intended for CI only." >&2
  exit 1
fi

DOWNLOAD_URL="https://downloads.claude.ai/claude-code-releases"
PLATFORM="linux-x64"

VERSION="$(curl -fsSL --retry 3 --retry-delay 2 "$DOWNLOAD_URL/stable")"
CHECKSUM="$(curl -fsSL --retry 3 --retry-delay 2 "$DOWNLOAD_URL/$VERSION/manifest.json" \
  | jq -r ".platforms[\"$PLATFORM\"].checksum")"

tmp_claude="$(mktemp)"
trap 'rm -f "$tmp_claude"' EXIT

curl -fsSL --retry 3 --retry-delay 2 "$DOWNLOAD_URL/$VERSION/$PLATFORM/claude" -o "$tmp_claude"
echo "${CHECKSUM}  $tmp_claude" | sha256sum -c -
mkdir -p ~/.local/bin
chmod +x "$tmp_claude"
mv "$tmp_claude" ~/.local/bin/claude
trap - EXIT
echo "Installed Claude Code $VERSION"
