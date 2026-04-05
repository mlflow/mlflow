#!/usr/bin/env bash
# Downloads the Claude Code binary directly from GCS with a pinned SHA256 checksum,
# avoiding `curl | bash` which pipes an unverified script with access to CI secrets.
# Ref: https://github.com/dagster-io/erk/blob/61ecee08754717959bb2f9cb6e7079df81ba80ea/.github/actions/setup-claude-code/action.yml
#
# Usage: .claude/scripts/install-claude.sh
set -euo pipefail

if [ "${CI:-}" != "true" ]; then
  echo "Error: This script is intended for CI only." >&2
  exit 1
fi

VERSION="2.1.71"
CHECKSUM="61002e5f5c4190e9a775bd9cf90e57fff3f0379fb2c8edc653ac0942a347babd"
GCS_BUCKET="https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases"
PLATFORM="linux-x64"

tmp_claude="$(mktemp)"
trap 'rm -f "$tmp_claude"' EXIT

curl -fsSL --retry 3 --retry-delay 2 "$GCS_BUCKET/$VERSION/$PLATFORM/claude" -o "$tmp_claude"
echo "${CHECKSUM}  $tmp_claude" | sha256sum -c -
mkdir -p ~/.local/bin
chmod +x "$tmp_claude"
mv "$tmp_claude" ~/.local/bin/claude
trap - EXIT
echo "Installed Claude Code $VERSION"
