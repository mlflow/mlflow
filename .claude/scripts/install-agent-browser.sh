#!/usr/bin/env bash
# Installs the agent-browser binary from its GitHub release. The npm package only wraps
# these same per-platform binaries, so downloading one directly avoids needing Node.
# Upstream publishes no checksum manifest, so the digest below is pinned here and must
# be recomputed when VERSION changes.
set -euo pipefail

if [ "${CI:-}" != "true" ]; then
  echo "Error: This script is intended for CI only." >&2
  exit 1
fi

VERSION="0.34.0"
PLATFORM="linux-x64"
CHECKSUM="69eadf5d8d6003a06a5cd2f914ebb261c7754fe1335a9190122c334e91909789"

URL="https://github.com/vercel-labs/agent-browser/releases/download/v$VERSION/agent-browser-$PLATFORM"

tmp_bin="$(mktemp)"
trap 'rm -f "$tmp_bin"' EXIT

curl -fsSL --retry 3 --retry-delay 2 "$URL" -o "$tmp_bin"
echo "${CHECKSUM}  $tmp_bin" | sha256sum -c -
mkdir -p ~/.local/bin
chmod +x "$tmp_bin"
mv "$tmp_bin" ~/.local/bin/agent-browser
trap - EXIT
echo "Installed agent-browser $VERSION"
