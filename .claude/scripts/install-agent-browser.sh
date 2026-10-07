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

VERSION="0.38.1"
PLATFORM="linux-x64"
CHECKSUM="5100149a1903211c889de4e545bf36d90803740cea4f99aa22651649f9205ea1"

URL="https://github.com/vercel-labs/agent-browser/releases/download/v$VERSION/agent-browser-$PLATFORM"

tmp_bin="$(mktemp)"
trap 'rm -f "$tmp_bin"' EXIT

curl -fsSL --connect-timeout 10 --max-time 120 --retry 3 --retry-delay 2 "$URL" -o "$tmp_bin"
echo "${CHECKSUM}  $tmp_bin" | sha256sum -c -
mkdir -p ~/.local/bin
chmod +x "$tmp_bin"
mv "$tmp_bin" ~/.local/bin/agent-browser
trap - EXIT
echo "Installed agent-browser $VERSION"
