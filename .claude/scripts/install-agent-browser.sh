#!/usr/bin/env bash
# Installs the agent-browser binary from its GitHub release. The npm package only wraps
# these same per-platform binaries, so downloading one directly avoids needing Node.
# Upstream publishes no checksum manifest, so the digests below are pinned here and must
# be recomputed for every platform when VERSION changes.
set -euo pipefail

if [ "${CI:-}" != "true" ]; then
  echo "Error: This script is intended for CI only." >&2
  exit 1
fi

VERSION="0.39.1"
case "$(uname -m)" in
  x86_64)
    PLATFORM="linux-x64"
    CHECKSUM="6906d3a2a67b05809932a93d99e53d1132373f0a52a617b2d6032dbb954df6fc"
    ;;
  aarch64 | arm64)
    PLATFORM="linux-arm64"
    CHECKSUM="9d8fdc586056df65747927ae166157c83af91b52cea287175e5544afd2591d33"
    ;;
  *)
    echo "Error: unsupported architecture $(uname -m)" >&2
    exit 1
    ;;
esac

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
