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

VERSION="0.45.0"
case "$(uname -m)" in
  x86_64)
    PLATFORM="linux-x64"
    CHECKSUM="5eff2baaabb337ec728e4571ccfb228ad1fb6babce529fcaf7a6062fcdec6087"
    ;;
  aarch64 | arm64)
    PLATFORM="linux-arm64"
    CHECKSUM="dcdab91ae7ce8ee3bd6a189e3e5ce41b8f713e4b07c25cb18506b984960cd224"
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
