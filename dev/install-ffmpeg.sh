#!/usr/bin/env bash
set -euo pipefail

FFMPEG_VERSION="n7.1.3-42-g39ee683e8f"
FFMPEG_BUILD="autobuild-2026-02-28-12-59"
FFMPEG_ARCHIVE="ffmpeg-${FFMPEG_VERSION}-linux64-gpl-7.1.tar.xz"
FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/${FFMPEG_BUILD}/${FFMPEG_ARCHIVE}"
FFMPEG_SHA256="72f2b6d13e87c9bda02b07c6079b6c97457295b0ac940279d9d83a70c1ac163d"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
curl -sL -o "$TMPDIR/ffmpeg.tar.xz" "$FFMPEG_URL"
echo "${FFMPEG_SHA256}  $TMPDIR/ffmpeg.tar.xz" | sha256sum -c -
tar -xJf "$TMPDIR/ffmpeg.tar.xz" -C "$TMPDIR"
cp "$TMPDIR/${FFMPEG_ARCHIVE%.tar.xz}/bin/ffmpeg" /usr/local/bin/
ffmpeg -version
