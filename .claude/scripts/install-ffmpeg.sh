#!/usr/bin/env bash
set -euo pipefail

arch=$(uname -m)
case "$arch" in
  x86_64) platform=linux64 ;;
  aarch64 | arm64) platform=linuxarm64 ;;
  *)
    echo "Error: unsupported architecture $arch" >&2
    exit 1
    ;;
esac

# Daily builds expire. Select a dated release after the 72-hour cooldown.
# https://github.com/FFmpeg-Builds/FFmpeg-Builds/blob/master/README.md#release-retention-policy
asset=$(gh api 'repos/FFmpeg-Builds/FFmpeg-Builds/releases?per_page=20' | jq -er --arg platform "$platform" '
  (now - 72 * 3600) as $cutoff
  | map(select(.tag_name | startswith("autobuild-"))
        | select((.published_at | fromdateiso8601) <= $cutoff))
  | max_by(.published_at).assets[]
  | select(.name | endswith("-" + $platform + "-gpl-9.0.tar.xz"))
  | [.name, .browser_download_url, (.digest | ltrimstr("sha256:"))]
  | @tsv
')
IFS=$'\t' read -r filename url checksum <<< "$asset"

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
archive="$tmp_dir/$filename"
build=${filename%.tar.xz}

curl -fsSL --retry 3 "$url" -o "$archive"
printf '%s  %s\n' "$checksum" "$archive" | sha256sum -c -
tar -xJf "$archive" -C "$tmp_dir" "$build/bin/ffmpeg" "$build/bin/ffprobe"
install -Dm755 -t "$HOME/.local/bin" "$tmp_dir/$build/bin/ffmpeg" "$tmp_dir/$build/bin/ffprobe"
