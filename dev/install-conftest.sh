#!/usr/bin/env bash
set -euo pipefail

DESTINATION="/tmp/conftest/bin"
mkdir -p "$DESTINATION"
curl -s -L "https://github.com/open-policy-agent/conftest/releases/download/v0.56.0/conftest_0.56.0_Linux_x86_64.tar.gz" | tar -xz -C "$DESTINATION"
if [[ "$GITHUB_ACTIONS" == "true" ]]; then
  echo "$DESTINATION" >> "$GITHUB_PATH"
fi
