#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${TMPDIR:-/tmp}"

env -u NO_PROXY -u no_proxy claude \
  --model databricks-claude-sonnet-5-5 \
  --effort high \
  --max-budget-usd 5 \
  --permission-mode auto \
  --disallowed-tools WebSearch \
  --print \
  --verbose \
  --output-format stream-json \
  "$PROMPT"
