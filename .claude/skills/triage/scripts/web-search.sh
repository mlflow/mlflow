#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo 'usage: web-search.sh "<query>"' >&2
  exit 2
fi

jq -nc --arg input "$*" '{
  model: "databricks-gpt-6-luna",
  input: $input,
  tools: [{type: "web_search"}],
  tool_choice: "required"
}' | curl -sS --fail-with-body --max-time 60 \
  -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" \
  -H "$ANTHROPIC_CUSTOM_HEADERS" \
  --json @- \
  "${ANTHROPIC_BASE_URL%/anthropic}/openai/v1/responses"
