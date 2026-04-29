#!/usr/bin/env bash
# Usage: claude --output-format stream-json ... | .claude/scripts/stream.sh [output-file]

tee "${1:-/dev/null}" \
  | jq --unbuffered -r '
    if .type == "assistant" then
      .message.content[] |
      if .type == "text" then
        "🤖 \(.text)"
      elif .type == "tool_use" then
        "🔧 \(.name)\(if .input then ": \(.input | tostring | .[0:200])" else "" end)"
      elif .type == "thinking" then
        "🧠 thinking (\(.thinking | length) chars)"
      else
        empty
      end
    elif .type == "user" then
      .message.content[]?
      | select(.type == "tool_result")
      | "📥 tool_result (\(.content | tostring | length) chars)\(if .is_error then " ❌" else "" end)"
    elif .type == "result" then
      "✅ Done (\((.duration_ms / 100 | round) / 10)s, \(.num_turns) turns, \(.usage.input_tokens + .usage.output_tokens) tokens, $\(.total_cost_usd * 100 | round / 100))"
    else
      empty
    end'
