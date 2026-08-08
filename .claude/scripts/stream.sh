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
      elif .type == "thinking" and (.thinking | length) > 0 then
        "🧠 thinking (\(.thinking | length) chars)"
      else
        empty
      end
    elif .type == "user" then
      .message.content[]?
      | select(.type == "tool_result")
      | (.content | tostring) as $c
      # Denials arrive as ordinary is_error results, so flag them with their reason.
      | if .is_error and ($c | test("Permission for this action was denied")) then
          "🚫 permission denied: \($c[0:200])"
        else
          "📥 tool_result (\($c | length) chars)\(if .is_error then " ❌" else "" end)"
        end
    elif .type == "system" and .subtype == "init" then
      "🚀 init: \(.model) (v\(.claude_code_version), session \(.session_id[0:8]))"
    elif .type == "result" then
      "✅ Done (\((.duration_ms / 100 | round) / 10)s, \(.num_turns) turns, \(.usage.input_tokens + .usage.output_tokens) tokens, $\(.total_cost_usd * 100 | round / 100))"
      + (
        (.permission_denials // [])
        | if length == 0 then ""
          else "\n🚫 \(length) permission denial(s): " + (map("\(.tool_name)(\(.tool_input | tostring | .[0:120]))") | join("; "))
          end
      )
    else
      empty
    end'
