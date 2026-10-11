#!/usr/bin/env bash
# Usage: claude --output-format stream-json ... | .claude/scripts/stream.sh [output-file]

tee "${1:-/dev/null}" \
  | jq --unbuffered -r '
    {
      Bash: "💻", Read: "📖", Write: "📝", Edit: "✏️",
      NotebookEdit: "✏️", Glob: "📁", Grep: "🔬",
      WebSearch: "🔍", WebFetch: "🌐", Task: "🤝",
      Agent: "🤝", Skill: "🎓", TodoWrite: "☑️"
    } as $tools
    | if .type == "assistant" then
      .message.content[] |
      if .type == "text" then
        "🤖 \(.text)"
      elif .type == "tool_use" then
        (if .name == "Bash" and (.input | type == "object") and (.input | has("description")) then
          .input | {description} + del(.description)
        else
          .input
        end) as $input
        | "\($tools[.name] // "🔧") \(.name)\(if $input then ": \($input | tostring | .[0:200])" else "" end)"
      elif .type == "thinking" and (.thinking | length) > 0 then
        "🧠 thinking (\(.thinking | length) chars)"
      else
        empty
      end
    elif .type == "user" then
      .message.content[]?
      | select(.type == "tool_result")
      | (.content | tostring) as $c
      | if .is_error then
          "❌ tool_result error: \($c[0:200])"
        else
          "📥 tool_result (\($c | length) chars)"
        end
    elif .type == "system" and .subtype == "init" then
      "🚀 init: \(.model) (v\(.claude_code_version), session \(.session_id[0:8]))"
    elif .type == "result" then
      "✅ Done (\((.duration_ms / 100 | round) / 10)s, \(.num_turns) turns, \(.usage.input_tokens + .usage.output_tokens) tokens, $\(.total_cost_usd * 100 | round / 100))"
    else
      empty
    end'
