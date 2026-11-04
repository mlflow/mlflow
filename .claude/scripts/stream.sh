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
        (if .name == "Bash" and (.input | type == "object") then
          .input as $i
          | ($i | if has("run_in_background") then {run_in_background} else {} end)
            + (if $i | has("timeout") then {timeout: "\($i.timeout / 1000)s"} else {} end)
            + ($i | if has("description") then {description} else {} end)
            + ($i | del(.run_in_background, .timeout, .description))
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
    elif .type == "system" and .subtype == "background_tasks_changed" then
      "⏳ Background tasks: \(.tasks | length)"
    elif .type == "system" and .subtype == "init" then
      "🚀 init: \(.model) (v\(.claude_code_version), session \(.session_id[0:8]))"
    elif .type == "result" then
      "✅ Done (\((.duration_ms / 100 | round) / 10)s, \(.num_turns) turns, \(.usage.input_tokens + .usage.output_tokens) tokens, $\(.total_cost_usd * 100 | round / 100))"
    else
      empty
    end' \
  | {
    if [ ! -r /proc/meminfo ]; then
      exec cat
    fi
    # Prefix each line with the runner's current memory usage.
    while IFS= read -r line; do
      while read -r key value _; do
        case $key in
          MemTotal:) mem_total=$value ;;
          MemAvailable:) mem_available=$value ;;
        esac
      done < /proc/meminfo
      # kB to tenths of GiB
      used=$(((mem_total - mem_available) * 10 / 1048576))
      total=$((mem_total * 10 / 1048576))
      # Pad the used amount to the total's width so the prefixes line up.
      total_gib=$((total / 10))
      printf '[mem %*d.%d/%d.%dG] %s\n' \
        "${#total_gib}" $((used / 10)) $((used % 10)) "$total_gib" $((total % 10)) "$line"
    done
  }
exit "${PIPESTATUS[1]}"
