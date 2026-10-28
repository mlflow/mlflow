#!/usr/bin/env bash
# Answer a question with a web search, through the gateway's OpenAI surface.
# The proxy attaches the credential, so no token is needed here.
#
# TODO: delete this script and drop `--disallowed-tools WebSearch` from
# `review.yml` once the gateway serves Anthropic's own web_search. Today it
# reaches Anthropic models only through an MCP server:
# https://docs.databricks.com/aws/en/machine-learning/model-serving/web-search
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo 'usage: search-web.sh "<query>"' >&2
  exit 2
fi

: "${WEB_SEARCH_BASE_URL:?WEB_SEARCH_BASE_URL is unset}"
MODEL="${WEB_SEARCH_MODEL:-databricks-gpt-5-4-mini}"

request=$(jq -n --arg model "$MODEL" --arg input "$*" '{
  model: $model,
  tools: [{type: "web_search"}],
  # Declaring the tool only offers it, and an answer from model knowledge is
  # exactly what the caller came here to avoid.
  tool_choice: "required",
  input: $input,
  # Reasoning counts against this, and a review question spends more of it than
  # the answer does.
  max_output_tokens: 4000,
  reasoning: {effort: "low"}
}')

# `--retry` covers the transient statuses and connection failures, not a 400, so
# a blip costs a moment rather than the answer.
if ! response=$(curl -sS --fail-with-body --max-time 180 --retry 2 \
  -H "Content-Type: application/json" -d "$request" "$WEB_SEARCH_BASE_URL/responses"); then
  echo "search-web: request failed: $response" >&2
  exit 1
fi

# The answer carries its citations inline: each url_citation annotation is a pair
# of offsets into this text, not a source the text leaves out.
answer=$(jq -r '[.output[]? | select(.type == "message") | .content[]?.text] | join("\n")' <<<"$response")

status=$(jq -c '{status, incomplete_details, error}' <<<"$response")

# A budget spent entirely on reasoning, a refusal, or an error object served at
# 200 all reach here with nothing to print, which reads like a search that found
# nothing. The status alone says which: the response itself is mostly encrypted
# reasoning blobs.
if [ -z "$answer" ]; then
  echo "search-web: no answer: $status" >&2
  exit 1
fi

# `tool_choice` asks for a search, but only the response says whether one ran: a
# backend that ignored it answers from model knowledge and reads the same.
if [ "$(jq -r '[.output[]?.type] | index("web_search_call")' <<<"$response")" = "null" ]; then
  echo "search-web: answer did not come from a search: $status" >&2
  exit 1
fi

# A truncated answer stops mid-sentence but otherwise reads like a whole one.
if [ "$(jq -r '.status' <<<"$response")" != "completed" ]; then
  echo "search-web: $status" >&2
fi

printf '%s\n' "$answer"
