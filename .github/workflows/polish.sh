#!/usr/bin/env bash
set -euo pipefail

# Fetch PR information
echo "Fetching PR information..." >&2
pr_json=$(gh pr view "$PR_NUMBER" --repo "$REPO" --json title,body)
original_title=$(echo "$pr_json" | jq -r '.title')
pr_body=$(echo "$pr_json" | jq -r '.body // ""')

echo "Original title: $original_title" >&2

# Fetch PR diff
echo "Fetching PR diff..." >&2
diff=$(gh pr diff "$PR_NUMBER" --repo "$REPO" 2>/dev/null || echo "")

# Truncate diff if too long (keep first 50000 chars)
max_diff_length=50000
if [ ${#diff} -gt $max_diff_length ]; then
  diff="${diff:0:$max_diff_length}

... [diff truncated due to length] ..."
fi

# Build the prompt
read -r -d '' prompt << 'PROMPT_EOF' || true
You are an expert at writing clear, concise pull request titles for the MLflow open-source project.

Given the following information about a pull request, rewrite the title to be more descriptive and follow best practices.

## Current PR Title
__ORIGINAL_TITLE__

## PR Description
__PR_BODY__

## Code Changes (Diff)
```diff
__DIFF__
```

## Guidelines for a good PR title:
1. Start with a verb in imperative mood (e.g., "Add", "Fix", "Update", "Remove", "Refactor")
2. Be specific about what changed and where
3. Keep it concise
4. Do not include issue numbers in the title (they belong in the PR body)
5. Focus on the "what" and "why", not the "how"
6. Use proper capitalization (capitalize first letter, no period at end)
7. Use backticks for code/file references (e.g., `ClassName`, `function_name`, `module.path`)

Rewrite the PR title following these guidelines.
PROMPT_EOF

# Substitute placeholders
prompt="${prompt/__ORIGINAL_TITLE__/$original_title}"
prompt="${prompt/__PR_BODY__/${pr_body:-(No description provided)}}"
prompt="${prompt/__DIFF__/$diff}"

# Build the API request payload with structured output
echo "Calling Claude API..." >&2
request_payload=$(jq -n \
  --arg prompt "$prompt" \
  '{
    model: "claude-haiku-4-5-20251001",
    max_tokens: 256,
    messages: [{ role: "user", content: $prompt }],
    output_config: {
      format: {
        type: "json_schema",
        schema: {
          type: "object",
          properties: {
            title: {
              type: "string",
              description: "The rewritten PR title. Should be concise, descriptive, and follow the guidelines."
            }
          },
          required: ["title"],
          additionalProperties: false
        }
      }
    }
  }')

# Call the Anthropic API
response=$(curl -s -X POST "https://api.anthropic.com/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d "$request_payload")

# Log the full response for debugging (token usage, cost, etc.)
echo "API Response:" >&2
echo "$response" | jq . >&2

# Check for errors
if echo "$response" | jq -e '.error' > /dev/null 2>&1; then
  error_message=$(echo "$response" | jq -r '.error.message')
  echo "Error from Claude API: $error_message" >&2
  exit 1
fi

# Extract and output the new title (response is JSON in content[0].text)
echo "$response" | jq -r '.content[0].text' | jq -r '.title'
