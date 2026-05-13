#!/usr/bin/env bash
set -euo pipefail

repo="$1"        # owner/repo format
pr_number="$2"
max_seconds=1800  # 30 minutes

# Find the latest session for this PR
session_id=$(
  gh agent-task list \
    --json id,pullRequestNumber,createdAt,repository \
    --jq "
      [.[] | select(.repository == \"${repo}\" and .pullRequestNumber == ${pr_number})]
      | sort_by(.createdAt)
      | last
      | .id
    "
)
echo "Polling session $session_id for PR #${pr_number}"

while true; do
  if (( SECONDS > max_seconds )); then
    echo "Timed out after ${max_seconds}s waiting for Copilot to finish"
    exit 1
  fi
  state=$(gh agent-task view "$session_id" --json state --jq '.state')
  echo "State: $state (elapsed ${SECONDS}s)"
  if [[ "$state" != "queued" && "$state" != "in_progress" ]]; then
    echo "Copilot finished with state: $state"
    break
  fi
  sleep 30
done

# Mark PR ready for review if still in draft
is_draft=$(gh pr view "$pr_number" --repo "$repo" --json isDraft --jq '.isDraft')
transitioned_to_ready=false
if [[ "$is_draft" == "true" ]]; then
  gh pr ready "$pr_number" --repo "$repo"
  echo "Marked PR #${pr_number} as ready for review"
  transitioned_to_ready=true
fi

# Poll for Copilot review completion only if we just transitioned from draft to ready
if [[ "$transitioned_to_ready" == "true" ]]; then
  review_max_seconds=600  # 10 minutes
  review_start=$SECONDS
  echo "Waiting for Copilot review..."
  while true; do
    if (( SECONDS - review_start > review_max_seconds )); then
      echo "Warning: Timed out after ${review_max_seconds}s waiting for Copilot review"
      break
    fi
    review_info=$(
      gh api "repos/${repo}/pulls/${pr_number}/reviews" \
        --jq '
          [.[] | select(.user.login == "copilot-pull-request-reviewer[bot]")]
          | last
          | {state: (.state // empty), id: (.id // empty)}
        ' 2>/dev/null \
        || {
          echo "Warning: Failed to fetch Copilot review state; treating as no review yet" >&2
          echo '{}'
        }
    )
    review_state=$(echo "$review_info" | jq -r '.state // empty')
    review_id=$(echo "$review_info" | jq -r '.id // empty')
    if [[ -n "$review_state" ]]; then
      comment_count=$(
        gh api "repos/${repo}/pulls/${pr_number}/reviews/${review_id}/comments" \
          --jq 'length' 2>/dev/null || echo "0"
      )
      echo "Copilot review: $review_state ($comment_count comment(s), elapsed $((SECONDS - review_start))s)"
      break
    fi
    echo "Waiting for Copilot review... (elapsed $((SECONDS - review_start))s)"
    sleep 30
  done
fi
