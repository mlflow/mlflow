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
if [[ "$is_draft" == "true" ]]; then
  gh pr ready "$pr_number" --repo "$repo"
  echo "Marked PR #${pr_number} as ready for review"
fi
