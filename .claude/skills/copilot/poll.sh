#!/usr/bin/env bash
set -euo pipefail

session_id="$1"
repo="$2"        # owner/repo format
pr_number="$3"
max_seconds=1800  # 30 minutes

while true; do
  if (( SECONDS > max_seconds )); then
    echo "Timed out after ${max_seconds}s waiting for Copilot to finish"
    exit 1
  fi
  state=$(gh agent-task view "$session_id" --json state --jq '.state')
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
