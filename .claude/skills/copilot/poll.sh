#!/usr/bin/env bash
set -euo pipefail

repo="$1"        # owner/repo format (e.g., mlflow/mlflow)
pr_number="$2"
max_seconds=1800  # 30 minutes

while true; do
  if (( SECONDS > max_seconds )); then
    echo "Timed out after ${max_seconds}s waiting for Copilot to finish"
    exit 1
  fi
  result=$(gh api "repos/${repo}/issues/${pr_number}/timeline" \
    --paginate \
    --jq '.[] | select(.event == "copilot_work_finished") | .created_at')
  if [[ -n "$result" ]]; then
    echo "Copilot finished at $result"
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
