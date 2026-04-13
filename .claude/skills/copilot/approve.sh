#!/usr/bin/env bash
# Rerun action_required workflow runs for Copilot PRs.
# The /approve API fails with 'not a fork pull request'. Use /rerun instead.
set -euo pipefail

repo="$1"
pr_number="$2"

head_sha=$(gh pr view "$pr_number" --repo "$repo" --json headRefOid --jq '.headRefOid')

run_ids=$(
  gh api --paginate "repos/${repo}/actions/runs?head_sha=${head_sha}" \
    --jq '
      .workflow_runs[]
      | select(.conclusion == "action_required" and .actor.login == "Copilot")
      | .id
    '
)

if [[ -z "$run_ids" ]]; then
  echo "No action_required workflow runs found"
  exit 0
fi

echo "Rerunning action_required workflows..."
pids=()
while IFS= read -r run_id; do
  (
    if gh api --method POST "repos/${repo}/actions/runs/${run_id}/rerun" >/dev/null 2>&1; then
      echo "  Rerun triggered for run $run_id"
    else
      echo "  Failed to rerun run $run_id"
    fi
  ) &
  pids+=($!)
done <<< "$run_ids"
wait "${pids[@]}"
