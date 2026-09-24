#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--dry-run] OWNER/REPO SHA"
  echo "Rerun failed jobs for a commit; skip if checks are pending or a failed run"
  echo "has exhausted MAX_RETRIES (default: 3, excluding the initial attempt)."
}

if [[ $# == 1 && ( "$1" == --help || "$1" == -h ) ]]; then
  usage
  exit 0
fi
dry_run=false
if [[ ${1:-} == --dry-run ]]; then
  dry_run=true
  shift
fi
if [[ $# != 2 || ! "$1" =~ ^([^/]+)/([^/]+)$ ]]; then
  usage >&2
  exit 1
fi
owner=${BASH_REMATCH[1]}
repo=${BASH_REMATCH[2]}
repository=$1
sha=$2
max_retries=${MAX_RETRIES:-3}
if [[ ! "$sha" =~ ^[[:xdigit:]]{40}$ || ! "$max_retries" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "Error: SHA must be a full commit SHA and MAX_RETRIES a non-negative integer." >&2
  exit 1
fi

pages=$(gh api graphql --paginate --slurp \
  -f owner="$owner" -f name="$repo" -f sha="$sha" \
  -f query='
    query($owner: String!, $name: String!, $sha: String!, $endCursor: String) {
      repository(owner: $owner, name: $name) {
        object(expression: $sha) {
          ... on Commit {
            checkSuites(first: 100, after: $endCursor) {
              nodes {
                status
                conclusion
                app { slug }
                workflowRun {
                  databaseId
                  runAttempt
                  event
                  createdAt
                  workflow { resourcePath }
                }
              }
              pageInfo { hasNextPage endCursor }
            }
          }
        }
      }
    }')

# Deduplicate across pages before inspecting failures or retry budgets.
suites=$(jq -e '
  if any(.[]; .errors != null or .data.repository.object == null) then
    error("Could not resolve commit check suites")
  else
    [.[].data.repository.object.checkSuites.nodes[]?
     | select(.app.slug == "github-actions" and .workflowRun != null)
     # GitHub-managed runs such as Copilot reviews are not retry candidates.
     | select(.workflowRun.event != "dynamic")]
    | group_by([.workflowRun.workflow.resourcePath, .workflowRun.event])
    | map(max_by([.workflowRun.createdAt, .workflowRun.databaseId]))
  end
' <<< "$pages")

if jq -e 'any(.[]; .status != "COMPLETED")' <<< "$suites" >/dev/null; then
  echo "Skipping $sha: workflows are pending or in progress."
  exit 0
fi

runs=$(jq '
  [.[]
   | select(.conclusion == "FAILURE" or .conclusion == "TIMED_OUT"
            or .conclusion == "STARTUP_FAILURE")
   | {id: .workflowRun.databaseId,
      attempt: .workflowRun.runAttempt,
      rerun: (.workflowRun.workflow.resourcePath |
              endswith("/rerun.yml") or endswith("/retry.yml") | not),
      workflow: (.workflowRun.workflow.resourcePath | split("/")[-1])}]
' <<< "$suites")

if jq -e --argjson limit "$max_retries" \
  'any(.[]; .attempt == null or .attempt - 1 >= $limit)' <<< "$runs" >/dev/null; then
  echo "Skipping $sha: a failed run has exhausted its retry budget or has no attempt count."
  exit 0
fi

rows=$(jq -r '.[] | select(.rerun) | [.id, .workflow] | @tsv' <<< "$runs")
if [[ -z "$rows" ]]; then
  echo "No eligible failed workflows for $sha."
  exit 0
fi

rerun() {
  if [[ "$dry_run" == true ]]; then
    echo "Would rerun failed jobs for $2: https://github.com/$repository/actions/runs/$1"
    return 0
  fi
  gh run rerun "$1" --repo "$repository" --failed
  echo "::notice title=Rerun failed jobs::$2: https://github.com/$repository/actions/runs/$1"
}

protect_id=""
protect_workflow=""
while IFS=$'\t' read -r run_id workflow; do
  if [[ "$workflow" == protect.yml ]]; then
    protect_id=$run_id
    protect_workflow=$workflow
  else
    rerun "$run_id" "$workflow" &
  fi
done <<< "$rows"

# Wait for every rerun request, not for the workflows themselves to finish.
wait
if [[ -n "$protect_id" ]]; then
  rerun "$protect_id" "$protect_workflow"
fi
