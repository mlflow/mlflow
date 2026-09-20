#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--dry-run] OWNER/REPO SHA"
  echo "Rerun failed jobs for a commit; skip if checks are pending or a failed run"
  echo "has reached MAX_ATTEMPTS (default: 5, including the initial attempt)."
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
max_attempts=${MAX_ATTEMPTS:-5}
if [[ ! "$sha" =~ ^[[:xdigit:]]{40}$ || ! "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: SHA must be a full commit SHA and MAX_ATTEMPTS a positive integer." >&2
  exit 1
fi

pages=$(gh api graphql --paginate --slurp \
  -f owner="$owner" -f name="$repo" -f sha="$sha" \
  -f query='
    query($owner: String!, $name: String!, $sha: String!, $endCursor: String) {
      repository(owner: $owner, name: $name) {
        object(expression: $sha) {
          ... on Commit {
            statusCheckRollup {
              contexts(first: 100, after: $endCursor) {
                nodes {
                  __typename
                  ... on StatusContext { state }
                  ... on CheckRun {
                    name
                    status
                    conclusion
                    startedAt
                    checkSuite {
                      app { slug }
                      workflowRun {
                        databaseId
                        runAttempt
                        event
                        workflow { name }
                      }
                    }
                  }
                }
                pageInfo { hasNextPage endCursor }
              }
            }
          }
        }
      }
    }')

# Deduplicate across pages before inspecting failures or retry budgets.
checks=$(jq -e '
  if any(.[]; .errors != null or .data.repository.object == null) then
    error("Could not resolve commit checks")
  else
    [.[].data.repository.object.statusCheckRollup.contexts.nodes[]?]
    | ([.[] | select(.__typename == "CheckRun")]
       | group_by([.name, .checkSuite.app.slug,
                   .checkSuite.workflowRun.workflow.name, .checkSuite.workflowRun.event])
       | map(max_by(.startedAt // "")))
      + [.[] | select(.__typename == "StatusContext")]
  end
' <<< "$pages")

if jq -e 'any(.[];
  (.__typename == "CheckRun" and .status != "COMPLETED")
  or (.__typename == "StatusContext" and .state == "PENDING")
)' <<< "$checks" >/dev/null; then
  echo "Skipping $sha: checks are pending or in progress."
  exit 0
fi

runs=$(jq '
  [.[]
   | select(.__typename == "CheckRun" and .checkSuite.app.slug == "github-actions")
   | select(.conclusion == "FAILURE" or .conclusion == "TIMED_OUT"
            or .conclusion == "STARTUP_FAILURE")
   | select(.checkSuite.workflowRun.databaseId != null)
   | {id: .checkSuite.workflowRun.databaseId,
      attempt: .checkSuite.workflowRun.runAttempt,
      rerun: ((.name | ascii_downcase) != "rerun"),
      protect: ((.name | ascii_downcase) == "protect")}]
  | group_by(.id)
  | map({id: .[0].id, attempt: (map(.attempt) | max),
         rerun: any(.[]; .rerun), protect: any(.[]; .protect)})
' <<< "$checks")

if jq -e --argjson limit "$max_attempts" \
  'any(.[]; .attempt == null or .attempt >= $limit)' <<< "$runs" >/dev/null; then
  echo "Skipping $sha: a failed run has exhausted its retry budget or has no attempt count."
  exit 0
fi

rows=$(jq -r '.[] | select(.rerun) | [.id, .protect] | @tsv' <<< "$runs")
if [[ -z "$rows" ]]; then
  echo "No eligible failed workflows for $sha."
  exit 0
fi

rerun() {
  if [[ "$dry_run" == true ]]; then
    echo "Would rerun failed jobs: https://github.com/$repository/actions/runs/$1"
    return 0
  fi
  echo "Rerunning failed jobs: https://github.com/$repository/actions/runs/$1"
  gh run rerun "$1" --repo "$repository" --failed
}

pids=""
protect_id=""
while IFS=$'\t' read -r run_id protect; do
  if [[ "$protect" == true ]]; then
    protect_id=$run_id
  else
    rerun "$run_id" &
    pids+="$! "
  fi
done <<< "$rows"

# Wait for every rerun request, not for the workflows themselves to finish.
result=0
for pid in $pids; do
  if ! wait "$pid"; then
    result=1
  fi
done
if [[ -n "$protect_id" ]]; then
  if ! rerun "$protect_id"; then
    result=1
  fi
fi
exit "$result"
