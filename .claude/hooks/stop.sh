#!/usr/bin/env bash
set -euo pipefail

jq '
  select(.stop_hook_active != true)
  | (.background_tasks // [])
  | select(length > 0)
  | {
      decision: "block",
      reason: (
        "Active background tasks remain. Before finishing, use TaskStop to clean up unneeded tasks. Tasks: "
        + (map({id, type, status, description}) | tojson)
      )
    }
'
