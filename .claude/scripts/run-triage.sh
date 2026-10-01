#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${TMPDIR:-/tmp}"

if [ "$NEEDS_UI" = true ]; then
  log=/tmp/triage-ui.log
  mkdir -p /tmp/triage-ui/artifacts
  if [ -f mlflow/server/js/build/index.html ]; then
    ui_url=http://localhost:5000
    uv run mlflow server --host 127.0.0.1 --port 5000 \
      --backend-store-uri sqlite:////tmp/triage-ui/mlflow.db \
      --default-artifact-root /tmp/triage-ui/artifacts \
      > "$log" 2>&1 &
  else
    HOST=localhost CI=false uv run dev/run_dev_server.py > "$log" 2>&1 &
  fi
  ui_pid=$!
  trap 'kill "$ui_pid" 2>/dev/null || true' EXIT

  for _ in $(seq 1 360); do
    if [ -z "${ui_url:-}" ]; then
      ui_url=$(sed -n 's/^Frontend: \(http:\/\/localhost:[0-9]*\).*/\1/p' "$log" | tail -1)
    fi
    if [ -n "${ui_url:-}" ] && curl --noproxy '*' -fsS "$ui_url/" > /dev/null; then
      printf '%s\n' "$ui_url" > /tmp/triage-ui-url
      break
    fi
    if ! kill -0 "$ui_pid" 2> /dev/null; then
      tail -80 "$log" >&2
      exit 1
    fi
    sleep 2
  done
  if [ ! -s /tmp/triage-ui-url ]; then
    tail -80 "$log" >&2
    exit 1
  fi
fi

env -u NO_PROXY -u no_proxy claude \
  --model databricks-claude-sonnet-5-5 \
  --effort high \
  --max-budget-usd 5 \
  --permission-mode auto \
  --disallowed-tools WebSearch \
  --print \
  --verbose \
  --output-format stream-json \
  "$PROMPT"
