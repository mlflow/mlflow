#!/usr/bin/env bash
#
# Benchmark the MLflow tracking server gateway (database-backed endpoints).
#
# Unlike run_comparison.sh which tests the standalone YAML-config gateway,
# this script tests the production code path where every request does:
#   - SQLAlchemy queries to fetch endpoint config + model definition
#   - Secret decryption
#   - Provider instance creation
#   - Optional budget limit checks
#
# Configuration via environment variables:
#   TRACKING_SERVER_WORKERS - Workers for mlflow server (default: 4)
#   REQUESTS               - Total requests per run (default: 2000)
#   MAX_CONCURRENT         - Max concurrent requests (default: 50)
#   RUNS                   - Number of benchmark runs (default: 3)
#   FAKE_RESPONSE_DELAY_MS - Simulated provider latency (default: 50)
#   USAGE_TRACKING         - Enable usage tracking/tracing (default: true, set "false" to disable)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v uv &>/dev/null && [ -f "$SCRIPT_DIR/../../pyproject.toml" ]; then
    RUN_PREFIX="uv run --extra gateway"
else
    RUN_PREFIX=""
fi

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

TRACKING_SERVER_WORKERS="${TRACKING_SERVER_WORKERS:-4}"
REQUESTS="${REQUESTS:-2000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
RUNS="${RUNS:-3}"
FAKE_RESPONSE_DELAY_MS="${FAKE_RESPONSE_DELAY_MS:-50}"
USAGE_TRACKING="${USAGE_TRACKING:-true}"
FAKE_SERVER_PORT=9000
MLFLOW_PORT=5000
ENDPOINT_NAME="benchmark-chat"

PIDS=()
TMPDIR_BENCH=$(mktemp -d)

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    rm -rf "$TMPDIR_BENCH"
    echo "Done."
}
trap cleanup EXIT

wait_for_port() {
    local port=$1
    local name=$2
    local health_path="${3:-/health}"
    local max_wait=30
    local waited=0
    echo "Waiting for $name on port $port..."
    while ! curl -s "http://127.0.0.1:$port$health_path" > /dev/null 2>&1; do
        sleep 0.5
        waited=$((waited + 1))
        if [ "$waited" -ge "$((max_wait * 2))" ]; then
            echo "ERROR: $name failed to start on port $port after ${max_wait}s"
            exit 1
        fi
    done
    echo "$name ready."
}

echo "=============================================="
echo " MLflow Tracking Server Gateway Benchmark"
echo "=============================================="
echo "Workers:        $TRACKING_SERVER_WORKERS"
echo "Requests/run:   $REQUESTS"
echo "Concurrency:    $MAX_CONCURRENT"
echo "Runs:           $RUNS"
echo "Fake delay:     ${FAKE_RESPONSE_DELAY_MS}ms"
echo "Usage tracking: $USAGE_TRACKING"
echo "DB:             SQLite (${TMPDIR_BENCH}/mlflow.db)"
echo ""

# 1. Start fake OpenAI server
echo "=== Starting fake OpenAI server ==="
FAKE_RESPONSE_DELAY_MS="$FAKE_RESPONSE_DELAY_MS" \
    $RUN_PREFIX gunicorn fake_openai_server:app \
    -k uvicorn.workers.UvicornWorker \
    -w 8 \
    -b "0.0.0.0:$FAKE_SERVER_PORT" \
    --log-level warning \
    > /dev/null 2>&1 &
PIDS+=($!)
wait_for_port "$FAKE_SERVER_PORT" "fake OpenAI server"

# 2. Start MLflow tracking server
echo "=== Starting MLflow tracking server (port $MLFLOW_PORT, $TRACKING_SERVER_WORKERS workers) ==="
$RUN_PREFIX mlflow server \
    --backend-store-uri "sqlite:///${TMPDIR_BENCH}/mlflow.db" \
    --host 0.0.0.0 \
    --port "$MLFLOW_PORT" \
    --workers "$TRACKING_SERVER_WORKERS" \
    --disable-security-middleware \
    > /dev/null 2>&1 &
PIDS+=($!)
wait_for_port "$MLFLOW_PORT" "MLflow tracking server" "/health"

# 3. Create gateway endpoint via REST API
echo ""
echo "=== Setting up gateway endpoint ==="
SETUP_ARGS=(
    --tracking-uri "http://127.0.0.1:$MLFLOW_PORT"
    --fake-server-url "http://127.0.0.1:$FAKE_SERVER_PORT/v1"
    --endpoint-name "$ENDPOINT_NAME"
)
if [ "$USAGE_TRACKING" = "false" ]; then
    SETUP_ARGS+=(--no-usage-tracking)
fi
$RUN_PREFIX python setup_tracking_server.py "${SETUP_ARGS[@]}"

# 4. Quick sanity check
echo ""
echo "=== Sanity check ==="
INVOKE_URL="http://127.0.0.1:$MLFLOW_PORT/gateway/$ENDPOINT_NAME/mlflow/invocations"
echo -n "Tracking server gateway: "
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$INVOKE_URL" \
    -X POST -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"test"}]}')
echo "$HTTP_CODE"
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: Sanity check failed (expected 200, got $HTTP_CODE)"
    echo "Trying with verbose output:"
    curl -s "$INVOKE_URL" \
        -X POST -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"test"}]}'
    echo ""
    exit 1
fi

# 5. Run benchmark
echo ""
echo "=== Running tracking server benchmark ==="
$RUN_PREFIX python benchmark_compare.py \
    --target tracking-server \
    --requests "$REQUESTS" \
    --max-concurrent "$MAX_CONCURRENT" \
    --runs "$RUNS" \
    --tracking-server-url "$INVOKE_URL"
