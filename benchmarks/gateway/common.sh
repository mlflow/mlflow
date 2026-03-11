#!/usr/bin/env bash
#
# Shared shell functions for gateway benchmark scripts.
# Sourced by run_comparison.sh, run_tracking_server_benchmark.sh, and run_full_stack_comparison.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v uv &>/dev/null && [ -f "$SCRIPT_DIR/../../pyproject.toml" ]; then
    RUN_PREFIX="uv run --extra gateway"
else
    RUN_PREFIX=""
fi

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Port constants
FAKE_SERVER_PORT=9000
MLFLOW_PORT=5000
LITELLM_PORT=4000
ENDPOINT_NAME="benchmark-chat"

# Default env vars (scripts can override before calling functions)
REQUESTS="${REQUESTS:-2000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
RUNS="${RUNS:-3}"
FAKE_RESPONSE_DELAY_MS="${FAKE_RESPONSE_DELAY_MS:-50}"

PIDS=()
TMPDIR_BENCH=$(mktemp -d)

# Set by scripts that need Docker cleanup (e.g. run_full_stack_comparison.sh).
# Read at runtime by cleanup(), so setting it after sourcing works fine.
DOCKER_CLEANUP_CMD="${DOCKER_CLEANUP_CMD:-}"

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
    if [ -n "$DOCKER_CLEANUP_CMD" ]; then
        eval "$DOCKER_CLEANUP_CMD" || true
    fi
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

start_fake_server() {
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
}

start_mlflow_server() {
    local workers=$1
    local backend_store_uri=$2
    echo "=== Starting MLflow server (port $MLFLOW_PORT, $workers workers) ==="
    $RUN_PREFIX mlflow server \
        --backend-store-uri "$backend_store_uri" \
        --host 0.0.0.0 \
        --port "$MLFLOW_PORT" \
        --workers "$workers" \
        --disable-security-middleware \
        > /dev/null 2>&1 &
    PIDS+=($!)
    wait_for_port "$MLFLOW_PORT" "MLflow server" "/health"
}

setup_gateway_endpoint() {
    echo ""
    echo "=== Setting up gateway endpoint ==="
    local setup_args=(
        --tracking-uri "http://127.0.0.1:$MLFLOW_PORT"
        --fake-server-url "http://127.0.0.1:$FAKE_SERVER_PORT/v1"
        --endpoint-name "$ENDPOINT_NAME"
    )
    if [ "$USAGE_TRACKING" = "false" ]; then
        setup_args+=(--no-usage-tracking)
    fi
    $RUN_PREFIX python setup_tracking_server.py "${setup_args[@]}"
}

sanity_check_mlflow() {
    local url=$1
    echo -n "MLflow: "
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" \
        -X POST -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"test"}]}')
    echo "$http_code"
    if [ "$http_code" != "200" ]; then
        echo "ERROR: MLflow sanity check failed (expected 200, got $http_code)"
        curl -s "$url" \
            -X POST -H "Content-Type: application/json" \
            -d '{"messages":[{"role":"user","content":"test"}]}'
        echo ""
        exit 1
    fi
}

sanity_check_litellm() {
    local url=$1
    echo -n "LiteLLM: "
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" \
        -X POST -H "Content-Type: application/json" -H "Authorization: Bearer sk-1234" \
        -d '{"model":"benchmark-chat","messages":[{"role":"user","content":"test"}]}')
    echo "$http_code"
    if [ "$http_code" != "200" ]; then
        echo "ERROR: LiteLLM sanity check failed (expected 200, got $http_code)"
        curl -s "$url" \
            -X POST -H "Content-Type: application/json" -H "Authorization: Bearer sk-1234" \
            -d '{"model":"benchmark-chat","messages":[{"role":"user","content":"test"}]}'
        echo ""
        exit 1
    fi
}

run_benchmark() {
    local target=$1
    local mlflow_url="${2:-}"
    local litellm_url="${3:-}"
    echo ""
    echo "=== Running benchmark ==="
    local args=(
        --target "$target"
        --requests "$REQUESTS"
        --max-concurrent "$MAX_CONCURRENT"
        --runs "$RUNS"
    )
    if [ -n "$mlflow_url" ]; then
        args+=(--mlflow-url "$mlflow_url")
    fi
    if [ -n "$litellm_url" ]; then
        args+=(--litellm-url "$litellm_url")
    fi
    $RUN_PREFIX python benchmark_compare.py "${args[@]}"
}
