#!/usr/bin/env bash
#
# Head-to-head benchmark: MLflow Gateway vs LiteLLM proxy
#
# Both proxies hit the same fake OpenAI server on the same machine.
# Uses aiohttp-based benchmark (same methodology as LiteLLM's benchmark_mock.py).
#
# Configuration via environment variables:
#   GATEWAY_WORKERS      - Workers for both proxies (default: 4)
#   REQUESTS             - Total requests per run (default: 2000)
#   MAX_CONCURRENT       - Max concurrent requests (default: 50)
#   RUNS                 - Number of benchmark runs (default: 3)
#   FAKE_RESPONSE_DELAY_MS - Simulated provider latency (default: 50)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v uv &>/dev/null && [ -f "$SCRIPT_DIR/../../pyproject.toml" ]; then
    RUN_PREFIX="uv run --extra gateway"
else
    RUN_PREFIX=""
fi

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

GATEWAY_WORKERS="${GATEWAY_WORKERS:-4}"
REQUESTS="${REQUESTS:-2000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
RUNS="${RUNS:-3}"
FAKE_RESPONSE_DELAY_MS="${FAKE_RESPONSE_DELAY_MS:-50}"
FAKE_SERVER_PORT=9000
MLFLOW_PORT=5000
LITELLM_PORT=4000

PIDS=()

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

echo "=========================================="
echo " MLflow Gateway vs LiteLLM - Head to Head"
echo "=========================================="
echo "Workers:        $GATEWAY_WORKERS"
echo "Requests/run:   $REQUESTS"
echo "Concurrency:    $MAX_CONCURRENT"
echo "Runs:           $RUNS"
echo "Fake delay:     ${FAKE_RESPONSE_DELAY_MS}ms"
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

# 2. Start MLflow Gateway
echo "=== Starting MLflow Gateway (port $MLFLOW_PORT, $GATEWAY_WORKERS workers) ==="
BENCHMARK_GATEWAY_CONFIG="$SCRIPT_DIR/gateway_config.yaml" \
PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" \
    $RUN_PREFIX gunicorn "run_gateway_with_middleware:create_app()" \
    -k uvicorn.workers.UvicornWorker \
    -w "$GATEWAY_WORKERS" \
    -b "0.0.0.0:$MLFLOW_PORT" \
    --log-level warning \
    > /dev/null 2>&1 &
PIDS+=($!)
wait_for_port "$MLFLOW_PORT" "MLflow Gateway"

# 3. Start LiteLLM proxy
echo "=== Starting LiteLLM proxy (port $LITELLM_PORT, $GATEWAY_WORKERS workers) ==="
$RUN_PREFIX litellm \
    --config "$SCRIPT_DIR/litellm_config.yaml" \
    --port "$LITELLM_PORT" \
    --num_workers "$GATEWAY_WORKERS" \
    > /dev/null 2>&1 &
PIDS+=($!)
wait_for_port "$LITELLM_PORT" "LiteLLM proxy" "/health/liveliness"

# 4. Quick sanity check
echo ""
echo "=== Sanity check ==="
echo -n "MLflow Gateway: "
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$MLFLOW_PORT/gateway/benchmark-chat/invocations \
    -X POST -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"test"}]}'
echo ""
echo -n "LiteLLM:        "
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$LITELLM_PORT/chat/completions \
    -X POST -H "Content-Type: application/json" -H "Authorization: Bearer sk-1234" \
    -d '{"model":"benchmark-chat","messages":[{"role":"user","content":"test"}]}'
echo ""

# 5. Run benchmark
echo ""
echo "=== Running head-to-head benchmark ==="
$RUN_PREFIX python benchmark_compare.py \
    --target both \
    --requests "$REQUESTS" \
    --max-concurrent "$MAX_CONCURRENT" \
    --runs "$RUNS" \
    --mlflow-url "http://127.0.0.1:$MLFLOW_PORT/gateway/benchmark-chat/invocations" \
    --litellm-url "http://127.0.0.1:$LITELLM_PORT/chat/completions"
