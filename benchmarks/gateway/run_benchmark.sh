#!/usr/bin/env bash
#
# MLflow AI Gateway Benchmark Orchestration Script
#
# Starts fake OpenAI server, gateway with middleware, resource monitor,
# and runs Locust in headless mode. Saves timestamped results.
#
# Configuration via environment variables:
#   GATEWAY_WORKERS      - Number of gunicorn workers (default: 2)
#   LOCUST_USERS         - Number of concurrent users (default: 1000)
#   LOCUST_SPAWN_RATE    - Users spawned per second (default: 500)
#   LOCUST_RUN_TIME      - Test duration (default: 60s)
#   FAKE_RESPONSE_DELAY_MS - Simulated provider latency in ms (default: 50)
#   FAKE_SERVER_PORT     - Port for fake OpenAI server (default: 9000)
#   GATEWAY_PORT         - Port for the gateway (default: 5000)
#
# Usage:
#   bash run_benchmark.sh
#   LOCUST_USERS=5 LOCUST_RUN_TIME=10s bash run_benchmark.sh  # smoke test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect whether to use uv run or direct commands.
# If "uv" is available and we're in a uv-managed project, prefix commands with "uv run".
if command -v uv &>/dev/null && [ -f "$SCRIPT_DIR/../../pyproject.toml" ]; then
    RUN_PREFIX="uv run --extra gateway"
else
    RUN_PREFIX=""
fi

# macOS: prevent fork-safety crash in Objective-C runtime (affects gunicorn workers)
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Configuration
GATEWAY_WORKERS="${GATEWAY_WORKERS:-2}"
LOCUST_USERS="${LOCUST_USERS:-1000}"
LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-500}"
LOCUST_RUN_TIME="${LOCUST_RUN_TIME:-60s}"
FAKE_RESPONSE_DELAY_MS="${FAKE_RESPONSE_DELAY_MS:-50}"
FAKE_SERVER_PORT="${FAKE_SERVER_PORT:-9000}"
GATEWAY_PORT="${GATEWAY_PORT:-5000}"

# Results directory
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/results/${TIMESTAMP}_w${GATEWAY_WORKERS}_d${FAKE_RESPONSE_DELAY_MS}ms_u${LOCUST_USERS}"
mkdir -p "$RESULTS_DIR"

# PIDs to clean up
PIDS=()

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping PID $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Wait briefly then force-kill stragglers
    sleep 1
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "Cleanup complete."
}

trap cleanup EXIT

wait_for_port() {
    local port=$1
    local name=$2
    local max_wait=30
    local waited=0
    echo "Waiting for $name on port $port..."
    while ! curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
        sleep 0.5
        waited=$((waited + 1))
        if [ "$waited" -ge "$((max_wait * 2))" ]; then
            echo "ERROR: $name failed to start on port $port after ${max_wait}s"
            exit 1
        fi
    done
    echo "$name is ready on port $port"
}

echo "=== MLflow AI Gateway Benchmark ==="
echo "Workers:       $GATEWAY_WORKERS"
echo "Locust users:  $LOCUST_USERS"
echo "Spawn rate:    $LOCUST_SPAWN_RATE"
echo "Run time:      $LOCUST_RUN_TIME"
echo "Fake delay:    ${FAKE_RESPONSE_DELAY_MS}ms"
echo "Results dir:   $RESULTS_DIR"
echo ""

# Save configuration
cat > "$RESULTS_DIR/config.txt" <<EOF
gateway_workers=$GATEWAY_WORKERS
locust_users=$LOCUST_USERS
locust_spawn_rate=$LOCUST_SPAWN_RATE
locust_run_time=$LOCUST_RUN_TIME
fake_response_delay_ms=$FAKE_RESPONSE_DELAY_MS
timestamp=$TIMESTAMP
EOF

# Number of fake server workers (scale with gateway workers to avoid bottleneck)
FAKE_SERVER_WORKERS="${FAKE_SERVER_WORKERS:-$((GATEWAY_WORKERS * 2))}"

# 1. Start fake OpenAI server (multiple workers to handle high concurrency)
echo "=== Starting fake OpenAI server (port $FAKE_SERVER_PORT, delay ${FAKE_RESPONSE_DELAY_MS}ms, $FAKE_SERVER_WORKERS workers) ==="
FAKE_RESPONSE_DELAY_MS="$FAKE_RESPONSE_DELAY_MS" \
    $RUN_PREFIX gunicorn fake_openai_server:app \
    -k uvicorn.workers.UvicornWorker \
    -w "$FAKE_SERVER_WORKERS" \
    -b "0.0.0.0:$FAKE_SERVER_PORT" \
    --log-level warning \
    > "$RESULTS_DIR/fake_server.log" 2>&1 &
PIDS+=($!)
wait_for_port "$FAKE_SERVER_PORT" "fake OpenAI server"

# 2. Start gateway with middleware (single uvicorn worker to avoid gunicorn fork issues)
echo "=== Starting MLflow Gateway (port $GATEWAY_PORT, $GATEWAY_WORKERS workers) ==="
BENCHMARK_GATEWAY_CONFIG="$SCRIPT_DIR/gateway_config.yaml" \
PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" \
    $RUN_PREFIX gunicorn "run_gateway_with_middleware:create_app()" \
    -k uvicorn.workers.UvicornWorker \
    -w "$GATEWAY_WORKERS" \
    -b "0.0.0.0:$GATEWAY_PORT" \
    --log-level warning \
    > "$RESULTS_DIR/gateway.log" 2>&1 &
GATEWAY_PID=$!
PIDS+=($GATEWAY_PID)
wait_for_port "$GATEWAY_PORT" "MLflow Gateway"

# 3. Start resource monitor
echo "=== Starting resource monitor ==="
$RUN_PREFIX python collect_resources.py "$GATEWAY_PID" "$RESULTS_DIR/resources.csv" &
PIDS+=($!)

# 4. Run Locust
echo "=== Running Locust load test ==="
$RUN_PREFIX locust \
    -f locustfile.py \
    --host "http://127.0.0.1:$GATEWAY_PORT" \
    --headless \
    --users "$LOCUST_USERS" \
    --spawn-rate "$LOCUST_SPAWN_RATE" \
    --run-time "$LOCUST_RUN_TIME" \
    --csv "$RESULTS_DIR/locust" \
    --csv-full-history \
    --print-stats \
    2>&1 | tee "$RESULTS_DIR/locust_output.log"

echo ""
echo "=== Benchmark complete ==="
echo "Results saved to: $RESULTS_DIR"
echo ""

# 5. Analyze results
echo "=== Results Summary ==="
$RUN_PREFIX python analyze_results.py "$RESULTS_DIR"
