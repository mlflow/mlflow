#!/usr/bin/env bash
#
# Run MLflow gateway benchmark with per-phase profiling enabled.
#
# Instruments each phase of the gateway request lifecycle:
#   parse_body, get_store, get_config, budget_check, create_provider,
#   build_trace_wrapper, provider_call, and total overhead.
#
# Usage:
#   REQUESTS=500 RUNS=1 bash benchmarks/gateway/run_profiled_benchmark.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if ! command -v docker &>/dev/null || ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is required and must be running."
    exit 1
fi

WORKERS="${WORKERS:-1}"
USAGE_TRACKING="${USAGE_TRACKING:-true}"
POSTGRES_CONTAINER="benchmark-postgres"
DOCKER_CLEANUP_CMD="docker rm -f $POSTGRES_CONTAINER > /dev/null 2>&1"
PROFILE_OUTPUT="$TMPDIR_BENCH/gateway_profile.txt"

echo "======================================================"
echo " Profiled MLflow Gateway Benchmark"
echo "======================================================"
echo "Workers:           $WORKERS"
echo "Requests/run:      $REQUESTS"
echo "Concurrency:       $MAX_CONCURRENT"
echo "Runs:              $RUNS"
echo "Fake delay:        ${FAKE_RESPONSE_DELAY_MS}ms"
echo "Usage tracking:    $USAGE_TRACKING"
echo ""

# Start PostgreSQL
echo "=== Starting PostgreSQL (Docker) ==="
docker rm -f "$POSTGRES_CONTAINER" > /dev/null 2>&1 || true
docker run --rm -d -p 5432:5432 \
    -e POSTGRES_PASSWORD=benchmarkpass -e POSTGRES_DB=mlflow \
    --name "$POSTGRES_CONTAINER" postgres:16-alpine \
    > /dev/null
echo "Waiting for PostgreSQL..."
MAX_PG_WAIT=30
PG_WAITED=0
while ! docker exec "$POSTGRES_CONTAINER" pg_isready -U postgres > /dev/null 2>&1; do
    sleep 0.5
    PG_WAITED=$((PG_WAITED + 1))
    if [ "$PG_WAITED" -ge "$((MAX_PG_WAIT * 2))" ]; then
        echo "ERROR: PostgreSQL failed to start after ${MAX_PG_WAIT}s"
        exit 1
    fi
done
echo "PostgreSQL ready."

$RUN_PREFIX pip install psycopg2-binary > /dev/null 2>&1 || true

# Initialize MLflow database schema
echo "Initializing MLflow database..."
$RUN_PREFIX python -c "
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
SqlAlchemyStore('postgresql://postgres:benchmarkpass@127.0.0.1:5432/mlflow', './mlartifacts')
print('Database initialized.')
"

start_fake_server

# Start MLflow with profiling enabled
echo ""
echo "=== Starting MLflow server with profiling (port $MLFLOW_PORT, $WORKERS worker) ==="

MLFLOW_GATEWAY_PROFILE=1 \
MLFLOW_GATEWAY_PROFILE_OUTPUT="$PROFILE_OUTPUT" \
    $RUN_PREFIX mlflow server \
    --backend-store-uri "postgresql://postgres:benchmarkpass@127.0.0.1:5432/mlflow" \
    --host 0.0.0.0 \
    --port "$MLFLOW_PORT" \
    --workers "$WORKERS" \
    --disable-security-middleware \
    > "$TMPDIR_BENCH/mlflow.log" 2>&1 &
MLFLOW_PID=$!
PIDS+=($MLFLOW_PID)
wait_for_port "$MLFLOW_PORT" "MLflow server" "/health"

setup_gateway_endpoint

# Warmup
echo ""
echo "=== Warmup (20 requests) ==="
MLFLOW_INVOKE_URL="http://127.0.0.1:$MLFLOW_PORT/gateway/$ENDPOINT_NAME/mlflow/invocations"
for i in $(seq 1 20); do
    curl -s -o /dev/null "$MLFLOW_INVOKE_URL" \
        -X POST -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"warmup"}]}'
done
echo "Warmup done."

# Run benchmark
echo ""
echo "=== Running benchmark ==="
$RUN_PREFIX python benchmark_compare.py \
    --target mlflow \
    --requests "$REQUESTS" \
    --max-concurrent "$MAX_CONCURRENT" \
    --runs "$RUNS" \
    --mlflow-url "$MLFLOW_INVOKE_URL"

# Kill workers gracefully so atexit handlers fire
echo ""
echo "=== Collecting profiling data ==="
WORKER_PIDS=$(pgrep -P "$MLFLOW_PID" 2>/dev/null || true)
if [ -n "$WORKER_PIDS" ]; then
    for wpid in $WORKER_PIDS; do
        kill -TERM "$wpid" 2>/dev/null || true
    done
    # Wait for atexit to write the file
    sleep 3
fi

if [ -f "$PROFILE_OUTPUT" ]; then
    echo ""
    cat "$PROFILE_OUTPUT"
else
    echo "(Profiling output not found at $PROFILE_OUTPUT)"
    echo "Last 20 lines of server log:"
    tail -20 "$TMPDIR_BENCH/mlflow.log"
fi
