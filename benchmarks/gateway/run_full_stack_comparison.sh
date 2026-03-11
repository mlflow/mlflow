#!/usr/bin/env bash
#
# Full-stack comparison: MLflow Tracking Server (PostgreSQL) vs LiteLLM (PostgreSQL)
#
# Unlike run_comparison.sh which tests the AI Gateway without usage tracking,
# this script tests both proxies with database-backed config and spend/usage
# tracking enabled — the production code path.
#
# - MLflow: tracking server with PostgreSQL + gateway endpoint + usage tracking
# - LiteLLM: proxy with PostgreSQL + spend tracking (auto-enabled with DB)
#
# Prerequisites:
#   - Docker (for PostgreSQL container)
#   - litellm[proxy] installed
#
# Configuration via environment variables:
#   WORKERS                - Workers for both proxies (default: 4)
#   REQUESTS               - Total requests per run (default: 2000)
#   MAX_CONCURRENT         - Max concurrent requests (default: 50)
#   RUNS                   - Number of benchmark runs (default: 3)
#   FAKE_RESPONSE_DELAY_MS - Simulated provider latency (default: 50)
#   USAGE_TRACKING         - Enable MLflow usage tracking/tracing (default: true)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v uv &>/dev/null && [ -f "$SCRIPT_DIR/../../pyproject.toml" ]; then
    RUN_PREFIX="uv run --extra gateway"
else
    RUN_PREFIX=""
fi

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Check Docker availability
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is required but not found."
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running."
    echo "Start Docker and try again."
    exit 1
fi

WORKERS="${WORKERS:-4}"
REQUESTS="${REQUESTS:-2000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
RUNS="${RUNS:-3}"
FAKE_RESPONSE_DELAY_MS="${FAKE_RESPONSE_DELAY_MS:-50}"
USAGE_TRACKING="${USAGE_TRACKING:-true}"
FAKE_SERVER_PORT=9000
MLFLOW_PORT=5000
LITELLM_PORT=4000
ENDPOINT_NAME="benchmark-chat"
POSTGRES_CONTAINER="benchmark-postgres"

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
    docker rm -f "$POSTGRES_CONTAINER" > /dev/null 2>&1 || true
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

echo "======================================================"
echo " Full-Stack Comparison: MLflow (PostgreSQL) vs LiteLLM (PostgreSQL)"
echo "======================================================"
echo "Workers:           $WORKERS"
echo "Requests/run:      $REQUESTS"
echo "Concurrency:       $MAX_CONCURRENT"
echo "Runs:              $RUNS"
echo "Fake delay:        ${FAKE_RESPONSE_DELAY_MS}ms"
echo "Usage tracking:    $USAGE_TRACKING"
echo "MLflow DB:         PostgreSQL (Docker container)"
echo "LiteLLM DB:        PostgreSQL (Docker container)"
echo ""

# 1. Start PostgreSQL via Docker
echo "=== Starting PostgreSQL (Docker) ==="
docker rm -f "$POSTGRES_CONTAINER" > /dev/null 2>&1 || true
docker run --rm -d -p 5432:5432 \
    -e POSTGRES_PASSWORD=benchmarkpass -e POSTGRES_DB=litellm \
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
# Create a separate database for MLflow
docker exec "$POSTGRES_CONTAINER" psql -U postgres -c "CREATE DATABASE mlflow;" > /dev/null 2>&1
echo "PostgreSQL ready (databases: mlflow, litellm)."

# 2. Ensure PostgreSQL drivers are available
echo ""
echo "=== Checking dependencies ==="
$RUN_PREFIX pip install psycopg2-binary > /dev/null 2>&1 || true

# 3. Start fake OpenAI server
echo ""
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

# 3. Start MLflow tracking server with PostgreSQL
echo ""
echo "=== Starting MLflow tracking server (port $MLFLOW_PORT, $WORKERS workers, PostgreSQL) ==="
$RUN_PREFIX mlflow server \
    --backend-store-uri "postgresql://postgres:benchmarkpass@127.0.0.1:5432/mlflow" \
    --host 0.0.0.0 \
    --port "$MLFLOW_PORT" \
    --workers "$WORKERS" \
    --disable-security-middleware \
    > /dev/null 2>&1 &
PIDS+=($!)
wait_for_port "$MLFLOW_PORT" "MLflow tracking server" "/health"

# 4. Create MLflow gateway endpoint via REST API
echo ""
echo "=== Setting up MLflow gateway endpoint ==="
SETUP_ARGS=(
    --tracking-uri "http://127.0.0.1:$MLFLOW_PORT"
    --fake-server-url "http://127.0.0.1:$FAKE_SERVER_PORT/v1"
    --endpoint-name "$ENDPOINT_NAME"
)
if [ "$USAGE_TRACKING" = "false" ]; then
    SETUP_ARGS+=(--no-usage-tracking)
fi
$RUN_PREFIX python setup_tracking_server.py "${SETUP_ARGS[@]}"

# 5. Ensure Prisma client is available for LiteLLM DB support
echo ""
echo "=== Setting up Prisma client for LiteLLM ==="
$RUN_PREFIX pip install prisma > /dev/null 2>&1 || true
$RUN_PREFIX prisma generate --schema "$($RUN_PREFIX python -c "import litellm; print(litellm.__path__[0])")/proxy/schema.prisma" > /dev/null 2>&1 || true

# 6. Start LiteLLM proxy with PostgreSQL
echo ""
echo "=== Starting LiteLLM proxy (port $LITELLM_PORT, $WORKERS workers, PostgreSQL) ==="
DATABASE_URL="postgresql://postgres:benchmarkpass@127.0.0.1:5432/litellm" \
LITELLM_SALT_KEY="sk-bench-salt-key-1234" \
    $RUN_PREFIX litellm \
    --config "litellm_config_db.yaml" \
    --port "$LITELLM_PORT" \
    --num_workers "$WORKERS" \
    > /dev/null 2>&1 &
PIDS+=($!)
wait_for_port "$LITELLM_PORT" "LiteLLM proxy" "/health/liveliness"

# 6. Sanity checks
echo ""
echo "=== Sanity check ==="
MLFLOW_INVOKE_URL="http://127.0.0.1:$MLFLOW_PORT/gateway/$ENDPOINT_NAME/mlflow/invocations"
LITELLM_URL="http://127.0.0.1:$LITELLM_PORT/chat/completions"

echo -n "MLflow tracking server gateway: "
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$MLFLOW_INVOKE_URL" \
    -X POST -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"test"}]}')
echo "$HTTP_CODE"
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: MLflow sanity check failed (expected 200, got $HTTP_CODE)"
    curl -s "$MLFLOW_INVOKE_URL" \
        -X POST -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"test"}]}'
    echo ""
    exit 1
fi

echo -n "LiteLLM (PostgreSQL):           "
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$LITELLM_URL" \
    -X POST -H "Content-Type: application/json" -H "Authorization: Bearer sk-1234" \
    -d '{"model":"benchmark-chat","messages":[{"role":"user","content":"test"}]}')
echo "$HTTP_CODE"
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: LiteLLM sanity check failed (expected 200, got $HTTP_CODE)"
    curl -s "$LITELLM_URL" \
        -X POST -H "Content-Type: application/json" -H "Authorization: Bearer sk-1234" \
        -d '{"model":"benchmark-chat","messages":[{"role":"user","content":"test"}]}'
    echo ""
    exit 1
fi

# 7. Run benchmark
echo ""
echo "=== Running full-stack comparison benchmark ==="
$RUN_PREFIX python benchmark_compare.py \
    --target both \
    --requests "$REQUESTS" \
    --max-concurrent "$MAX_CONCURRENT" \
    --runs "$RUNS" \
    --mlflow-url "$MLFLOW_INVOKE_URL" \
    --litellm-url "$LITELLM_URL"
