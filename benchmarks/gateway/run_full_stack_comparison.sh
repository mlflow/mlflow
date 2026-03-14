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

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

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
USAGE_TRACKING="${USAGE_TRACKING:-true}"
POSTGRES_CONTAINER="benchmark-postgres"
DOCKER_CLEANUP_CMD="docker rm -f $POSTGRES_CONTAINER > /dev/null 2>&1"

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

# Start PostgreSQL via Docker
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
docker exec "$POSTGRES_CONTAINER" psql -U postgres -c "CREATE DATABASE mlflow;" > /dev/null 2>&1
echo "PostgreSQL ready (databases: mlflow, litellm)."

# Ensure PostgreSQL drivers are available
echo ""
echo "=== Checking dependencies ==="
$RUN_PREFIX pip install psycopg2-binary > /dev/null 2>&1 || true

start_fake_server

echo ""
start_mlflow_server "$WORKERS" "postgresql://postgres:benchmarkpass@127.0.0.1:5432/mlflow"
setup_gateway_endpoint

# Set up Prisma client for LiteLLM DB support
echo ""
echo "=== Setting up Prisma client for LiteLLM ==="
$RUN_PREFIX pip install prisma > /dev/null 2>&1 || true
$RUN_PREFIX prisma generate --schema "$($RUN_PREFIX python -c "import litellm; print(litellm.__path__[0])")/proxy/schema.prisma" > /dev/null 2>&1 || true

# Start LiteLLM proxy with PostgreSQL
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

# Sanity checks
echo ""
echo "=== Sanity check ==="
MLFLOW_INVOKE_URL="http://127.0.0.1:$MLFLOW_PORT/gateway/$ENDPOINT_NAME/mlflow/invocations"
LITELLM_URL="http://127.0.0.1:$LITELLM_PORT/chat/completions"
sanity_check_mlflow "$MLFLOW_INVOKE_URL"
sanity_check_litellm "$LITELLM_URL"

run_benchmark both "$MLFLOW_INVOKE_URL" "$LITELLM_URL"
