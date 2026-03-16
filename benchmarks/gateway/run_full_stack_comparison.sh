#!/usr/bin/env bash
#
# Full-stack comparison: MLflow (PostgreSQL) vs LiteLLM (PostgreSQL) vs Portkey (stateless)
#
# Unlike run_comparison.sh which tests the AI Gateway without usage tracking,
# this script tests proxies with database-backed config and spend/usage
# tracking enabled — the production code path.
#
# - MLflow: tracking server with PostgreSQL + gateway endpoint + usage tracking
# - LiteLLM: proxy with PostgreSQL + spend tracking (auto-enabled with DB)
# - Portkey: stateless proxy via npx (no DB mode in OSS — routing via per-request headers)
#
# Prerequisites:
#   - Docker (for PostgreSQL container)
#   - litellm[proxy] installed
#   - Node.js / npx (optional, for Portkey — skipped if not found)
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

# Check for npx (needed for Portkey)
HAS_NPX=false
if command -v npx &>/dev/null; then
    HAS_NPX=true
fi

if [ "$HAS_NPX" = "true" ]; then
    BENCH_TARGET="all"
else
    echo "WARNING: npx not found — skipping Portkey AI Gateway (install Node.js to include it)"
    BENCH_TARGET="both"
fi

echo "======================================================"
echo " Full-Stack: MLflow (PostgreSQL) vs LiteLLM (PostgreSQL) vs Portkey"
echo "======================================================"
echo "Workers:           $WORKERS"
echo "Requests/run:      $REQUESTS"
echo "Concurrency:       $MAX_CONCURRENT"
echo "Runs:              $RUNS"
echo "Fake delay:        ${FAKE_RESPONSE_DELAY_MS}ms"
echo "Usage tracking:    $USAGE_TRACKING"
echo "MLflow DB:         PostgreSQL (Docker container)"
echo "LiteLLM DB:        PostgreSQL (Docker container)"
echo "Portkey:           $([ "$HAS_NPX" = "true" ] && echo "enabled (stateless, no DB in OSS)" || echo "skipped (npx not found)")"
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

# Start LiteLLM proxy WITHOUT DB (no tracking baseline)
echo ""
echo "=== Starting LiteLLM proxy without tracking (port $LITELLM_NOTRACK_PORT, $WORKERS workers, no DB) ==="
$RUN_PREFIX litellm \
    --config "litellm_config.yaml" \
    --port "$LITELLM_NOTRACK_PORT" \
    --num_workers "$WORKERS" \
    > /dev/null 2>&1 &
PIDS+=($!)
wait_for_port "$LITELLM_NOTRACK_PORT" "LiteLLM proxy (no tracking)" "/health/liveliness"

# Start LiteLLM proxy with DB + payload logging (store_prompts_in_spend_logs)
echo ""
echo "=== Starting LiteLLM proxy with payload logging (port $LITELLM_PAYLOAD_PORT, $WORKERS workers, PostgreSQL) ==="
DATABASE_URL="postgresql://postgres:benchmarkpass@127.0.0.1:5432/litellm" \
LITELLM_SALT_KEY="sk-bench-salt-key-1234" \
    $RUN_PREFIX litellm \
    --config "litellm_config_db_payload.yaml" \
    --port "$LITELLM_PAYLOAD_PORT" \
    --num_workers "$WORKERS" \
    > /dev/null 2>&1 &
PIDS+=($!)
wait_for_port "$LITELLM_PAYLOAD_PORT" "LiteLLM proxy (payload logging)" "/health/liveliness"

# Start Portkey AI Gateway (if npx available)
PORTKEY_URL=""
if [ "$HAS_NPX" = "true" ]; then
    start_portkey_gateway
    PORTKEY_URL="http://127.0.0.1:$PORTKEY_PORT/v1/chat/completions"
fi

# Sanity checks
echo ""
echo "=== Sanity check ==="
MLFLOW_INVOKE_URL="http://127.0.0.1:$MLFLOW_PORT/gateway/$ENDPOINT_NAME/mlflow/invocations"
LITELLM_URL="http://127.0.0.1:$LITELLM_PORT/chat/completions"
LITELLM_NOTRACK_URL="http://127.0.0.1:$LITELLM_NOTRACK_PORT/chat/completions"
LITELLM_PAYLOAD_URL="http://127.0.0.1:$LITELLM_PAYLOAD_PORT/chat/completions"
sanity_check_mlflow "$MLFLOW_INVOKE_URL"
sanity_check_litellm "$LITELLM_URL"
echo -n "LiteLLM (no tracking): "
http_code=$(curl -s -o /dev/null -w "%{http_code}" "$LITELLM_NOTRACK_URL" \
    -X POST -H "Content-Type: application/json" -H "Authorization: Bearer sk-1234" \
    -d '{"model":"benchmark-chat","messages":[{"role":"user","content":"test"}]}')
echo "$http_code"
if [ "$http_code" != "200" ]; then
    echo "ERROR: LiteLLM (no tracking) sanity check failed (expected 200, got $http_code)"
    exit 1
fi
echo -n "LiteLLM (payload logging): "
http_code=$(curl -s -o /dev/null -w "%{http_code}" "$LITELLM_PAYLOAD_URL" \
    -X POST -H "Content-Type: application/json" -H "Authorization: Bearer sk-1234" \
    -d '{"model":"benchmark-chat","messages":[{"role":"user","content":"test"}]}')
echo "$http_code"
if [ "$http_code" != "200" ]; then
    echo "ERROR: LiteLLM (payload logging) sanity check failed (expected 200, got $http_code)"
    exit 1
fi
if [ -n "$PORTKEY_URL" ]; then
    sanity_check_portkey "$PORTKEY_URL"
fi

run_benchmark "$BENCH_TARGET" "$MLFLOW_INVOKE_URL" "$LITELLM_URL" "$PORTKEY_URL" "$LITELLM_NOTRACK_URL" "$LITELLM_PAYLOAD_URL"
