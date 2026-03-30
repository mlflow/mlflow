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
# Supports both SQLite (default) and PostgreSQL backends. When using PostgreSQL,
# the script can optionally start a Docker container for you.
#
# Configuration via environment variables:
#   TRACKING_SERVER_WORKERS - Workers for mlflow server (default: 4)
#   REQUESTS               - Total requests per run (default: 2000)
#   MAX_CONCURRENT         - Max concurrent requests (default: 50)
#   RUNS                   - Number of benchmark runs (default: 3)
#   FAKE_RESPONSE_DELAY_MS - Simulated provider latency (default: 50)
#   USAGE_TRACKING         - Enable usage tracking/tracing (default: true, set "false" to disable)
#   BACKEND_STORE_URI      - Database URI (default: SQLite in temp dir)
#                            Set to "postgres" to auto-start a Docker PostgreSQL container
#                            Or provide a full URI like "postgresql://user:pass@host:port/db"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TRACKING_SERVER_WORKERS="${TRACKING_SERVER_WORKERS:-4}"
USAGE_TRACKING="${USAGE_TRACKING:-true}"
BACKEND_STORE_URI="${BACKEND_STORE_URI:-sqlite:///${TMPDIR_BENCH}/mlflow.db}"
POSTGRES_CONTAINER=""

# Handle "postgres" shorthand — start a Docker container automatically
if [ "$BACKEND_STORE_URI" = "postgres" ] || [ "$BACKEND_STORE_URI" = "postgresql" ]; then
    if ! command -v docker &>/dev/null; then
        echo "ERROR: Docker is required for PostgreSQL but not found."
        exit 1
    fi
    if ! docker info > /dev/null 2>&1; then
        echo "ERROR: Docker daemon is not running."
        exit 1
    fi

    POSTGRES_CONTAINER="benchmark-postgres"
    DOCKER_CLEANUP_CMD="docker rm -f $POSTGRES_CONTAINER > /dev/null 2>&1"

    echo "=== Starting PostgreSQL (Docker) ==="
    docker rm -f "$POSTGRES_CONTAINER" > /dev/null 2>&1 || true
    docker run --rm -d -p 5432:5432 \
        -e POSTGRES_PASSWORD=benchmarkpass -e POSTGRES_DB=mlflow \
        --name "$POSTGRES_CONTAINER" postgres:16-alpine \
        -c max_connections=500 \
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

    echo ""
    echo "=== Checking dependencies ==="
    $RUN_PREFIX pip install psycopg2-binary > /dev/null 2>&1 || true

    BACKEND_STORE_URI="postgresql://postgres:benchmarkpass@127.0.0.1:5432/mlflow"
fi

# Determine DB display name
case "$BACKEND_STORE_URI" in
    sqlite:*)  DB_DISPLAY="SQLite" ;;
    postgres*) DB_DISPLAY="PostgreSQL" ;;
    *)         DB_DISPLAY="$BACKEND_STORE_URI" ;;
esac

echo "=============================================="
echo " MLflow Tracking Server Gateway Benchmark"
echo "=============================================="
echo "Workers:        $TRACKING_SERVER_WORKERS"
echo "Requests/run:   $REQUESTS"
echo "Concurrency:    $MAX_CONCURRENT"
echo "Runs:           $RUNS"
echo "Fake delay:     ${FAKE_RESPONSE_DELAY_MS}ms"
echo "Usage tracking: $USAGE_TRACKING"
echo "DB:             $DB_DISPLAY"
echo ""

start_fake_server
start_mlflow_server "$TRACKING_SERVER_WORKERS" "$BACKEND_STORE_URI"
setup_gateway_endpoint

# Sanity check
echo ""
echo "=== Sanity check ==="
INVOKE_URL="http://127.0.0.1:$MLFLOW_PORT/gateway/$ENDPOINT_NAME/mlflow/invocations"
sanity_check_mlflow "$INVOKE_URL"

run_benchmark "$INVOKE_URL"
