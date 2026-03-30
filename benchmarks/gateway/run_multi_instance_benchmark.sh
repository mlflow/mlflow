#!/usr/bin/env bash
#
# Multi-instance MLflow AI Gateway benchmark behind nginx load balancer.
#
# Runs N separate MLflow server instances on different ports, load-balanced
# by nginx (Docker). Useful for testing gateway performance under sustained
# load with horizontal scaling.
#
# Prerequisites:
#   - Docker (for PostgreSQL and nginx containers)
#
# Configuration via environment variables:
#   INSTANCES              - Number of MLflow instances (default: 4)
#   WORKERS_PER_INSTANCE   - Workers per instance (default: 4)
#   REQUESTS               - Total requests per run (default: 10000)
#   MAX_CONCURRENT         - Max concurrent requests (default: 200)
#   RUNS                   - Number of benchmark runs (default: 3)
#   FAKE_RESPONSE_DELAY_MS - Simulated provider latency (default: 50)
#   USAGE_TRACKING         - Enable MLflow usage tracking/tracing (default: true)

# Set higher defaults before common.sh applies its own (2000/50).
REQUESTS="${REQUESTS:-10000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-200}"

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

INSTANCES="${INSTANCES:-4}"
WORKERS_PER_INSTANCE="${WORKERS_PER_INSTANCE:-4}"
USAGE_TRACKING="${USAGE_TRACKING:-true}"

POSTGRES_CONTAINER="benchmark-postgres"
NGINX_CONTAINER="benchmark-nginx"
DOCKER_CLEANUP_CMD="docker rm -f $POSTGRES_CONTAINER $NGINX_CONTAINER > /dev/null 2>&1"

MLFLOW_BASE_PORT=5100
MLFLOW_LB_PORT=5000
TOTAL_WORKERS=$((INSTANCES * WORKERS_PER_INSTANCE))

echo "======================================================"
echo " Multi-Instance MLflow AI Gateway Benchmark (nginx LB)"
echo "======================================================"
echo "Instances:         $INSTANCES"
echo "Workers/instance:  $WORKERS_PER_INSTANCE"
echo "Total workers:     $TOTAL_WORKERS"
echo "Requests/run:      $REQUESTS"
echo "Concurrency:       $MAX_CONCURRENT"
echo "Runs:              $RUNS"
echo "Fake delay:        ${FAKE_RESPONSE_DELAY_MS}ms"
echo "Usage tracking:    $USAGE_TRACKING"
echo "MLflow ports:      ${MLFLOW_BASE_PORT}-$((MLFLOW_BASE_PORT + INSTANCES - 1)) -> LB :${MLFLOW_LB_PORT}"
echo ""

# ── PostgreSQL ───────────────────────────────────────────────
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

# ── Dependencies ─────────────────────────────────────────────
echo ""
echo "=== Checking dependencies ==="
$RUN_PREFIX pip install psycopg2-binary > /dev/null 2>&1 || true

# ── Fake OpenAI server (more workers for multi-instance load) ──
echo "=== Starting fake OpenAI server (16 workers) ==="
FAKE_SERVER_LOG="$TMPDIR_BENCH/fake_openai_server.log"
FAKE_RESPONSE_DELAY_MS="$FAKE_RESPONSE_DELAY_MS" \
    $RUN_PREFIX gunicorn fake_openai_server:app \
    -k uvicorn.workers.UvicornWorker \
    -w 16 \
    -b "0.0.0.0:$FAKE_SERVER_PORT" \
    --log-level warning \
    > "$FAKE_SERVER_LOG" 2>&1 &
PIDS+=($!)
wait_for_port "$FAKE_SERVER_PORT" "fake OpenAI server" "/health" "$FAKE_SERVER_LOG"

# ── Generate nginx config ────────────────────────────────────
NGINX_CONF_DIR="$TMPDIR_BENCH/nginx/conf.d"
mkdir -p "$NGINX_CONF_DIR"

# MLflow upstream config
{
    echo "upstream mlflow_backends {"
    for i in $(seq 0 $((INSTANCES - 1))); do
        echo "    server host.docker.internal:$((MLFLOW_BASE_PORT + i));"
    done
    echo "    keepalive 128;"
    echo "}"
    echo ""
    echo "server {"
    echo "    listen ${MLFLOW_LB_PORT};"
    echo "    location / {"
    echo "        proxy_pass http://mlflow_backends;"
    echo "        proxy_http_version 1.1;"
    echo "        proxy_set_header Connection \"\";"
    echo "        proxy_set_header Host \$host;"
    echo "        proxy_set_header X-Real-IP \$remote_addr;"
    echo "    }"
    echo "}"
} > "$NGINX_CONF_DIR/mlflow.conf"

# Main nginx.conf
cat > "$TMPDIR_BENCH/nginx/nginx.conf" <<'NGINX_MAIN'
worker_processes auto;
events {
    worker_connections 4096;
}
http {
    keepalive_timeout 65;
    keepalive_requests 10000;
    include /etc/nginx/conf.d/*.conf;
}
NGINX_MAIN

# ── Start nginx ──────────────────────────────────────────────
echo ""
echo "=== Starting nginx load balancer (Docker) ==="
docker rm -f "$NGINX_CONTAINER" > /dev/null 2>&1 || true
docker run --rm -d \
    --name "$NGINX_CONTAINER" \
    --add-host=host.docker.internal:host-gateway \
    -v "$TMPDIR_BENCH/nginx/nginx.conf:/etc/nginx/nginx.conf:ro" \
    -v "$NGINX_CONF_DIR:/etc/nginx/conf.d:ro" \
    -p "${MLFLOW_LB_PORT}:${MLFLOW_LB_PORT}" \
    nginx:alpine \
    > /dev/null

# ── Start MLflow instances ───────────────────────────────────
# Start instance 1 first to initialize the DB schema.
# Starting all simultaneously causes CREATE TABLE race conditions.
MLFLOW_BACKEND_URI="postgresql://postgres:benchmarkpass@127.0.0.1:5432/mlflow"

echo ""
echo "=== Starting $INSTANCES MLflow instances ($WORKERS_PER_INSTANCE workers each) ==="
echo "  Starting MLflow instance 1 on port $MLFLOW_BASE_PORT (initializes DB schema)..."
MLFLOW_LOG_1="$TMPDIR_BENCH/mlflow_${MLFLOW_BASE_PORT}.log"
$RUN_PREFIX mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_URI" \
    --host 0.0.0.0 \
    --port "$MLFLOW_BASE_PORT" \
    --workers "$WORKERS_PER_INSTANCE" \
    --disable-security-middleware \
    > "$MLFLOW_LOG_1" 2>&1 &
PIDS+=($!)
wait_for_port "$MLFLOW_BASE_PORT" "MLflow instance 1" "/health" "$MLFLOW_LOG_1"

for i in $(seq 1 $((INSTANCES - 1))); do
    port=$((MLFLOW_BASE_PORT + i))
    echo "  Starting MLflow instance $((i + 1)) on port $port..."
    log_file="$TMPDIR_BENCH/mlflow_${port}.log"
    $RUN_PREFIX mlflow server \
        --backend-store-uri "$MLFLOW_BACKEND_URI" \
        --host 0.0.0.0 \
        --port "$port" \
        --workers "$WORKERS_PER_INSTANCE" \
        --disable-security-middleware \
        > "$log_file" 2>&1 &
    PIDS+=($!)
done
for i in $(seq 1 $((INSTANCES - 1))); do
    port=$((MLFLOW_BASE_PORT + i))
    wait_for_port "$port" "MLflow instance $((i + 1))" "/health" "$TMPDIR_BENCH/mlflow_${port}.log"
done

# ── Setup gateway endpoint (via instance 1, DB is shared) ────
echo ""
echo "=== Setting up gateway endpoint ==="
SETUP_ARGS=(
    --tracking-uri "http://127.0.0.1:$MLFLOW_BASE_PORT"
    --fake-server-url "http://127.0.0.1:$FAKE_SERVER_PORT/v1"
    --endpoint-name "$ENDPOINT_NAME"
)
if [ "$USAGE_TRACKING" = "false" ]; then
    SETUP_ARGS+=(--no-usage-tracking)
fi
$RUN_PREFIX python setup_tracking_server.py "${SETUP_ARGS[@]}"

# ── Reload nginx and run benchmark ───────────────────────────
docker exec "$NGINX_CONTAINER" nginx -s reload > /dev/null 2>&1
sleep 1

echo ""
echo "=== Sanity check (via nginx LB) ==="
MLFLOW_INVOKE_URL="http://127.0.0.1:$MLFLOW_LB_PORT/gateway/$ENDPOINT_NAME/mlflow/invocations"
sanity_check_mlflow "$MLFLOW_INVOKE_URL"

run_benchmark "$MLFLOW_INVOKE_URL"

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Configuration summary"
echo "======================================================"
echo "  Instances:        $INSTANCES"
echo "  Workers/instance: $WORKERS_PER_INSTANCE"
echo "  Total workers:    $TOTAL_WORKERS"
echo "  Load balancer:    nginx (Docker, round-robin)"
echo "  Concurrency:      $MAX_CONCURRENT"
echo "  Requests/run:     $REQUESTS"
echo "  Fake delay:       ${FAKE_RESPONSE_DELAY_MS}ms"
echo "  Usage tracking:   $USAGE_TRACKING"
