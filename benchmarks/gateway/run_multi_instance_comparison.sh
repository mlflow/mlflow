#!/usr/bin/env bash
#
# Multi-instance comparison: MLflow vs LiteLLM vs Portkey behind nginx load balancer.
#
# Matches LiteLLM's benchmark methodology:
#   https://docs.litellm.ai/docs/benchmarks
#
# - Multiple gateway instances on separate ports, load-balanced by nginx
# - High concurrency (default: 1000 simulated users)
# - High request count (default: 10000 requests per run)
#
# Each "instance" is a separate gunicorn process on its own port with WORKERS_PER_INSTANCE
# workers. This matches LiteLLM's setup of N separate 4-vCPU instances.
#
# Prerequisites:
#   - Docker (for PostgreSQL and nginx containers)
#   - litellm[proxy] installed
#   - Node.js / npx (optional, for Portkey — skipped if not found)
#
# Configuration via environment variables:
#   INSTANCES              - Number of gateway instances (default: 4)
#   WORKERS_PER_INSTANCE   - Workers per instance (default: 4)
#   REQUESTS               - Total requests per run (default: 10000)
#   MAX_CONCURRENT         - Max concurrent requests (default: 1000)
#   RUNS                   - Number of benchmark runs (default: 3)
#   FAKE_RESPONSE_DELAY_MS - Simulated provider latency (default: 50)
#   USAGE_TRACKING         - Enable MLflow usage tracking/tracing (default: true)

# Set higher defaults before common.sh applies its own (2000/50).
# User-provided env vars take precedence since we use :- syntax.
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

# Port ranges for multi-instance (avoid clashing with common.sh constants)
MLFLOW_BASE_PORT=5100
LITELLM_BASE_PORT=4100
# nginx load balancer ports
MLFLOW_LB_PORT=5000
LITELLM_LB_PORT=4000

# Check for npx (needed for Portkey)
HAS_NPX=false
if command -v npx &>/dev/null; then
    HAS_NPX=true
fi

TOTAL_WORKERS=$((INSTANCES * WORKERS_PER_INSTANCE))

echo "======================================================"
echo " Multi-Instance: MLflow vs LiteLLM vs Portkey (nginx LB)"
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
echo "LiteLLM ports:     ${LITELLM_BASE_PORT}-$((LITELLM_BASE_PORT + INSTANCES - 1)) -> LB :${LITELLM_LB_PORT}"
echo "Portkey:           $([ "$HAS_NPX" = "true" ] && echo "enabled (single instance, port $PORTKEY_PORT)" || echo "skipped (npx not found)")"
echo ""

# ── PostgreSQL ───────────────────────────────────────────────
echo "=== Starting PostgreSQL (Docker) ==="
docker rm -f "$POSTGRES_CONTAINER" > /dev/null 2>&1 || true
docker run --rm -d -p 5432:5432 \
    -e POSTGRES_PASSWORD=benchmarkpass -e POSTGRES_DB=litellm \
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
docker exec "$POSTGRES_CONTAINER" psql -U postgres -c "CREATE DATABASE mlflow;" > /dev/null 2>&1
echo "PostgreSQL ready (databases: mlflow, litellm)."

# ── Dependencies ─────────────────────────────────────────────
echo ""
echo "=== Checking dependencies ==="
$RUN_PREFIX pip install psycopg2-binary > /dev/null 2>&1 || true

# ── Fake OpenAI server (more workers to handle multi-instance load) ──
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
generate_nginx_conf() {
    local name=$1
    local lb_port=$2
    local base_port=$3
    local n_instances=$4

    local conf="upstream ${name}_backends {\n"
    for i in $(seq 0 $((n_instances - 1))); do
        # host.docker.internal resolves to the host machine from inside Docker
        conf+="    server host.docker.internal:$((base_port + i));\n"
    done
    # Keep persistent connections to upstream backends to avoid ephemeral port exhaustion
    conf+="    keepalive 128;\n"
    conf+="}\n\n"
    conf+="server {\n"
    conf+="    listen ${lb_port};\n"
    conf+="    location / {\n"
    conf+="        proxy_pass http://${name}_backends;\n"
    conf+="        proxy_http_version 1.1;\n"
    conf+="        proxy_set_header Connection \"\";\n"
    conf+="        proxy_set_header Host \$host;\n"
    conf+="        proxy_set_header X-Real-IP \$remote_addr;\n"
    conf+="    }\n"
    conf+="}\n"
    echo -e "$conf"
}

NGINX_CONF_DIR="$TMPDIR_BENCH/nginx/conf.d"
mkdir -p "$NGINX_CONF_DIR"

# MLflow upstream
generate_nginx_conf "mlflow" "$MLFLOW_LB_PORT" "$MLFLOW_BASE_PORT" "$INSTANCES" \
    > "$NGINX_CONF_DIR/mlflow.conf"

# LiteLLM upstream
generate_nginx_conf "litellm" "$LITELLM_LB_PORT" "$LITELLM_BASE_PORT" "$INSTANCES" \
    > "$NGINX_CONF_DIR/litellm.conf"

# Main nginx.conf (in parent dir, separate from conf.d)
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
    -p "${LITELLM_LB_PORT}:${LITELLM_LB_PORT}" \
    nginx:alpine \
    > /dev/null

# ── Helpers for sequential start/stop ────────────────────────
GATEWAY_PIDS=()

kill_gateway_pids() {
    for pid in "${GATEWAY_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "${GATEWAY_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    GATEWAY_PIDS=()
}

# ── Pre-setup: Prisma client for LiteLLM ────────────────────
echo ""
echo "=== Setting up Prisma client for LiteLLM ==="
$RUN_PREFIX pip install prisma > /dev/null 2>&1 || true
$RUN_PREFIX prisma generate --schema "$($RUN_PREFIX python -c "import litellm; print(litellm.__path__[0])")/proxy/schema.prisma" > /dev/null 2>&1 || true

# ── Benchmark each gateway sequentially ──────────────────────
# Each gateway gets full machine resources — no competing processes.
MLFLOW_BACKEND_URI="postgresql://postgres:benchmarkpass@127.0.0.1:5432/mlflow"
MLFLOW_INVOKE_URL="http://127.0.0.1:$MLFLOW_LB_PORT/gateway/$ENDPOINT_NAME/mlflow/invocations"
LITELLM_URL="http://127.0.0.1:$LITELLM_LB_PORT/chat/completions"
PORTKEY_URL="http://127.0.0.1:$PORTKEY_PORT/v1/chat/completions"
RESULTS_FILE="$TMPDIR_BENCH/results.json"

run_bench() {
    local target=$1; shift
    local args=(
        --target "$target"
        --requests "$REQUESTS"
        --max-concurrent "$MAX_CONCURRENT"
        --runs "$RUNS"
        --save-results "$RESULTS_FILE"
        "$@"
    )
    echo ""
    echo "=== Running benchmark ($REQUESTS requests × $RUNS runs, $MAX_CONCURRENT concurrency) ==="
    $RUN_PREFIX python benchmark_compare.py "${args[@]}"
}

# ════════════════════════════════════════════════════════════
# 1. MLflow
# ════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Phase 1: MLflow AI Gateway                         ║"
echo "╚══════════════════════════════════════════════════════╝"

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
GATEWAY_PIDS+=($!)
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
    GATEWAY_PIDS+=($!)
    PIDS+=($!)
done
for i in $(seq 1 $((INSTANCES - 1))); do
    port=$((MLFLOW_BASE_PORT + i))
    wait_for_port "$port" "MLflow instance $((i + 1))" "/health" "$TMPDIR_BENCH/mlflow_${port}.log"
done

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

docker exec "$NGINX_CONTAINER" nginx -s reload > /dev/null 2>&1
sleep 1
sanity_check_mlflow "$MLFLOW_INVOKE_URL"
run_bench "mlflow" --mlflow-url "$MLFLOW_INVOKE_URL"

echo ""
echo "=== Stopping MLflow instances ==="
kill_gateway_pids

# ════════════════════════════════════════════════════════════
# 2. LiteLLM
# ════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Phase 2: LiteLLM Proxy                             ║"
echo "╚══════════════════════════════════════════════════════╝"

echo "=== Starting $INSTANCES LiteLLM instances ($WORKERS_PER_INSTANCE workers each) ==="
echo "  Starting LiteLLM instance 1 on port $LITELLM_BASE_PORT (initializes DB schema)..."
LITELLM_LOG_1="$TMPDIR_BENCH/litellm_${LITELLM_BASE_PORT}.log"
DATABASE_URL="postgresql://postgres:benchmarkpass@127.0.0.1:5432/litellm" \
LITELLM_SALT_KEY="sk-bench-salt-key-1234" \
    $RUN_PREFIX litellm \
    --config "litellm_config_db.yaml" \
    --port "$LITELLM_BASE_PORT" \
    --num_workers "$WORKERS_PER_INSTANCE" \
    > "$LITELLM_LOG_1" 2>&1 &
GATEWAY_PIDS+=($!)
PIDS+=($!)
wait_for_port "$LITELLM_BASE_PORT" "LiteLLM instance 1" "/health/liveliness" "$LITELLM_LOG_1"

for i in $(seq 1 $((INSTANCES - 1))); do
    port=$((LITELLM_BASE_PORT + i))
    echo "  Starting LiteLLM instance $((i + 1)) on port $port..."
    log_file="$TMPDIR_BENCH/litellm_${port}.log"
    DATABASE_URL="postgresql://postgres:benchmarkpass@127.0.0.1:5432/litellm" \
    LITELLM_SALT_KEY="sk-bench-salt-key-1234" \
        $RUN_PREFIX litellm \
        --config "litellm_config_db.yaml" \
        --port "$port" \
        --num_workers "$WORKERS_PER_INSTANCE" \
        > "$log_file" 2>&1 &
    GATEWAY_PIDS+=($!)
    PIDS+=($!)
done
for i in $(seq 1 $((INSTANCES - 1))); do
    port=$((LITELLM_BASE_PORT + i))
    wait_for_port "$port" "LiteLLM instance $((i + 1))" "/health/liveliness" "$TMPDIR_BENCH/litellm_${port}.log"
done

docker exec "$NGINX_CONTAINER" nginx -s reload > /dev/null 2>&1
sleep 1
sanity_check_litellm "$LITELLM_URL"
run_bench "litellm" --litellm-url "$LITELLM_URL"

echo ""
echo "=== Stopping LiteLLM instances ==="
kill_gateway_pids

# ════════════════════════════════════════════════════════════
# 3. Portkey
# ════════════════════════════════════════════════════════════
if [ "$HAS_NPX" = "true" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  Phase 3: Portkey AI Gateway                        ║"
    echo "╚══════════════════════════════════════════════════════╝"

    start_portkey_gateway
    sanity_check_portkey "$PORTKEY_URL"
    run_bench "portkey" --portkey-url "$PORTKEY_URL"
fi

# ── Combined comparison table ────────────────────────────────
$RUN_PREFIX python benchmark_compare.py --target mlflow --load-results "$RESULTS_FILE"

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
echo "  Portkey:          single instance (stateless, no DB)"
echo ""
echo "Note: Portkey runs as a single Node.js process (no multi-instance)."
echo "This matches LiteLLM's published benchmark methodology."
