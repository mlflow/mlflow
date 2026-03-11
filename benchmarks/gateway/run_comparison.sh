#!/usr/bin/env bash
#
# Head-to-head benchmark: MLflow AI Gateway vs LiteLLM proxy
#
# Both proxies hit the same fake OpenAI server on the same machine.
# Uses aiohttp-based benchmark (same methodology as LiteLLM's benchmark_mock.py).
#
# MLflow runs as a tracking server with SQLite (AI Gateway code path).
# LiteLLM runs in YAML-config-only mode (no database).
# Usage tracking is disabled by default to reduce DB overhead on the MLflow side.
#
# Configuration via environment variables:
#   GATEWAY_WORKERS        - Workers for both proxies (default: 4)
#   REQUESTS               - Total requests per run (default: 2000)
#   MAX_CONCURRENT         - Max concurrent requests (default: 50)
#   RUNS                   - Number of benchmark runs (default: 3)
#   FAKE_RESPONSE_DELAY_MS - Simulated provider latency (default: 50)
#   USAGE_TRACKING         - Enable MLflow usage tracking/tracing (default: false)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

GATEWAY_WORKERS="${GATEWAY_WORKERS:-4}"
USAGE_TRACKING="${USAGE_TRACKING:-false}"

echo "=========================================="
echo " MLflow AI Gateway vs LiteLLM - Head to Head"
echo "=========================================="
echo "Workers:        $GATEWAY_WORKERS"
echo "Requests/run:   $REQUESTS"
echo "Concurrency:    $MAX_CONCURRENT"
echo "Runs:           $RUNS"
echo "Fake delay:     ${FAKE_RESPONSE_DELAY_MS}ms"
echo "Usage tracking: $USAGE_TRACKING"
echo "MLflow DB:      SQLite (${TMPDIR_BENCH}/mlflow.db)"
echo ""

start_fake_server
start_mlflow_server "$GATEWAY_WORKERS" "sqlite:///${TMPDIR_BENCH}/mlflow.db"
setup_gateway_endpoint

# Start LiteLLM proxy
echo ""
echo "=== Starting LiteLLM proxy (port $LITELLM_PORT, $GATEWAY_WORKERS workers) ==="
$RUN_PREFIX litellm \
    --config "litellm_config.yaml" \
    --port "$LITELLM_PORT" \
    --num_workers "$GATEWAY_WORKERS" \
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
