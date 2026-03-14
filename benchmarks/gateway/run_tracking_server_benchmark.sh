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
# Configuration via environment variables:
#   TRACKING_SERVER_WORKERS - Workers for mlflow server (default: 4)
#   REQUESTS               - Total requests per run (default: 2000)
#   MAX_CONCURRENT         - Max concurrent requests (default: 50)
#   RUNS                   - Number of benchmark runs (default: 3)
#   FAKE_RESPONSE_DELAY_MS - Simulated provider latency (default: 50)
#   USAGE_TRACKING         - Enable usage tracking/tracing (default: true, set "false" to disable)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TRACKING_SERVER_WORKERS="${TRACKING_SERVER_WORKERS:-4}"
USAGE_TRACKING="${USAGE_TRACKING:-true}"

echo "=============================================="
echo " MLflow Tracking Server Gateway Benchmark"
echo "=============================================="
echo "Workers:        $TRACKING_SERVER_WORKERS"
echo "Requests/run:   $REQUESTS"
echo "Concurrency:    $MAX_CONCURRENT"
echo "Runs:           $RUNS"
echo "Fake delay:     ${FAKE_RESPONSE_DELAY_MS}ms"
echo "Usage tracking: $USAGE_TRACKING"
echo "DB:             SQLite (${TMPDIR_BENCH}/mlflow.db)"
echo ""

start_fake_server
start_mlflow_server "$TRACKING_SERVER_WORKERS" "sqlite:///${TMPDIR_BENCH}/mlflow.db"
setup_gateway_endpoint

# Sanity check
echo ""
echo "=== Sanity check ==="
INVOKE_URL="http://127.0.0.1:$MLFLOW_PORT/gateway/$ENDPOINT_NAME/mlflow/invocations"
sanity_check_mlflow "$INVOKE_URL"

run_benchmark mlflow "$INVOKE_URL"
