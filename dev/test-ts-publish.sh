#!/usr/bin/env bash
#
# Tests TypeScript SDK publish flow using a local Verdaccio registry.
#
# This script:
#   1. Starts a Verdaccio instance on localhost:4873
#   2. Builds all TypeScript packages
#   3. Publishes packages to Verdaccio in dependency order (core first)
#   4. Installs each published package from Verdaccio into a temp project
#   5. Runs basic smoke tests (require + verify exports)
#   6. Cleans up on exit
#
# Usage:
#   bash dev/test-ts-publish.sh
#
# Requirements: node >= 18, npm, curl

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TS_ROOT="${REPO_ROOT}/libs/typescript"

VERDACCIO_PID=""
WORK_DIR=""

cleanup() {
  echo ""
  echo "Cleaning up..."
  # Remove any .npmrc written during publish/smoke test
  rm -f "${TS_ROOT}/.npmrc"
  if [[ -n "$VERDACCIO_PID" ]] && kill -0 "$VERDACCIO_PID" 2>/dev/null; then
    kill "$VERDACCIO_PID" || true
    wait "$VERDACCIO_PID" 2>/dev/null || true
  fi
  if [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]]; then
    rm -rf "$WORK_DIR"
  fi
}
trap cleanup EXIT

WORK_DIR="$(mktemp -d)"
VERDACCIO_CONFIG="${WORK_DIR}/verdaccio-config.yaml"
REGISTRY_URL="http://localhost:4873"

# --- 1. Write Verdaccio config ---
# @mlflow/* packages must come from local only (no proxy).
# Everything else is proxied from the public registry so transitive deps resolve.
cat > "$VERDACCIO_CONFIG" <<'YAML'
storage: ./storage

uplinks:
  npmjs:
    url: https://registry.npmjs.org/

packages:
  '@mlflow/*':
    access: $all
    publish: $all
  '@*/*':
    access: $all
    publish: $all
    proxy: npmjs
  '**':
    access: $all
    publish: $all
    proxy: npmjs

auth:
  htpasswd:
    file: ./htpasswd
    max_users: 100

listen: 0.0.0.0:4873

log:
  type: stdout
  level: warn
YAML

# --- 2. Start Verdaccio ---
echo "Starting Verdaccio..."
cd "$WORK_DIR"
npx verdaccio@latest --config "$VERDACCIO_CONFIG" &
VERDACCIO_PID=$!

echo "Waiting for Verdaccio (pid ${VERDACCIO_PID})..."
for i in $(seq 1 30); do
  if curl -sf "${REGISTRY_URL}/-/ping" >/dev/null 2>&1; then
    echo "Verdaccio is ready"
    break
  fi
  if ! kill -0 "$VERDACCIO_PID" 2>/dev/null; then
    echo "Error: Verdaccio process exited unexpectedly" >&2
    exit 1
  fi
  if [[ "$i" -eq 30 ]]; then
    echo "Error: Verdaccio did not start within 30 seconds" >&2
    exit 1
  fi
  sleep 1
done

# --- 3. Create npm auth token ---
TOKEN=$(curl -sf -XPUT \
  -H "Content-Type: application/json" \
  -d '{"name":"test","password":"test1234"}' \
  "${REGISTRY_URL}/-/user/org.couchdb.user:test" \
  | node -pe 'JSON.parse(require("fs").readFileSync(0,"utf8")).token')

if [[ -z "$TOKEN" ]]; then
  echo "Error: Failed to obtain auth token from Verdaccio" >&2
  exit 1
fi

echo "Auth token obtained"

# --- 4. Install and build all packages ---
echo ""
echo "=== Installing and building packages ==="
cd "$TS_ROOT"
npm install
npm run build

# --- 5. Publish to Verdaccio in dependency order ---
echo ""
echo "=== Publishing packages to Verdaccio ==="

write_npmrc() {
  local dir="$1"
  cat > "${dir}/.npmrc" <<EOF
registry=${REGISTRY_URL}/
@mlflow:registry=${REGISTRY_URL}/
//localhost:4873/:_authToken=${TOKEN}
EOF
}

publish_pkg() {
  local dir="$1" name="$2"
  echo "Publishing ${name}..."
  cd "$dir"
  npm publish 2>&1
  echo "  Published ${name} successfully"
}

# Write .npmrc at workspace root so npm reads it for all workspace members
write_npmrc "$TS_ROOT"

# Core first (integration packages depend on @mlflow/core)
publish_pkg "${TS_ROOT}/core" "@mlflow/core"

# Then integrations
for pkg in openai anthropic gemini; do
  publish_pkg "${TS_ROOT}/integrations/${pkg}" "@mlflow/${pkg}"
done

rm -f "${TS_ROOT}/.npmrc"

# --- 6. Smoke tests: install from Verdaccio and verify ---
echo ""
echo "=== Running smoke tests ==="

SMOKE_DIR="${WORK_DIR}/smoke-test"
mkdir -p "$SMOKE_DIR"
cd "$SMOKE_DIR"

cat > package.json <<'JSON'
{"name":"smoke-test","version":"1.0.0","private":true}
JSON

# Write .npmrc in the smoke test dir so npm reads it automatically
write_npmrc "$SMOKE_DIR"

# Install core and verify
echo ""
echo "--- @mlflow/core ---"
npm install @mlflow/core 2>&1
node -e "
  const m = require('@mlflow/core');
  const expected = ['init', 'trace', 'startSpan', 'withSpan', 'flushTraces', 'MlflowClient'];
  const missing = expected.filter(s => typeof m[s] === 'undefined');
  if (missing.length > 0) {
    console.error('FAIL: @mlflow/core missing exports: ' + missing.join(', '));
    process.exit(1);
  }
  console.log('PASS: @mlflow/core');
"

# Install and verify each integration
echo ""
echo "--- @mlflow/openai ---"
npm install @mlflow/openai --legacy-peer-deps 2>&1
node -e "
  const m = require('@mlflow/openai');
  if (typeof m.tracedOpenAI !== 'function') {
    console.error('FAIL: @mlflow/openai missing tracedOpenAI');
    process.exit(1);
  }
  console.log('PASS: @mlflow/openai');
"

echo ""
echo "--- @mlflow/anthropic ---"
npm install @mlflow/anthropic --legacy-peer-deps 2>&1
node -e "
  const m = require('@mlflow/anthropic');
  if (typeof m.tracedAnthropic !== 'function') {
    console.error('FAIL: @mlflow/anthropic missing tracedAnthropic');
    process.exit(1);
  }
  console.log('PASS: @mlflow/anthropic');
"

echo ""
echo "--- @mlflow/gemini ---"
npm install @mlflow/gemini --legacy-peer-deps 2>&1
node -e "
  const m = require('@mlflow/gemini');
  if (typeof m.tracedGemini !== 'function') {
    console.error('FAIL: @mlflow/gemini missing tracedGemini');
    process.exit(1);
  }
  console.log('PASS: @mlflow/gemini');
"

echo ""
echo "=== All smoke tests passed ==="
