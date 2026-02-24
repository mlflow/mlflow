#!/bin/sh
set -euo pipefail

echo ">>> Wait a bit for Garage RPC..."
sleep 10

echo ">>> garage status"
/garage status

echo ">>> Resolve node id"
NODE_ID=$(/garage status | awk '/Node ID/{print $3}')

echo ">>> layout assign/apply (id=${NODE_ID})"
/garage layout assign -z "${INIT_LAYOUT_ZONE}" -c "${INIT_LAYOUT_CAPACITY}" "${NODE_ID}" || true
/garage layout apply --version 1 || true

echo ">>> create/list key"
# 4) Chiavi/bucket/allow (idempotente con 'key create')
if [ -f /run/garage-keys/env ]; then
  echo ">>> Found existing /run/garage-keys/env, reusing credentials"
  . /run/garage-keys/env
else
  echo ">>> creating access key: ${INIT_KEY_NAME}"
  set +e
  KEY_OUT=$(/garage key create "${INIT_KEY_NAME}" 2>&1)
  RC=$?
  set -e
  if [ $RC -ne 0 ]; then
    echo "WARN: 'garage key create' failed (${RC}):"
    echo "$KEY_OUT"
    echo ">>> trying to read existing key info"
    KEY_OUT=$(/garage key info "${INIT_KEY_NAME}")
    echo "$KEY_OUT"
  fi

  echo ">>> parsing key output"
  echo "$KEY_OUT"
  
  KEY_ID=$(echo "$KEY_OUT" | grep -oE 'GK[a-f0-9]{24}' | head -1)
  SECRET=$(echo "$KEY_OUT" | awk -F': ' '/Secret key/{print $2}')

  if [ -z "${KEY_ID:-}" ]; then
    echo "FATAL: unable to get KEY_ID (key may not exist or command changed)"; exit 1
  fi

  echo ">>> ensure bucket + allow"
  /garage bucket create "${INIT_BUCKET}" || true
  /garage bucket allow --read --write --owner "${INIT_BUCKET}" --key "${KEY_ID}" || true

  
  KEY_ID=$(printf '%s\n' "$KEY_ID" | tr -d '[:space:]')
  SECRET=$(printf '%s\n' "$SECRET" | tr -d '[:space:]')


  if [ -z "$KEY_ID" ] || [ -z "$SECRET" ] || [ -z "$INIT_S3_REGION" ]; then
    echo "FATAL: missing KEY_ID/SECRET/REGION, aborting." >&2
    exit 1
  fi

  mkdir -p /run/garage-keys
  echo ">>> saving credentials to /run/garage-keys/env"
  printf 'S3_ACCESS_KEY=%s\nS3_SECRET_KEY=%s\nS3_REGION=%s\n' \
     "${KEY_ID}" "${SECRET:-}" "${INIT_S3_REGION}" > /run/garage-keys/env
  RES=$(cat /run/garage-keys/env)
  echo "$RES"
fi

touch /run/garage-keys/READY
echo ">>> init done."