#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)

DOCKER_BUILDKIT=1 docker build -f "${ROOT_DIR}/docker/Dockerfile.full.dev" -t mlflow-docker-test "$ROOT_DIR"