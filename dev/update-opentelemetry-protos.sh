#!/usr/bin/env bash
#
# Script to update OpenTelemetry proto files in the MLflow repository.
#
# Usage:
#   ./dev/update-opentelemetry-protos.sh
#

set -eo pipefail

VERSION="v1.7.0"
ARCHIVE_URL="https://github.com/open-telemetry/opentelemetry-proto/archive/refs/tags/${VERSION}.tar.gz"

# Get repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/mlflow/protos/opentelemetry"

# Create target directory and navigate to it
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

# Remove existing proto directory
rm -rf proto

echo "Fetching and extracting .proto files from: $ARCHIVE_URL"

# Stream and extract only the proto files needed by MLflow
curl -sL "$ARCHIVE_URL" | tar --strip-components=1 -xzf - \
  '*/opentelemetry/proto/trace/v1/trace.proto' \
  '*/opentelemetry/proto/common/v1/common.proto' \
  '*/opentelemetry/proto/resource/v1/resource.proto'

echo "Extraction complete."
