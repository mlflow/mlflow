#!/usr/bin/env bash
#
# Script to update OpenTelemetry proto files in the MLflow repository.
#
# Usage:
#   ./mlflow/protos/opentelemetry/update.sh
#

set -eo pipefail

# Commit SHA from opentelemetry-proto repository
# v1.7.0: https://github.com/open-telemetry/opentelemetry-proto/commit/8654ab7a5a43ca25fe8046e59dcd6935c3f76de0
COMMIT_SHA="8654ab7a5a43ca25fe8046e59dcd6935c3f76de0"
ARCHIVE_URL="https://github.com/open-telemetry/opentelemetry-proto/archive/${COMMIT_SHA}.tar.gz"

# Navigate to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Remove existing proto directory
rm -rf proto

echo "Fetching and extracting .proto files from: $ARCHIVE_URL"

# Stream and extract only the proto files needed by MLflow
curl -fsSL "$ARCHIVE_URL" | tar --strip-components=1 -xzf - \
  '*/opentelemetry/proto/trace/v1/trace.proto' \
  '*/opentelemetry/proto/common/v1/common.proto' \
  '*/opentelemetry/proto/resource/v1/resource.proto'

echo "Extraction complete."
