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

# Download and extract to temp directory (works on both macOS BSD tar and GNU tar)
TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

curl -fsSL "$ARCHIVE_URL" | tar -xzf - -C "$TEMP_DIR"

# Copy only the proto files we need
EXTRACTED_DIR="${TEMP_DIR}/opentelemetry-proto-${COMMIT_SHA}"
mkdir -p proto/trace/v1 proto/common/v1 proto/resource/v1
cp "${EXTRACTED_DIR}/opentelemetry/proto/trace/v1/trace.proto" proto/trace/v1/
cp "${EXTRACTED_DIR}/opentelemetry/proto/common/v1/common.proto" proto/common/v1/
cp "${EXTRACTED_DIR}/opentelemetry/proto/resource/v1/resource.proto" proto/resource/v1/

echo "Extraction complete."
