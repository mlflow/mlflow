#!/usr/bin/env bash
set -euo pipefail

# Parse command line arguments
BUILD_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            BUILD_ARGS="--no-cache"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-cache]"
            exit 1
            ;;
    esac
done

if ! docker info >/dev/null 2>&1; then
    echo "Docker is not available. Please install Docker and try again."
    exit 1
fi

echo "Building Docker image for proto compilation..."
IMAGE_NAME="mlflow-protos-gen"
docker build -t "$IMAGE_NAME" -f dev/Dockerfile.protos $BUILD_ARGS .

# Cleanup function to ensure container is always removed
cleanup() {
    if [ -n "${CONTAINER_ID:-}" ]; then
        echo "Cleaning up container..."
        docker stop "$CONTAINER_ID" || true
        docker rm "$CONTAINER_ID" || true
    fi
}

echo "Running proto compilation..."
CONTAINER_ID=$(docker run -d --name mlflow-protos-gen "$IMAGE_NAME" python dev/generate_protos.py)
# Set trap to call cleanup on EXIT (normal or error)
trap cleanup EXIT
docker wait "$CONTAINER_ID"
docker logs "$CONTAINER_ID"

echo "Copying generated files back..."
docker cp "$CONTAINER_ID:/mlflow/mlflow/protos/." "mlflow/protos/"
docker cp "$CONTAINER_ID:/mlflow/tests/protos/." "tests/protos/"
docker cp "$CONTAINER_ID:/mlflow/mlflow/java/client/src/main/java/." "mlflow/java/client/src/main/java/"

echo "Generating GraphQL schema from Protobuf files..."
python ./dev/proto_to_graphql/code_generator.py

echo "✓ Done!"
