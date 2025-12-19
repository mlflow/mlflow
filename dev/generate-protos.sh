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
    echo "Docker is not available. Please install Docker or start Docker daemon and try again."
    exit 1
fi

echo "Building Docker image for proto compilation..."
IMAGE_NAME="mlflow-protos-gen"
docker build -t "$IMAGE_NAME" -f dev/Dockerfile.protos $BUILD_ARGS .

echo "Running proto compilation..."
CONTAINER_NAME="mlflow-protos-gen-$$"
# Cleanup function to ensure container is always removed
cleanup() {
    echo "Cleaning up container..."
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
}
# Set trap to call cleanup on EXIT (normal or error)
trap cleanup EXIT

docker run --name "$CONTAINER_NAME" "$IMAGE_NAME" python dev/generate_protos.py

echo "Copying generated files back..."
docker cp "$CONTAINER_NAME:/mlflow/mlflow/protos/." "mlflow/protos/"
docker cp "$CONTAINER_NAME:/mlflow/tests/protos/." "tests/protos/"
docker cp "$CONTAINER_NAME:/mlflow/mlflow/java/client/src/main/java/." "mlflow/java/client/src/main/java/"

echo "Generating GraphQL schema from Protobuf files..."
uv run ./dev/proto_to_graphql/code_generator.py

echo "Done!"
