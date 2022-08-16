ROOT_DIR=$(git rev-parse --show-toplevel)
DOCKER_BUILDKIT=1 docker build -f "${ROOT_DIR}/dev/Dockerfile.test" -t mlflow-test-env "$ROOT_DIR"
docker run -v "${ROOT_DIR}"/tests:/app/tests -v "${ROOT_DIR}"/mlflow:/app/mlflow --name "mlflow-test" -it mlflow-test-env
docker container rm mlflow-test
