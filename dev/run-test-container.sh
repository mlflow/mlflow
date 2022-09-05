ROOT_DIR=$(git rev-parse --show-toplevel)
DOCKER_BUILDKIT=1 docker build -f "${ROOT_DIR}/dev/Dockerfile.test" -t mlflow-test-env "$ROOT_DIR"
docker container rm -f mlflow-test 2> /dev/null
docker run \
  -v "${ROOT_DIR}"/tests:/app/tests \
  -v "${ROOT_DIR}"/mlflow:/app/mlflow \
  -v "${ROOT_DIR}"/pylintrc:/app/pylintrc \
  -v "${ROOT_DIR}"/pyproject.toml:/app/pyproject.toml \
  -v "${ROOT_DIR}"/pytest.ini:/app/pytest.ini \
  -v "${ROOT_DIR}"/conftest.py:/app/conftest.py \
  --name "mlflow-test" \
  -it mlflow-test-env

docker container rm mlflow-test
