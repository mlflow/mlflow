from mlflow.server.auth.utils import is_unprotected_route


def test_health_without_prefix_is_unprotected():
    assert is_unprotected_route("/health")


def test_health_with_prefix_is_unprotected():
    assert is_unprotected_route("/mlflow/health", static_prefix="/mlflow")


def test_version_with_prefix_is_unprotected():
    assert is_unprotected_route("/mlflow/version", static_prefix="/mlflow")


def test_static_assets_with_prefix_are_unprotected():
    assert is_unprotected_route("/mlflow/static/app.js", static_prefix="/mlflow")


def test_api_routes_remain_protected():
    assert not is_unprotected_route(
        "/mlflow/api/2.0/mlflow/runs/list",
        static_prefix="/mlflow",
    )