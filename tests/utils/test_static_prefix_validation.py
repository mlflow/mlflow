import pytest

from mlflow.utils.static_prefix_validation import validate_static_prefix_security


@pytest.mark.parametrize("value", ["/myteam", "/ajax-api/3.0/jobs-v2", "/gateway-v2"])
def test_validate_static_prefix_security_accepts_safe_values(value):
    validate_static_prefix_security(value)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("/{user}", "must not contain"),
        ("/x{y}", "must not contain"),
        ("/gateway", "conflicts with the reserved route"),
        ("/gateway/sub", "conflicts with the reserved route"),
        ("/v1", "conflicts with the reserved route"),
        ("/ajax-api/3.0/jobs", "conflicts with the reserved route"),
        ("/api/2.0/mlflow-artifacts/artifacts", "conflicts with the reserved route"),
        ("/ajax-api/2.0/mlflow-artifacts/artifacts/sub", "conflicts with the reserved route"),
        ("/health", "conflicts with the unauthenticated routes"),
        ("/healthcheck", "conflicts with the unauthenticated routes"),
        ("/static-files", "conflicts with the unauthenticated routes"),
        ("/favicon.ico", "conflicts with the unauthenticated routes"),
    ],
)
def test_validate_static_prefix_security_rejects_unsafe_values(value, match):
    with pytest.raises(ValueError, match=match):
        validate_static_prefix_security(value)
