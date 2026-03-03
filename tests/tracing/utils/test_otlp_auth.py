import base64

from mlflow.tracing.utils.otlp import MLFLOW_EXPERIMENT_ID_HEADER, build_otlp_headers


def test_build_otlp_headers_no_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    # Prevent ~/.mlflow/credentials from leaking into the test
    monkeypatch.setenv("HOME", str(tmp_path))

    headers = build_otlp_headers("42")
    assert headers == {MLFLOW_EXPERIMENT_ID_HEADER: "42"}
    assert "Authorization" not in headers


def test_build_otlp_headers_basic_auth(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "s3cret")

    headers = build_otlp_headers("7")
    expected = base64.standard_b64encode(b"admin:s3cret").decode()
    assert headers[MLFLOW_EXPERIMENT_ID_HEADER] == "7"
    assert headers["Authorization"] == f"Basic {expected}"


def test_build_otlp_headers_bearer_token(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "tok-abc")
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

    headers = build_otlp_headers("1")
    assert headers[MLFLOW_EXPERIMENT_ID_HEADER] == "1"
    assert headers["Authorization"] == "Bearer tok-abc"


def test_build_otlp_headers_basic_auth_takes_precedence_over_token(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "tok-xyz")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pass")

    headers = build_otlp_headers("5")
    expected = base64.standard_b64encode(b"admin:pass").decode()
    assert headers[MLFLOW_EXPERIMENT_ID_HEADER] == "5"
    assert headers["Authorization"] == f"Basic {expected}"
