import shutil
from unittest import mock

import pytest

from mlflow.rfunc.backend import RFuncBackend, _r_quote


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("localhost", "'localhost'"),
        ("0.0.0.0", "'0.0.0.0'"),
        ("foo bar", "'foo bar'"),
        ("it's", "'it\\'s'"),
        ("back\\slash", "'back\\\\slash'"),
        (
            "localhost'); file.create('/tmp/pwned'); cat('",
            "'localhost\\'); file.create(\\'/tmp/pwned\\'); cat(\\''",
        ),
    ],
)
def test_r_quote_escapes_single_quotes_and_backslashes(value, expected):
    assert _r_quote(value) == expected


def test_serve_escapes_malicious_host():
    backend = RFuncBackend(config={})
    malicious_host = "localhost'); file.create('/tmp/pwned'); cat('"

    with (
        mock.patch(
            "mlflow.rfunc.backend._download_artifact_from_uri",
            return_value="/tmp/model",
        ) as mock_download,
        mock.patch("mlflow.rfunc.backend._execute") as mock_execute,
    ):
        backend.serve(
            model_uri="models:/foo/1",
            port=5000,
            host=malicious_host,
            timeout=None,
        )

    mock_download.assert_called_once_with("models:/foo/1")
    mock_execute.assert_called_once()
    (command,) = mock_execute.call_args.args

    # The injected payload must not appear as a separate top-level R statement;
    # the single quotes that would close the host argument must be escaped.
    assert "file.create('/tmp/pwned')" not in command
    assert "\\'); file.create(\\'/tmp/pwned\\'); cat(\\'" in command
    # The full host value is contained as a single R single-quoted string literal.
    assert "host = 'localhost\\'); file.create(\\'/tmp/pwned\\'); cat(\\''" in command
    # The model path is also passed as an R string literal (no surrounding `'{}'`
    # template anymore - `_r_quote` adds its own quotes).
    assert "mlflow::mlflow_rfunc_serve('/tmp/model', port = 5000, host = " in command


def test_serve_benign_host():
    backend = RFuncBackend(config={})

    with (
        mock.patch(
            "mlflow.rfunc.backend._download_artifact_from_uri",
            return_value="/tmp/model",
        ),
        mock.patch("mlflow.rfunc.backend._execute") as mock_execute,
    ):
        backend.serve(
            model_uri="models:/foo/1",
            port=5000,
            host="127.0.0.1",
            timeout=None,
        )

    mock_execute.assert_called_once_with(
        "mlflow::mlflow_rfunc_serve('/tmp/model', port = 5000, host = '127.0.0.1')"
    )


def test_generate_dockerfile_without_model(tmp_path):
    RFuncBackend(config={}).generate_dockerfile(model_uri=None, output_dir=str(tmp_path))

    dockerfile = (tmp_path / "Dockerfile").read_text()
    assert "FROM rocker/r-ver:4.4.2" in dockerfile
    assert "install.packages('mlflow'" in dockerfile
    assert "No model baked in" in dockerfile
    assert 'ENTRYPOINT ["Rscript", "-e", "mlflow::mlflow_rfunc_serve(' in dockerfile
    assert "EXPOSE 8080" in dockerfile


def test_generate_dockerfile_with_model_copies_it(tmp_path):
    src_model = tmp_path / "src_model"
    src_model.mkdir()
    (src_model / "MLmodel").write_text("flavors: {}\n")
    out = tmp_path / "out"

    def fake_download(uri, output_path):
        shutil.copytree(src_model, output_path)

    with mock.patch("mlflow.rfunc.backend._download_artifact_from_uri", side_effect=fake_download):
        RFuncBackend(config={}).generate_dockerfile(model_uri="models:/m/1", output_dir=str(out))

    assert "COPY model /opt/ml/model" in (out / "Dockerfile").read_text()
    assert (out / "model" / "MLmodel").exists()
