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
            enable_mlserver=False,
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
            enable_mlserver=False,
        )

    mock_execute.assert_called_once_with(
        "mlflow::mlflow_rfunc_serve('/tmp/model', port = 5000, host = '127.0.0.1')"
    )
