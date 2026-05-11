import shlex
from unittest import mock

import pytest

from mlflow.pyfunc.backend import _STDIN_SERVER_SCRIPT, PyFuncBackend


@pytest.mark.parametrize(
    "local_path",
    [
        "/tmp/model$(touch /tmp/pwned)",
        "/tmp/model; rm -rf /tmp/foo",
        "/tmp/model with spaces",
        "/tmp/model`whoami`",
        "/tmp/model|cat /etc/passwd",
        "/tmp/plain_model",
    ],
)
def test_serve_stdin_shell_quotes_local_path(local_path):
    backend = PyFuncBackend(config={}, env_manager="virtualenv")

    mock_environment = mock.MagicMock()
    with (
        mock.patch(
            "mlflow.pyfunc.backend._download_artifact_from_uri", return_value=local_path
        ) as mock_download,
        mock.patch.object(backend, "prepare_env", return_value=mock_environment) as mock_prepare,
    ):
        backend.serve_stdin(model_uri="models:/test/1")

    mock_download.assert_called_once_with("models:/test/1")
    mock_prepare.assert_called_once_with(local_path)
    mock_environment.execute.assert_called_once()

    command = mock_environment.execute.call_args.kwargs["command"]
    # The path must appear in shell-quoted form to neutralize shell metacharacters.
    assert shlex.quote(local_path) in command
    assert command.startswith(f"python {shlex.quote(str(_STDIN_SERVER_SCRIPT))} --model-uri ")

    # Tokenizing with shlex must round-trip the original local_path as a single argument so
    # metacharacters like $(...), ;, |, and backticks never get interpreted by the shell.
    tokens = shlex.split(command)
    assert tokens[-1] == local_path
    assert tokens[-2] == "--model-uri"
