from unittest import mock

from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.utils.env_manager import VIRTUALENV


def test_generate_dockerfile_logs_output_dir(tmp_path):
    backend = get_flavor_backend(None, docker_build=True, env_manager=VIRTUALENV)
    with mock.patch("mlflow.pyfunc.backend._logger.debug") as mock_debug:
        backend.generate_dockerfile(model_uri=None, output_dir=str(tmp_path))
        mock_debug.assert_any_call(
            f"generated dockerfile at {tmp_path}", extra={"dockerfile": mock.ANY}
        )
