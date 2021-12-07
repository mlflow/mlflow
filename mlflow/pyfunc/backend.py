import logging
import os

import subprocess
import posixpath
from mlflow.models import FlavorBackend
from mlflow.models.docker_utils import _build_image, DISABLE_ENV_CREATION
from mlflow.models.container import ENABLE_MLSERVER
from mlflow.pyfunc import ENV, scoring_server, mlserver

from mlflow.utils.conda import get_or_create_conda_env, get_conda_bin_executable, get_conda_command
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)


class PyFuncBackend(FlavorBackend):
    """
    Flavor backend implementation for the generic python models.
    """

    def __init__(self, config, workers=1, no_conda=False, install_mlflow=False, **kwargs):
        super().__init__(config=config, **kwargs)
        self._nworkers = workers or 1
        self._no_conda = no_conda
        self._install_mlflow = install_mlflow

    def prepare_env(self, model_uri):
        local_path = _download_artifact_from_uri(model_uri)
        if self._no_conda or ENV not in self._config:
            return 0
        conda_env_path = os.path.join(local_path, self._config[ENV])
        command = 'python -c ""'
        return _execute_in_conda_env(conda_env_path, command, self._install_mlflow)

    def predict(self, model_uri, input_path, output_path, content_type, json_format):
        """
        Generate predictions using generic python model saved with MLflow.
        Return the prediction results as a JSON.
        """
        local_path = _download_artifact_from_uri(model_uri)
        # NB: Absolute windows paths do not work with mlflow apis, use file uri to ensure
        # platform compatibility.
        local_uri = path_to_local_file_uri(local_path)
        if not self._no_conda and ENV in self._config:
            conda_env_path = os.path.join(local_path, self._config[ENV])
            command = (
                'python -c "from mlflow.pyfunc.scoring_server import _predict; _predict('
                "model_uri={model_uri}, "
                "input_path={input_path}, "
                "output_path={output_path}, "
                "content_type={content_type}, "
                'json_format={json_format})"'
            ).format(
                model_uri=repr(local_uri),
                input_path=repr(input_path),
                output_path=repr(output_path),
                content_type=repr(content_type),
                json_format=repr(json_format),
            )
            return _execute_in_conda_env(conda_env_path, command, self._install_mlflow)
        else:
            scoring_server._predict(local_uri, input_path, output_path, content_type, json_format)

    def serve(self, model_uri, port, host, enable_mlserver):  # pylint: disable=W0221
        """
        Serve pyfunc model locally.
        """
        local_path = _download_artifact_from_uri(model_uri)

        server_implementation = mlserver if enable_mlserver else scoring_server
        command, command_env = server_implementation.get_cmd(local_path, port, host, self._nworkers)

        if not self._no_conda and ENV in self._config:
            conda_env_path = os.path.join(local_path, self._config[ENV])
            return _execute_in_conda_env(
                conda_env_path, command, self._install_mlflow, command_env=command_env
            )
        else:
            _logger.info("=== Running command '%s'", command)
            if os.name != "nt":
                subprocess.Popen(["bash", "-c", command], env=command_env).wait()
            else:
                subprocess.Popen(command, env=command_env).wait()

    def can_score_model(self):
        if self._no_conda:
            # noconda => already in python and dependencies are assumed to be installed.
            return True
        conda_path = get_conda_bin_executable("conda")
        try:
            p = subprocess.Popen(
                [conda_path, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            _, _ = p.communicate()
            return p.wait() == 0
        except FileNotFoundError:
            # Can not find conda
            return False

    def build_image(
        self, model_uri, image_name, install_mlflow=False, mlflow_home=None, enable_mlserver=False
    ):
        def copy_model_into_container(dockerfile_context_dir):
            model_cwd = os.path.join(dockerfile_context_dir, "model_dir")
            os.mkdir(model_cwd)
            model_path = _download_artifact_from_uri(model_uri, output_path=model_cwd)
            return """
                COPY {model_dir} /opt/ml/model
                RUN python -c \
                'from mlflow.models.container import _install_pyfunc_deps;\
                _install_pyfunc_deps(\
                    "/opt/ml/model", \
                    install_mlflow={install_mlflow}, \
                    enable_mlserver={enable_mlserver})'
                ENV {disable_env}="true"
                ENV {ENABLE_MLSERVER}={enable_mlserver}
                """.format(
                disable_env=DISABLE_ENV_CREATION,
                model_dir=str(posixpath.join("model_dir", os.path.basename(model_path))),
                install_mlflow=repr(install_mlflow),
                ENABLE_MLSERVER=ENABLE_MLSERVER,
                enable_mlserver=repr(enable_mlserver),
            )

        # The pyfunc image runs the same server as the Sagemaker image
        pyfunc_entrypoint = (
            'ENTRYPOINT ["python", "-c", "from mlflow.models import container as C; C._serve()"]'
        )
        _build_image(
            image_name=image_name,
            mlflow_home=mlflow_home,
            custom_setup_steps_hook=copy_model_into_container,
            entrypoint=pyfunc_entrypoint,
        )


def _execute_in_conda_env(conda_env_path, command, install_mlflow, command_env=None):
    if command_env is None:
        command_env = os.environ
    env_id = os.environ.get("MLFLOW_HOME", VERSION) if install_mlflow else None
    conda_env_name = get_or_create_conda_env(conda_env_path, env_id=env_id)
    activate_conda_env = get_conda_command(conda_env_name)
    if install_mlflow:
        if "MLFLOW_HOME" in os.environ:  # dev version
            install_mlflow = "pip install -e {} 1>&2".format(os.environ["MLFLOW_HOME"])
        else:
            install_mlflow = "pip install mlflow=={} 1>&2".format(VERSION)

        activate_conda_env += [install_mlflow]
    if os.name != "nt":
        separator = " && "
    else:
        separator = " & "

    command = separator.join(activate_conda_env + [command])
    _logger.info("=== Running command '%s'", command)

    if os.name != "nt":
        child = subprocess.Popen(["bash", "-c", command], close_fds=True, env=command_env)
    else:
        child = subprocess.Popen(["cmd", "/c", command], close_fds=True, env=command_env)
    rc = child.wait()
    if rc != 0:
        raise Exception(
            "Command '{0}' returned non zero return code. Return code = {1}".format(command, rc)
        )
