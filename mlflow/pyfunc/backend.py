import logging
import os
from six.moves import shlex_quote
import subprocess

from mlflow.models import FlavorBackend
from mlflow.pyfunc import ENV
from mlflow.pyfunc import scoring_server
from mlflow.projects import _get_or_create_conda_env, _get_conda_bin_executable
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)


class PyFuncBackend(FlavorBackend):
    """
        Flavor backend implementation for the generic python models.
    """
    def __init__(self, config, no_conda=False, install_mlflow=False, **kwargs):
        super(PyFuncBackend, self).__init__(config=config, **kwargs)
        self._no_conda = no_conda
        self._install_mlflow = install_mlflow

    def predict(self, model_uri, input_path, output_path, content_type, json_format, ):
        """
        Generate predictions using generic python model saved with MLflow.
        Return the prediction results as a JSON.
        """
        with TempDir() as tmp:
            local_path = _download_artifact_from_uri(model_uri, output_path=tmp.path())
            if not self._no_conda and ENV in self._config:
                conda_env_path = os.path.join(local_path, self._config[ENV])
                command = ('python -c "from mlflow.models.cli import _predict; _predict('
                           'model_uri={model_uri}, '
                           'input_path={input_path}, '
                           'output_path={output_path}, '
                           'content_type={content_type}, '
                           'json_format={json_format}, '
                           'no_conda=True, '
                           'install_mlflow=False)"'
                           ).format(
                    model_uri=repr(model_uri),
                    input_path=repr(input_path),
                    output_path=repr(output_path),
                    content_type=repr(content_type),
                    json_format=repr(json_format))
                return _execute_in_conda_env(conda_env_path, command, self._install_mlflow)
            else:
                scoring_server._predict(local_path, input_path, output_path, content_type,
                                        json_format)

    def serve(self, model_uri, port, host):
        """
        Serve pyfunc model locally.
        """
        with TempDir() as tmp:
            local_path = shlex_quote(_download_artifact_from_uri(model_uri, output_path=tmp.path()))
            if not self._no_conda and ENV in self._config:
                conda_env_path = os.path.join(local_path, self._config[ENV])

                command = ('python -c "from mlflow.models.cli import _serve; _serve('
                           'model_uri={model_uri}, '
                           'port={port}, '
                           'host={host}, '
                           'no_conda=True,'
                           'install_mlflow=False'
                           ).format(
                    model_uri=repr(model_uri),
                    input_path=repr(input_path),
                    output_path=repr(output_path))

                return _execute_in_conda_env(conda_env_path, command,
                                             self._install_mlflow)
            else:
                scoring_server._serve(local_path, port, host)

    def can_score_model(self):
        if self._no_conda:
            return True  # already in python; dependencies are assumed to be installed (no_conda)
        conda_path = _get_conda_bin_executable("conda")
        try:
            p = subprocess.Popen([conda_path, "--version"], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            _, _ = p.communicate()
            return p.wait() == 0
        except FileNotFoundError:
            # Can not find conda
            return False


def _execute_in_conda_env(conda_env_path, command, install_mlflow):
    conda_env_name = _get_or_create_conda_env(conda_env_path)
    activate_path = _get_conda_bin_executable("activate")
    activate_conda_env = ["source {0} {1}".format(activate_path, conda_env_name)]

    if install_mlflow:
        if "MLFLOW_HOME" in os.environ:  # dev version
            install_mlflow = "pip install -e {} 1>&2".format(os.environ["MLFLOW_HOME"])
        else:
            from mlflow.version import VERSION
            install_mlflow = "pip install -U mlflow>={} 1>&2".format(VERSION)

        activate_conda_env += [install_mlflow]

    command = " && ".join(activate_conda_env + [command])
    _logger.info("=== Running command '%s'", command)
    child = subprocess.Popen(["bash", "-c", command], close_fds=True)
    rc = child.wait()
    if rc != 0:
        raise Exception("Command '{0}' returned non zero return code. Return code = {1}".format(
            command, rc
        ))
