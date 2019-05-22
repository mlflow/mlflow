import logging
import os
import subprocess

from mlflow.pyfunc import ENV

from mlflow.pyfunc import scoring_server
from mlflow.models import FlavorBackend

from mlflow.utils.file_utils import TempDir
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.projects import _get_or_create_conda_env, _get_conda_bin_executable

from six.moves import shlex_quote

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
                command = ("mlflow models predict --no-conda "
                           "--model-uri {0} "
                           "--content-type {1} "
                           "--json-format {2}").format(
                    shlex_quote(local_path),
                    content_type,
                    json_format)
                if input_path is not None:
                    command += " -i {}".format(shlex_quote(input_path))
                if output_path is not None:
                    command += " -o {}".format(shlex_quote(output_path))
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
                command = "mlflow models serve --no-conda --model-uri {0} --port {1} --host {2}". \
                    format(shlex_quote(local_path), port, host)
                return scoring_server._execute_in_conda_env(conda_env_path, command,
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
        activate_conda_env += ["pip install -U mlflow 1>&2"]

    command = " && ".join(activate_conda_env + [command])
    _logger.info("=== Running command '%s'", command)
    child = subprocess.Popen(["bash", "-c", command], close_fds=True)
    rc = child.wait()
    if rc != 0:
        raise Exception("Command '{0}' returned non zero return code. Return code = {1}".format(
            command, rc
        ))
