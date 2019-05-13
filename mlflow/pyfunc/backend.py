import os
import subprocess

from mlflow.pyfunc import ENV

from mlflow.pyfunc import scoring_server
from mlflow.models import FlavorBackend

from mlflow.utils.file_utils import TempDir
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.projects import _get_conda_bin_executable

class PyFuncBackend(FlavorBackend):

    def predict(self, model_uri, input_path, output_path, content_type, no_conda, **kwargs):
        with TempDir() as tmp:
            local_path = _download_artifact_from_uri(model_uri, output_path=tmp.path())
            if not no_conda and ENV in self._config:
                conda_env_path = os.path.join(local_path, self._config[ENV])
                command = "python {0} predict {1} {2} {3} {4}".format(scoring_server.__file__,
                                                                      local_path, input_path,
                                                                      output_path, content_type)
                return scoring_server._execute_in_conda_env(conda_env_path, command)
            else:
                scoring_server._predict(local_path, input_path, output_path, content_type)

    def serve(self, model_uri, port, host, no_conda, **kwargs):
        with TempDir() as tmp:
            local_path = _download_artifact_from_uri(model_uri, output_path=tmp.path())
            if not no_conda and ENV in self._config:
                conda_env_path = os.path.join(local_path, self._config[ENV])
                command = "python {0} serve {1} {2} {3}".format(scoring_server.__file__,
                                                               local_path, port, host)
                return scoring_server._execute_in_conda_env(conda_env_path, command)
            else:
                scoring_server._serve(local_path, port, host)

    def can_score_model(self, no_conda, **kwargs):
        if no_conda:
            return True  # already in python; dependencies are assumed to be installed (no_conda)
        conda_path = _get_conda_bin_executable("conda")
        p = subprocess.Popen([conda_path, "--version"])
        return p.wait() == 0
