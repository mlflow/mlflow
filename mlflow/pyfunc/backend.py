import os
import subprocess

from mlflow.pyfunc import ENV

from mlflow.pyfunc import scoring_server
from mlflow.models import FlavorBackend

from mlflow.utils.file_utils import TempDir
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.projects import _get_conda_bin_executable

from six.moves import shlex_quote


class PyFuncBackend(FlavorBackend):
    """
        Flavor backend implementation for the generic python models.
    """

    def __init__(self, config, no_conda=False, **kwargs):
        super(PyFuncBackend, self).__init__(config=config, **kwargs)
        self._no_conda = no_conda

    def predict(self, model_uri, input_path, output_path, content_type, json_format):
        """
        Generate predictions using generic python model saved with MLflow.
        Return the prediction results as a JSON.
        """
        with TempDir() as tmp:
            local_path = _download_artifact_from_uri(model_uri, output_path=tmp.path())
            if not self._no_conda and ENV in self._config:
                conda_env_path = os.path.join(local_path, self._config[ENV])
                # NOTE: We're calling main in the pyfunc scoring server belonging to the current
                # conda environment. The model environment may contain mlflow with different version
                # than the one in the current active environment. This is the intended behavior.
                # We need to make sure the scoring server is consistent with the outside mlflow
                # while the model that is being loaded may depend on a different version of mlflow.
                # The hope is that the scoring server is self contained enough and does not have
                # external mlflow dependencies that would be incompatible between mlflow versions.
                if input_path is None:
                    input_path = "__stdin__"
                if output_path is None:
                    output_path = "__stdout__"
                command = "python {0} predict {1} {2} {3} {4} {5}".format(scoring_server.__file__,
                                                                          shlex_quote(local_path),
                                                                          shlex_quote(input_path),
                                                                          shlex_quote(output_path),
                                                                          content_type,
                                                                          json_format)
                return scoring_server._execute_in_conda_env(conda_env_path, command)
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
                command = "python {0} serve {1} {2} {3}".format(scoring_server.__file__,
                                                                shlex_quote(local_path),
                                                                port, host)
                return scoring_server._execute_in_conda_env(conda_env_path, command)
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
