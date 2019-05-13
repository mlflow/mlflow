import logging
import os
import re
import subprocess

from mlflow.models import FlavorBackend
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir

_logger = logging.getLogger(__name__)


class RFuncBackend(FlavorBackend):
    version_pattern = re.compile("version ([0-9]+[.][0-9]+[.][0-9]+)")

    def predict(self, model_uri, input_path, output_path, **kwargs):
        """
           Serve an RFunction model saved with MLflow.
           Return the prediction results as a JSON DataFrame.

           If a ``run-id`` is specified, ``model-path`` is treated as an artifact path within that run;
           otherwise it is treated as a local path.
           """
        with TempDir() as tmp:
            model_path = _download_artifact_from_uri(model_uri, output_path=tmp.path())
            str_cmd = "mlflow::mlflow_rfunc_predict('{0}', {1}, {2})"
            command = str_cmd.format(model_path, _str_optional(input_path),
                                     _str_optional(output_path))
            _execute(command)

    def serve(self, model_uri, port, **kwargs):
        with TempDir() as tmp:
            model_path = _download_artifact_from_uri(model_uri, output_path=tmp.path())
            command = "mlflow::mlflow_rfunc_serve('{0}', port = {1})".format(model_path, port)
            _execute(command)

    def can_score_model(self, **kwargs):
        process = subprocess.Popen(["Rscript", "--version"], close_fds=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.wait() != 0:
            return False

        output = stdout  # process.stdout.read().decode("utf-8")
        version = self.version_pattern.search(stderr.decode("utf-8"))
        if not version:
            return False
        version = [int(x) for x in version.group(1).split(".")]
        return version[0] > 3 or version[0] == 3 and version[1] >= 3


def _execute(command):
    env = os.environ.copy()
    import sys
    process = subprocess.Popen(["Rscript", "-e", command], env=env, close_fds=False,
                               stdin=sys.stdin,
                               stdout=sys.stdout,
                               stderr=sys.stderr)
    if process.wait() != 0:
        raise Exception("Command returned non zero exit code.")


def _str_optional(s):
    return "NULL" if s is None else "'{}'".format(str(s))
