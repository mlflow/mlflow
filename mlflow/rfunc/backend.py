import logging
import os
import re
from shlex import quote
import subprocess

from mlflow.models import FlavorBackend
from mlflow.tracking.artifact_utils import _download_artifact_from_uri

_logger = logging.getLogger(__name__)


class RFuncBackend(FlavorBackend):
    """
    Flavor backend implementation for the generic R models.
    Predict and serve locally models with 'crate' flavor.
    """

    version_pattern = re.compile("version ([0-9]+[.][0-9]+[.][0-9]+)")

    def predict(self, model_uri, input_path, output_path, content_type, json_format):
        """
        Generate predictions using R model saved with MLflow.
        Return the prediction results as a JSON.
        """
        model_path = _download_artifact_from_uri(model_uri)
        str_cmd = (
            "mlflow:::mlflow_rfunc_predict(model_path = '{0}', input_path = {1}, "
            "output_path = {2}, content_type = {3}, json_format = {4})"
        )
        command = str_cmd.format(
            quote(model_path),
            _str_optional(input_path),
            _str_optional(output_path),
            _str_optional(content_type),
            _str_optional(json_format),
        )
        _execute(command)

    def serve(self, model_uri, port, host, enable_mlserver):
        """
        Generate R model locally.

        NOTE: The `enable_mlserver` parameter is there to comply with the
        FlavorBackend interface but is not supported by MLServer yet.
        https://github.com/SeldonIO/MLServer/issues/183
        """
        if enable_mlserver:
            raise Exception("The MLServer inference server is not yet supported in the R backend.")

        model_path = _download_artifact_from_uri(model_uri)
        command = "mlflow::mlflow_rfunc_serve('{0}', port = {1}, host = '{2}')".format(
            quote(model_path), port, host
        )
        _execute(command)

    def can_score_model(self):
        process = subprocess.Popen(
            ["Rscript", "--version"], close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        _, stderr = process.communicate()
        if process.wait() != 0:
            return False

        version = self.version_pattern.search(stderr.decode("utf-8"))
        if not version:
            return False
        version = [int(x) for x in version.group(1).split(".")]
        return version[0] > 3 or version[0] == 3 and version[1] >= 3


def _execute(command):
    env = os.environ.copy()
    import sys

    process = subprocess.Popen(
        ["Rscript", "-e", command],
        env=env,
        close_fds=False,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if process.wait() != 0:
        raise Exception("Command returned non zero exit code.")


def _str_optional(s):
    return "NULL" if s is None else "'{}'".format(quote(str(s)))
