import logging
import os
import re
import subprocess
import sys

from mlflow.exceptions import MlflowException
from mlflow.models import FlavorBackend
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.string_utils import quote

_logger = logging.getLogger(__name__)


class RFuncBackend(FlavorBackend):
    """
    Flavor backend implementation for the generic R models.
    Predict and serve locally models with 'crate' flavor.
    """

    def build_image(
        self, model_uri, image_name, install_mlflow, mlflow_home, enable_mlserver, base_image=None
    ):
        pass

    def generate_dockerfile(
        self, model_uri, output_path, install_mlflow, mlflow_home, enable_mlserver, base_image=None
    ):
        pass

    version_pattern = re.compile(r"version ([0-9]+\.[0-9]+\.[0-9]+)")

    def predict(
        self, model_uri, input_path, output_path, content_type, pip_requirements_override=None
    ):
        """
        Generate predictions using R model saved with MLflow.
        Return the prediction results as a JSON.
        """
        if pip_requirements_override is not None:
            raise MlflowException("pip_requirements_override is not supported in the R backend.")
        model_path = _download_artifact_from_uri(model_uri)
        str_cmd = (
            "mlflow:::mlflow_rfunc_predict(model_path = '{0}', input_path = {1}, "
            "output_path = {2}, content_type = {3})"
        )
        command = str_cmd.format(
            quote(model_path),
            _str_optional(input_path),
            _str_optional(output_path),
            _str_optional(content_type),
        )
        _execute(command)

    def serve(
        self,
        model_uri,
        port,
        host,
        timeout,
        enable_mlserver,
        synchronous=True,
        stdout=None,
        stderr=None,
    ):
        """
        Generate R model locally.

        NOTE: The `enable_mlserver` parameter is there to comply with the
        FlavorBackend interface but is not supported by MLServer yet.
        https://github.com/SeldonIO/MLServer/issues/183
        """
        if enable_mlserver:
            raise Exception("The MLServer inference server is not yet supported in the R backend.")

        if timeout:
            _logger.warning("Timeout is not yet supported in the R backend.")

        if not synchronous:
            raise Exception("RBackend does not support call with synchronous=False")

        if stdout is not None or stderr is not None:
            raise Exception("RBackend does not support redirect stdout/stderr.")

        model_path = _download_artifact_from_uri(model_uri)
        command = "mlflow::mlflow_rfunc_serve('{}', port = {}, host = '{}')".format(
            quote(model_path), port, host
        )
        _execute(command)

    def can_score_model(self):
        # `Rscript --version` writes to stderr in R < 4.2.0 but stdout in R >= 4.2.0.
        process = subprocess.Popen(
            ["Rscript", "--version"],
            close_fds=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, _ = process.communicate()
        if process.wait() != 0:
            return False

        version = self.version_pattern.search(stdout.decode("utf-8"))
        if not version:
            return False
        version = [int(x) for x in version.group(1).split(".")]
        return version[0] > 3 or version[0] == 3 and version[1] >= 3


def _execute(command):
    env = os.environ.copy()

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
    return "NULL" if s is None else f"'{quote(str(s))}'"
