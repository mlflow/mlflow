import logging
import os
import re
import subprocess
from six.moves import shlex_quote
import shlex
import sys

import mlflow
from mlflow.pyfunc import ENV, scoring_server
from mlflow.models import FlavorBackend
from mlflow.utils.process import exec_cmd
from mlflow.utils.file_utils import TempDir
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.projects import _get_or_create_conda_env, _get_conda_bin_executable


_logger = logging.getLogger(__name__)


class PyFuncBackend(FlavorBackend):
    """
        Flavor backend implementation for the generic python models.
    """

    def __init__(self, config, workers=1, gunicorn_opts=None, no_conda=False, **kwargs):
        super(PyFuncBackend, self).__init__(config=config, **kwargs)
        self._no_conda = no_conda
        self._workers = workers
        self._gunicorn_opts = gunicorn_opts

    def predict(self, model_uri, input_path, output_path, content_type, json_format):
        """
        Generate predictions using generic python model saved with MLflow.
        Return the prediction results as a JSON.
        """
        with TempDir() as tmp:
            local_path = _download_artifact_from_uri(model_uri, output_path=tmp.path())
            if not self._no_conda and ENV in self._config:
                conda_env_path = os.path.join(local_path, self._config[ENV])

                command = ('python -c "import sys; from mlflow.pyfunc import scoring_server;'
                           'scoring_server._predict({local_path}, {input_path}, {output_path}, '
                           '{content_type}, {json_format})"').format(
                    local_path=repr(shlex_quote(local_path)),
                    input_path="None" if input_path is None else repr(shlex_quote(input_path)),
                    output_path="None" if output_path is None else repr(
                        shlex_quote(output_path)),
                    content_type=repr(shlex_quote(content_type)),
                    json_format=repr(shlex_quote(json_format)))
                command = _execute_in_conda_env(conda_env_path, command)
                _logger.info("=== Running command '%s'", command)
                p = subprocess.Popen(command,
                                     stdin=sys.stdin,
                                     stdout=sys.stdout,
                                     stderr=sys.stderr)
                rc = p.wait()
                if rc != 0:
                    raise Exception("Command returned non-zero exitcode: %s" % rc)
            else:
                scoring_server._predict(local_path, input_path, output_path, content_type,
                                        json_format)

    def serve(self, model_uri, port, host):
        """
        Serve pyfunc model locally.
        """
        with TempDir() as tmp:
            local_path = shlex_quote(_download_artifact_from_uri(model_uri, output_path=tmp.path()))
            env_map = {scoring_server.MLFLOW_MODEL_PATH: local_path}
            opts = shlex.split(self._gunicorn_opts) if self._gunicorn_opts else []
            opts = " ".join(opts)
            serve_command = ("gunicorn {opts} -b {bind_address} "
                             "-w {workers} mlflow.pyfunc.scoring_server.wsgi:app").format(
                opts=opts, bind_address="%s:%s" % (host, port), workers=self._workers)
            if not self._no_conda and ENV in self._config:
                conda_env_path = os.path.join(local_path, self._config[ENV])
                serve_command = _execute_in_conda_env(conda_env_path, serve_command)

            _logger.info("=== Running command '%s'", serve_command)
            exec_cmd(serve_command, env=env_map, stream_output=True)

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


# gunicorn requires extra deps for the following worker classes
_worker_class_pattern = re.compile(
    "gunicorn .* (-k|--worker-class)[ ]+(eventlet|gevent|tornado|gthread)")


def _execute_in_conda_env(conda_env_path, command):
    conda_env_name = _get_or_create_conda_env(conda_env_path)
    activate_path = _get_conda_bin_executable("activate")
    from mlflow.version import VERSION
    install_mlflow = "pip install -U mlflow>={} 1>&2".format(VERSION)

    if VERSION.endswith("dev0"):
        install_mlflow = "pip install -e {} 1>&2".format(os.path.dirname(mlflow.__path__[0]))

    worker_class = re.search(_worker_class_pattern, command)
    if worker_class:
        install_mlflow = install_mlflow + " && pip install gunicorn[{}] 1>&2".format(
            worker_class.group(2))

    command = " && ".join(
        ["source {} {}".format(activate_path, conda_env_name),
         "{} ".format(install_mlflow), command]
    )
    return ["bash", "-c", command]
