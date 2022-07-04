import logging
import os
import subprocess
import posixpath
import sys
import warnings


from mlflow.models import FlavorBackend
from mlflow.models.docker_utils import _build_image, DISABLE_ENV_CREATION
from mlflow.models.container import ENABLE_MLSERVER
from mlflow.pyfunc import ENV, scoring_server, mlserver

from mlflow.utils.conda import get_or_create_conda_env, get_conda_bin_executable, get_conda_command
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.conda import _get_conda_extra_env_vars
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.virtualenv import (
    _get_or_create_virtualenv,
    _execute_in_virtualenv,
    _get_pip_install_mlflow,
)
from mlflow.version import VERSION


_logger = logging.getLogger(__name__)

_IS_UNIX = os.name != "nt"


class PyFuncBackend(FlavorBackend):
    """
    Flavor backend implementation for the generic python models.
    """

    def __init__(
        self,
        config,
        workers=1,
        env_manager=_EnvManager.CONDA,
        install_mlflow=False,
        env_root_dir=None,
        **kwargs,
    ):
        """
        :param env_root_dir: Root path for conda env. If None, use Conda's default environments
                             directory. Note if this is set, conda package cache path becomes
                             "{env_root_dir}/conda_cache_pkgs" instead of the global package cache
                             path, and pip package cache path becomes
                             "{env_root_dir}/pip_cache_pkgs" instead of the global package cache
                             path.
        """
        super().__init__(config=config, **kwargs)
        self._nworkers = workers or 1
        self._env_manager = env_manager
        self._install_mlflow = install_mlflow
        self._env_id = os.environ.get("MLFLOW_HOME", VERSION) if install_mlflow else None
        self._env_root_dir = env_root_dir

    def prepare_env(self, model_uri, capture_output=False):
        local_path = _download_artifact_from_uri(model_uri)

        command = 'python -c ""'
        if self._env_manager == _EnvManager.VIRTUALENV:
            activate_cmd = _get_or_create_virtualenv(
                local_path,
                self._env_id,
                env_root_dir=self._env_root_dir,
                capture_output=capture_output,
            )
            return _execute_in_virtualenv(
                activate_cmd,
                command,
                self._install_mlflow,
                env_root_dir=self._env_root_dir,
                capture_output=capture_output,
            )
        elif self._env_manager == _EnvManager.LOCAL or ENV not in self._config:
            return 0

        conda_env_path = os.path.join(local_path, self._config[ENV])
        conda_env_name = get_or_create_conda_env(
            conda_env_path,
            env_id=self._env_id,
            capture_output=capture_output,
            env_root_dir=self._env_root_dir,
        )

        command = 'python -c ""'
        return _execute_in_conda_env(
            conda_env_name, command, self._install_mlflow, env_root_dir=self._env_root_dir
        )

    def predict(self, model_uri, input_path, output_path, content_type, json_format):
        """
        Generate predictions using generic python model saved with MLflow.
        Return the prediction results as a JSON.
        """
        local_path = _download_artifact_from_uri(model_uri)
        # NB: Absolute windows paths do not work with mlflow apis, use file uri to ensure
        # platform compatibility.
        local_uri = path_to_local_file_uri(local_path)
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
        if self._env_manager == _EnvManager.CONDA and ENV in self._config:
            conda_env_path = os.path.join(local_path, self._config[ENV])
            conda_env_name = get_or_create_conda_env(
                conda_env_path,
                env_id=self._env_id,
                capture_output=False,
                env_root_dir=self._env_root_dir,
            )
            return _execute_in_conda_env(
                conda_env_name, command, self._install_mlflow, env_root_dir=self._env_root_dir
            )
        elif self._env_manager == _EnvManager.VIRTUALENV:
            activate_cmd = _get_or_create_virtualenv(
                local_path, self._env_id, env_root_dir=self._env_root_dir
            )
            return _execute_in_virtualenv(
                activate_cmd, command, self._install_mlflow, env_root_dir=self._env_root_dir
            )
        else:
            scoring_server._predict(local_uri, input_path, output_path, content_type, json_format)

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
    ):  # pylint: disable=W0221
        """
        Serve pyfunc model locally.
        """
        local_path = _download_artifact_from_uri(model_uri)

        server_implementation = mlserver if enable_mlserver else scoring_server
        command, command_env = server_implementation.get_cmd(
            local_path, port, host, timeout, self._nworkers
        )

        if sys.platform.startswith("linux"):

            def setup_sigterm_on_parent_death():
                """
                Uses prctl to automatically send SIGTERM to the command process when its parent is
                dead.

                This handles the case when the parent is a PySpark worker process.
                If a user cancels the PySpark job, the worker process gets killed, regardless of
                PySpark daemon and worker reuse settings.
                We use prctl to ensure the command process receives SIGTERM after spark job
                cancellation.
                The command process itself should handle SIGTERM properly.
                This is a no-op on macOS because prctl is not supported.

                Note:
                When a pyspark job canceled, the UDF python process are killed by signal "SIGKILL",
                This case neither "atexit" nor signal handler can capture SIGKILL signal.
                prctl is the only way to capture SIGKILL signal.
                """
                try:
                    import ctypes
                    import signal

                    libc = ctypes.CDLL("libc.so.6")
                    # Set the parent process death signal of the command process to SIGTERM.
                    libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG, see prctl.h
                except OSError as e:
                    # TODO: find approach for supporting MacOS/Windows system which does
                    #  not support prctl.
                    warnings.warn(f"Setup libc.prctl PR_SET_PDEATHSIG failed, error {repr(e)}.")

        else:
            setup_sigterm_on_parent_death = None

        if _IS_UNIX:
            # Add "exec" before the starting scoring server command, so that the scoring server
            # process replaces the bash process, otherwise the scoring server process is created
            # as a child process of the bash process.
            # Note we in `mlflow.pyfunc.spark_udf`, use prctl PR_SET_PDEATHSIG to ensure scoring
            # server process being killed when UDF process exit. The PR_SET_PDEATHSIG can only
            # send signal to the bash process, if the scoring server process is created as a
            # child process of the bash process, then it cannot receive the signal sent by prctl.
            # TODO: For Windows, there's no equivalent things of Unix shell's exec. Windows also
            #  does not support prctl. We need to find an approach to address it.
            command = "exec " + command

        if self._env_manager == _EnvManager.CONDA and ENV in self._config:
            conda_env_path = os.path.join(local_path, self._config[ENV])

            conda_env_name = get_or_create_conda_env(
                conda_env_path,
                env_id=self._env_id,
                capture_output=False,
                env_root_dir=self._env_root_dir,
            )

            child_proc = _execute_in_conda_env(
                conda_env_name,
                command,
                self._install_mlflow,
                command_env=command_env,
                synchronous=False,
                preexec_fn=setup_sigterm_on_parent_death,
                stdout=stdout,
                stderr=stderr,
                env_root_dir=self._env_root_dir,
            )
        elif self._env_manager == _EnvManager.VIRTUALENV:
            activate_cmd = _get_or_create_virtualenv(
                local_path, self._env_id, env_root_dir=self._env_root_dir
            )
            child_proc = _execute_in_virtualenv(
                activate_cmd,
                command,
                self._install_mlflow,
                command_env=command_env,
                capture_output=False,
                synchronous=False,
                env_root_dir=self._env_root_dir,
                preexec_fn=setup_sigterm_on_parent_death,
                stdout=stdout,
                stderr=stderr,
            )
        else:
            _logger.info("=== Running command '%s'", command)

            if os.name != "nt":
                command = ["bash", "-c", command]

            child_proc = subprocess.Popen(
                command,
                env=command_env,
                preexec_fn=setup_sigterm_on_parent_death,
                stdout=stdout,
                stderr=stderr,
            )

        if synchronous:
            rc = child_proc.wait()
            if rc != 0:
                raise Exception(
                    "Command '{0}' returned non zero return code. Return code = {1}".format(
                        command, rc
                    )
                )
            return 0
        else:
            return child_proc

    def can_score_model(self):
        if self._env_manager == _EnvManager.LOCAL:
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
                    enable_mlserver={enable_mlserver}, \
                    env_manager="{env_manager}")'
                ENV {disable_env}="true"
                ENV {ENABLE_MLSERVER}={enable_mlserver}
                """.format(
                disable_env=DISABLE_ENV_CREATION,
                model_dir=str(posixpath.join("model_dir", os.path.basename(model_path))),
                install_mlflow=repr(install_mlflow),
                ENABLE_MLSERVER=ENABLE_MLSERVER,
                enable_mlserver=repr(enable_mlserver),
                env_manager=self._env_manager,
            )

        # The pyfunc image runs the same server as the Sagemaker image
        pyfunc_entrypoint = (
            'ENTRYPOINT ["python", "-c", "from mlflow.models import container as C;'
            f'C._serve({repr(self._env_manager)})"]'
        )
        _build_image(
            image_name=image_name,
            mlflow_home=mlflow_home,
            env_manager=self._env_manager,
            custom_setup_steps_hook=copy_model_into_container,
            entrypoint=pyfunc_entrypoint,
        )


def _execute_in_conda_env(
    conda_env_name,
    command,
    install_mlflow,
    command_env=None,
    synchronous=True,
    preexec_fn=None,
    stdout=None,
    stderr=None,
    env_root_dir=None,
):
    """
    :param conda_env_path conda: conda environment file path
    :param command: command to run on the restored conda environment.
    :param install_mlflow: whether to install mlflow
    :param command_env: environment for child process.
    :param synchronous: If True, wait until server process exit and return 0, if process exit
                        with non-zero return code, raise exception.
                        If False, return the server process `Popen` instance immediately.
    :param stdout: Redirect server stdout
    :param stderr: Redirect server stderr
    :param env_root_dir: See doc of PyFuncBackend constructor argument `env_root_dir`.
    """
    if command_env is None:
        command_env = os.environ.copy()

    if env_root_dir is not None:
        command_env = {**command_env, **_get_conda_extra_env_vars(env_root_dir)}

    activate_conda_env = get_conda_command(conda_env_name)
    if install_mlflow:
        pip_install_mlflow = _get_pip_install_mlflow()
        activate_conda_env += [pip_install_mlflow]
    if _IS_UNIX:
        separator = " && "
    else:
        separator = " & "

    command = separator.join(activate_conda_env + [command])
    _logger.info("=== Running command '%s'", command)

    if _IS_UNIX:
        child = subprocess.Popen(
            ["bash", "-c", command],
            close_fds=True,
            env=command_env,
            preexec_fn=preexec_fn,
            stdout=stdout,
            stderr=stderr,
        )
    else:
        child = subprocess.Popen(
            ["cmd", "/c", command],
            close_fds=True,
            env=command_env,
            preexec_fn=preexec_fn,
            stdout=stdout,
            stderr=stderr,
        )

    if synchronous:
        rc = child.wait()
        if rc != 0:
            raise Exception(
                "Command '{0}' returned non zero return code. Return code = {1}".format(command, rc)
            )
        return 0
    else:
        return child
