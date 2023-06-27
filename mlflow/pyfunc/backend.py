import logging
import os
import pathlib
import subprocess
import posixpath
import sys
import warnings
import ctypes
import signal
from pathlib import Path

from mlflow.models import FlavorBackend
from mlflow.models.docker_utils import (
    _build_image,
    _generate_dockerfile_content,
    DISABLE_ENV_CREATION,
    SETUP_MINICONDA,
    SETUP_PYENV_AND_VIRTUALENV,
    _get_mlflow_install_step,
)
from mlflow.models.container import ENABLE_MLSERVER
from mlflow.pyfunc import ENV, scoring_server, mlserver, _extract_conda_env

from mlflow.utils.conda import get_or_create_conda_env, get_conda_bin_executable
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.file_utils import (
    path_to_local_file_uri,
    get_or_create_tmp_dir,
    get_or_create_nfs_tmp_dir,
)
from mlflow.utils.environment import Environment
from mlflow.utils.virtualenv import (
    _get_or_create_virtualenv,
    _get_pip_install_mlflow,
)
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.process import cache_return_value_per_process
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)

_IS_UNIX = os.name != "nt"
_STDIN_SERVER_SCRIPT = Path(__file__).parent.joinpath("stdin_server.py")


class PyFuncBackend(FlavorBackend):
    """
    Flavor backend implementation for the generic python models.
    """

    def __init__(
        self,
        config,
        workers=1,
        env_manager=_EnvManager.VIRTUALENV,
        install_mlflow=False,
        create_env_root_dir=False,
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
        if env_manager == _EnvManager.CONDA and ENV not in config:
            env_manager = _EnvManager.LOCAL
        self._env_manager = env_manager
        self._install_mlflow = install_mlflow
        self._env_id = os.environ.get("MLFLOW_HOME", VERSION) if install_mlflow else None
        self._create_env_root_dir = create_env_root_dir
        self._env_root_dir = env_root_dir
        self._environment = None

    def prepare_env(self, model_uri, capture_output=False):
        if self._environment is not None:
            return self._environment

        @cache_return_value_per_process
        def _get_or_create_env_root_dir(should_use_nfs):
            if should_use_nfs:
                root_tmp_dir = get_or_create_nfs_tmp_dir()
            else:
                root_tmp_dir = get_or_create_tmp_dir()

            env_root_dir = os.path.join(root_tmp_dir, "envs")
            os.makedirs(env_root_dir, exist_ok=True)
            return env_root_dir

        local_path = _download_artifact_from_uri(model_uri)
        if self._create_env_root_dir:
            if self._env_root_dir is not None:
                raise Exception("env_root_dir can not be set when create_env_root_dir=True")
            nfs_root_dir = get_nfs_cache_root_dir()
            env_root_dir = _get_or_create_env_root_dir(nfs_root_dir is not None)
        else:
            env_root_dir = self._env_root_dir

        if self._env_manager == _EnvManager.VIRTUALENV:
            activate_cmd = _get_or_create_virtualenv(
                local_path,
                self._env_id,
                env_root_dir=env_root_dir,
                capture_output=capture_output,
            )
            self._environment = Environment(activate_cmd)
        elif self._env_manager == _EnvManager.CONDA:
            conda_env_path = os.path.join(local_path, _extract_conda_env(self._config[ENV]))
            self._environment = get_or_create_conda_env(
                conda_env_path,
                env_id=self._env_id,
                capture_output=capture_output,
                env_root_dir=env_root_dir,
            )

        elif self._env_manager == _EnvManager.LOCAL:
            raise Exception("Prepare env should not be called with local env manager!")
        else:
            raise Exception(f"Unexpected env manager value '{self._env_manager}'")

        if self._install_mlflow:
            self._environment.execute(_get_pip_install_mlflow())
        else:
            self._environment.execute('python -c ""')

        return self._environment

    def predict(self, model_uri, input_path, output_path, content_type):
        """
        Generate predictions using generic python model saved with MLflow. The expected format of
        the input JSON is the Mlflow scoring format.
        Return the prediction results as a JSON.
        """
        local_path = _download_artifact_from_uri(model_uri)
        # NB: Absolute windows paths do not work with mlflow apis, use file uri to ensure
        # platform compatibility.
        local_uri = path_to_local_file_uri(local_path)

        if self._env_manager != _EnvManager.LOCAL:
            command = (
                'python -c "from mlflow.pyfunc.scoring_server import _predict; _predict('
                "model_uri={model_uri}, "
                "input_path={input_path}, "
                "output_path={output_path}, "
                "content_type={content_type})"
                '"'
            ).format(
                model_uri=repr(local_uri),
                input_path=repr(input_path),
                output_path=repr(output_path),
                content_type=repr(content_type),
            )
            return self.prepare_env(local_path).execute(command)
        else:
            scoring_server._predict(local_uri, input_path, output_path, content_type)

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

        if self._env_manager != _EnvManager.LOCAL:
            return self.prepare_env(local_path).execute(
                command,
                command_env,
                stdout=stdout,
                stderr=stderr,
                preexec_fn=setup_sigterm_on_parent_death,
                synchronous=synchronous,
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
                        f"Command '{command}' returned non zero return code. Return code = {rc}"
                    )
                return 0
            else:
                return child_proc

    def serve_stdin(
        self,
        model_uri,
        stdout=None,
        stderr=None,
    ):
        local_path = _download_artifact_from_uri(model_uri)
        return self.prepare_env(local_path).execute(
            command=f"python {_STDIN_SERVER_SCRIPT} --model-uri {local_path}",
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            synchronous=False,
        )

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

    def generate_dockerfile(
        self,
        model_uri,
        output_path="mlflow-dockerfile",
        install_mlflow=False,
        mlflow_home=None,
        enable_mlserver=False,
    ):
        copy_model_into_container = self.copy_model_into_container_wrapper(
            model_uri, install_mlflow, enable_mlserver
        )
        pyfunc_entrypoint = _pyfunc_entrypoint(
            self._env_manager, model_uri, install_mlflow, enable_mlserver
        )

        mlflow_home = os.path.abspath(mlflow_home) if mlflow_home else None

        is_conda = self._env_manager == _EnvManager.CONDA
        setup_miniconda = ""
        setup_pyenv_and_virtualenv = ""

        if is_conda:
            setup_miniconda = SETUP_MINICONDA
        else:
            setup_pyenv_and_virtualenv = SETUP_PYENV_AND_VIRTUALENV

        os.makedirs(output_path, exist_ok=True)

        _logger.debug("Created all folders in path", extra={"output_directory": output_path})
        install_mlflow = _get_mlflow_install_step(output_path, mlflow_home)

        custom_setup_steps = copy_model_into_container(output_path)

        dockerfile_text = _generate_dockerfile_content(
            setup_miniconda=setup_miniconda,
            setup_pyenv_and_virtualenv=setup_pyenv_and_virtualenv,
            install_mlflow=install_mlflow,
            custom_setup_steps=custom_setup_steps,
            entrypoint=pyfunc_entrypoint,
        )
        _logger.debug("generated dockerfile text", extra={"dockerfile": dockerfile_text})

        with open(os.path.join(output_path, "Dockerfile"), "w") as dockerfile:
            dockerfile.write(dockerfile_text)

    def build_image(
        self, model_uri, image_name, install_mlflow=False, mlflow_home=None, enable_mlserver=False
    ):
        copy_model_into_container = self.copy_model_into_container_wrapper(
            model_uri, install_mlflow, enable_mlserver
        )
        pyfunc_entrypoint = _pyfunc_entrypoint(
            self._env_manager, model_uri, install_mlflow, enable_mlserver
        )
        _build_image(
            image_name=image_name,
            mlflow_home=mlflow_home,
            env_manager=self._env_manager,
            custom_setup_steps_hook=copy_model_into_container,
            entrypoint=pyfunc_entrypoint,
        )

    def copy_model_into_container_wrapper(self, model_uri, install_mlflow, enable_mlserver):
        def copy_model_into_container(dockerfile_context_dir):
            # This function have to be included in another,
            # since `_build_image` function in `docker_utils` accepts only
            # single-argument function like this
            model_cwd = os.path.join(dockerfile_context_dir, "model_dir")
            pathlib.Path(model_cwd).mkdir(parents=True, exist_ok=True)
            if model_uri:
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
            else:
                return """
                    ENV {disable_env}="true"
                    ENV {ENABLE_MLSERVER}={enable_mlserver}
                    """.format(
                    disable_env=DISABLE_ENV_CREATION,
                    ENABLE_MLSERVER=ENABLE_MLSERVER,
                    enable_mlserver=repr(enable_mlserver),
                )

        return copy_model_into_container


def _pyfunc_entrypoint(env_manager, model_uri, install_mlflow, enable_mlserver):
    if model_uri:
        # The pyfunc image runs the same server as the Sagemaker image
        pyfunc_entrypoint = (
            'ENTRYPOINT ["python", "-c", "from mlflow.models import container as C;'
            f'C._serve({repr(env_manager)})"]'
        )
    else:
        entrypoint_code = "; ".join(
            [
                "from mlflow.models import container as C",
                "from mlflow.models.container import _install_pyfunc_deps",
                (
                    "_install_pyfunc_deps("
                    + '"/opt/ml/model", '
                    + f"install_mlflow={install_mlflow}, "
                    + f"enable_mlserver={enable_mlserver}, "
                    + f'env_manager="{env_manager}"'
                    + ")"
                ),
                f'C._serve("{env_manager}")',
            ]
        )
        pyfunc_entrypoint = 'ENTRYPOINT ["python", "-c", "{entrypoint_code}"]'.format(
            entrypoint_code=entrypoint_code.replace('"', '\\"')
        )

    return pyfunc_entrypoint
