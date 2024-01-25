import ctypes
import logging
import os
import pathlib
import posixpath
import shlex
import signal
import subprocess
import sys
import warnings
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.models import FlavorBackend, docker_utils
from mlflow.pyfunc import (
    ENV,
    _extract_conda_env,
    _mlflow_pyfunc_backend_predict,
    mlserver,
    scoring_server,
)
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.conda import get_conda_bin_executable, get_or_create_conda_env
from mlflow.utils.environment import Environment
from mlflow.utils.file_utils import (
    TempDir,
    get_or_create_nfs_tmp_dir,
    get_or_create_tmp_dir,
    path_to_local_file_uri,
)
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.process import ShellCommandException, cache_return_value_per_process
from mlflow.utils.virtualenv import (
    _get_or_create_virtualenv,
    _get_pip_install_mlflow,
)
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

    def prepare_env(self, model_uri, capture_output=False, pip_requirements_override=None):
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
                pip_requirements_override=pip_requirements_override,
            )
            self._environment = Environment(activate_cmd)
        elif self._env_manager == _EnvManager.CONDA:
            conda_env_path = os.path.join(local_path, _extract_conda_env(self._config[ENV]))
            self._environment = get_or_create_conda_env(
                conda_env_path,
                env_id=self._env_id,
                capture_output=capture_output,
                env_root_dir=env_root_dir,
                pip_requirements_override=pip_requirements_override,
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

    def predict(
        self,
        model_uri,
        input_path,
        output_path,
        content_type,
        pip_requirements_override=None,
    ):
        """
        Generate predictions using generic python model saved with MLflow. The expected format of
        the input JSON is the MLflow scoring format.
        Return the prediction results as a JSON.
        """
        local_path = _download_artifact_from_uri(model_uri)
        # NB: Absolute windows paths do not work with mlflow apis, use file uri to ensure
        # platform compatibility.
        local_uri = path_to_local_file_uri(local_path)

        if self._env_manager != _EnvManager.LOCAL:
            predict_cmd = [
                "python",
                _mlflow_pyfunc_backend_predict.__file__,
                "--model-uri",
                str(local_uri),
                "--content-type",
                shlex.quote(str(content_type)),
            ]
            if input_path:
                predict_cmd += ["--input-path", shlex.quote(str(input_path))]
            if output_path:
                predict_cmd += ["--output-path", shlex.quote(str(output_path))]

            if pip_requirements_override and self._env_manager == _EnvManager.CONDA:
                # Conda use = instead of == for version pinning
                pip_requirements_override = [
                    l.replace("==", "=") for l in pip_requirements_override
                ]

            environment = self.prepare_env(
                local_path, pip_requirements_override=pip_requirements_override
            )

            try:
                environment.execute(" ".join(predict_cmd))
            except ShellCommandException as e:
                raise MlflowException(
                    f"{e}\n\nAn exception occurred while running model prediction within a "
                    f"{self._env_manager} environment. You can find the error message "
                    f"from the prediction subprocess by scrolling above."
                ) from None
        else:
            if pip_requirements_override:
                raise MlflowException(
                    "`pip_requirements_override` is not supported for local env manager."
                    "Please use conda or virtualenv instead."
                )
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
                    warnings.warn(f"Setup libc.prctl PR_SET_PDEATHSIG failed, error {e!r}.")

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
        self, model_uri, output_dir, install_mlflow=False, mlflow_home=None, enable_mlserver=False
    ):
        os.makedirs(output_dir, exist_ok=True)
        _logger.debug("Created all folders in path", extra={"output_directory": output_dir})

        # Copy model to image if model_uri is specified
        custom_setup_steps = (
            self._get_copy_model_steps(output_dir, model_uri, install_mlflow, enable_mlserver)
            if model_uri
            else ""
        )

        pyfunc_entrypoint = self._pyfunc_entrypoint(model_uri, install_mlflow, enable_mlserver)

        dockerfile_text = docker_utils.generate_dockerfile(
            output_dir=output_dir,
            custom_setup_steps=custom_setup_steps,
            entrypoint=pyfunc_entrypoint,
            env_manager=self._env_manager,
            mlflow_home=mlflow_home,
            enable_mlserver=enable_mlserver,
            disable_env_creation=True,  # Always disable env creation for pyfunc
        )
        _logger.debug("generated dockerfile at {output_dir}", extra={"dockerfile": dockerfile_text})

    def build_image(
        self, model_uri, image_name, install_mlflow=False, mlflow_home=None, enable_mlserver=False
    ):
        with TempDir() as tmp:
            cwd = tmp.path()
            self.generate_dockerfile(model_uri, cwd, install_mlflow, mlflow_home, enable_mlserver)

            _logger.info("Building docker image with name %s", image_name)
            docker_utils.build_image_from_context(context_dir=cwd, image_name=image_name)

    def _get_copy_model_steps(self, output_dir, model_uri, install_mlflow, enable_mlserver):
        model_cwd = os.path.join(output_dir, "model_dir")
        pathlib.Path(model_cwd).mkdir(parents=True, exist_ok=True)

        # If model_uri is specified, copy the model to the image and install its dependencies
        model_path = _download_artifact_from_uri(model_uri, output_path=model_cwd)
        model_dir = str(posixpath.join("model_dir", os.path.basename(model_path)))

        install_deps_cmd = self._get_install_pyfunc_deps_cmd(install_mlflow, enable_mlserver)
        return f'COPY {model_dir} /opt/ml/model\nRUN python -c "{install_deps_cmd}"'

    def _pyfunc_entrypoint(self, model_uri, install_mlflow, enable_mlserver):
        if model_uri:
            # If model_uri is specified, dependencies are installed at build time so we don't
            # need to run the install command at runtime
            install_deps_cmd = ""
        else:
            install_deps_cmd = self._get_install_pyfunc_deps_cmd(install_mlflow, enable_mlserver)
        entrypoint = (
            f"from mlflow.models import container as C;{install_deps_cmd} "
            f"C._serve('{self._env_manager}')"
        )
        return f'ENTRYPOINT ["python", "-c", "{entrypoint}"]'

    def _get_install_pyfunc_deps_cmd(self, install_mlflow, enable_mlserver):
        return (
            "from mlflow.models.container import _install_pyfunc_deps; "
            f"_install_pyfunc_deps('/opt/ml/model', install_mlflow={install_mlflow}, "
            f"enable_mlserver={enable_mlserver}, env_manager='{self._env_manager}');"
        )
