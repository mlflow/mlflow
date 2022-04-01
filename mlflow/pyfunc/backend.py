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
from mlflow.utils.environment import EnvManager
from mlflow.version import VERSION


_logger = logging.getLogger(__name__)


class PyFuncBackend(FlavorBackend):
    """
    Flavor backend implementation for the generic python models.
    """

    def __init__(
        self, config, workers=1, env_manager=EnvManager.CONDA, install_mlflow=False, **kwargs
    ):
        super().__init__(config=config, **kwargs)
        self._nworkers = workers or 1
        self._env_manager = env_manager
        self._install_mlflow = install_mlflow
        self._env_id = os.environ.get("MLFLOW_HOME", VERSION) if install_mlflow else None

    def prepare_env(self, model_uri, capture_output=False):
        local_path = _download_artifact_from_uri(model_uri)
        if self._env_manager is EnvManager.LOCAL or ENV not in self._config:
            return 0
        conda_env_path = os.path.join(local_path, self._config[ENV])

        conda_env_name = get_or_create_conda_env(
            conda_env_path, env_id=self._env_id, capture_output=capture_output
        )

        command = 'python -c ""'
        return _execute_in_conda_env(conda_env_name, command, self._install_mlflow)

    def predict(self, model_uri, input_path, output_path, content_type, json_format):
        """
        Generate predictions using generic python model saved with MLflow.
        Return the prediction results as a JSON.
        """
        local_path = _download_artifact_from_uri(model_uri)
        # NB: Absolute windows paths do not work with mlflow apis, use file uri to ensure
        # platform compatibility.
        local_uri = path_to_local_file_uri(local_path)
        if self._env_manager is EnvManager.CONDA and ENV in self._config:
            conda_env_path = os.path.join(local_path, self._config[ENV])

            conda_env_name = get_or_create_conda_env(
                conda_env_path, env_id=self._env_id, capture_output=False
            )

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
            return _execute_in_conda_env(conda_env_name, command, self._install_mlflow)
        else:
            scoring_server._predict(local_uri, input_path, output_path, content_type, json_format)

    def serve(
        self,
        model_uri,
        port,
        host,
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
        command, command_env = server_implementation.get_cmd(local_path, port, host, self._nworkers)

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

        if self._env_manager is EnvManager.CONDA and ENV in self._config:
            conda_env_path = os.path.join(local_path, self._config[ENV])

            conda_env_name = get_or_create_conda_env(
                conda_env_path, env_id=self._env_id, capture_output=False
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
            )
        else:
            _logger.info("=== Running command '%s'", command)

            if os.name != "nt":
                command = ["bash", "-c", "exec " + command]

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
        if self._env_manager is EnvManager.LOCAL:
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
                    enable_mlserver={enable_mlserver})'
                ENV {disable_env}="true"
                ENV {ENABLE_MLSERVER}={enable_mlserver}
                """.format(
                disable_env=DISABLE_ENV_CREATION,
                model_dir=str(posixpath.join("model_dir", os.path.basename(model_path))),
                install_mlflow=repr(install_mlflow),
                ENABLE_MLSERVER=ENABLE_MLSERVER,
                enable_mlserver=repr(enable_mlserver),
            )

        # The pyfunc image runs the same server as the Sagemaker image
        pyfunc_entrypoint = (
            'ENTRYPOINT ["python", "-c", "from mlflow.models import container as C; C._serve()"]'
        )
        _build_image(
            image_name=image_name,
            mlflow_home=mlflow_home,
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
):
    """
    :param conda_env_path conda: conda environment file path
    :param command: command to run on the restored conda environment.
    :install_mlflow: whether to install mlflow
    :command_env: environment for child process.
    :param synchronous: If True, wait until server process exit and return 0, if process exit
                        with non-zero return code, raise exception.
                        If False, return the server process `Popen` instance immediately.
    :param stdout: Redirect server stdout
    :param stderr: Redirect server stderr
    """
    if command_env is None:
        command_env = os.environ

    # PIP_NO_INPUT=1 make pip run in non-interactive mode,
    # otherwise pip might prompt "yes or no" and ask stdin input
    command_env["PIP_NO_INPUT"] = "1"

    activate_conda_env = get_conda_command(conda_env_name)
    if install_mlflow:
        if "MLFLOW_HOME" in os.environ:  # dev version
            install_mlflow = "pip install -e {} 1>&2".format(os.environ["MLFLOW_HOME"])
        else:
            install_mlflow = "pip install mlflow=={} 1>&2".format(VERSION)

        activate_conda_env += [install_mlflow]
    if os.name != "nt":
        separator = " && "
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
    else:
        separator = " & "

    command = separator.join(activate_conda_env + [command])
    _logger.info("=== Running command '%s'", command)

    if os.name != "nt":
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


def _is_pyenv_available():
    return _run_command(["pyenv", "--version"], check=False).returncode == 0


def _validate_pyenv_is_available():
    if not _is_pyenv_available():
        raise Exception("pyenv must be available to use this feature")


def _is_virtualenv_available():
    return _run_command(["virtualenv", "--version"], check=False).returncode == 0


def _validate_virtualenv_is_available():
    if not _is_virtualenv_available():
        raise Exception("virtualenv must be available to use this feature")


def _get_env_name_for_virtualenv(string):
    return "mlflow-" + hashlib.sha1(string.encode("utf-8")).hexdigest()


def _get_mlflow_env_root():
    return os.getenv(
        _MLFLOW_ENV_ROOT_ENV_VAR, os.path.join(os.path.expanduser("~"), ".mlflow", "envs")
    )


def _install_python(version):
    is_windows = _is_windows()
    _logger.info("Installing python %s", version)
    # pyenv-win doesn't support `--skip-existing` but its behavior is enabled by default
    # https://github.com/pyenv-win/pyenv-win/pull/314
    pyenv_install_options = () if is_windows else ("--skip-existing",)
    _run_command(["pyenv", "install", *pyenv_install_options, version])

    if is_windows:
        # pyenv-win doesn't provide the `pyenv root` command
        pyenv_root = os.getenv("PYENV_ROOT")
        if pyenv_root is None:
            raise MlflowException("Environment variable 'PYENV_ROOT' must be set")
        path_to_bin = ("python.exe",)
    else:
        pyenv_root = _get_stdout(["pyenv", "root"]).strip()
        path_to_bin = ("bin", "python")
    return Path(pyenv_root).joinpath("versions", version, *path_to_bin)


def _execute_in_virtualenv(local_model_path, command, install_mlflow, env_id, command_env=None):
    # Read environment information
    model_path = Path(local_model_path)
    python_env_file = model_path / _PYTHON_ENV_FILE_NAME
    if python_env_file.exists():
        python_env = PythonEnv.from_yaml(python_env_file)
    else:
        _logger.info(
            "Attempting to restore the environment using %s because this model is missing %s.",
            _CONDA_ENV_FILE_NAME,
            _PYTHON_ENV_FILE_NAME,
        )
        conda_yaml_file = model_path / _CONDA_ENV_FILE_NAME
        with open(conda_yaml_file) as f:
            conda_yaml = yaml.safe_load(f)

        if _contains_conda_packages(conda_yaml):
            raise MlflowException(
                "Failed to restore the environment with virtualenv because this model depends on "
                "conda packages."
            )

        python_env = PythonEnv.from_conda_yaml(conda_yaml_file)

    python_bin_path = _install_python(python_env.python)

    # Generate an environment name
    env_name = _get_env_name_for_virtualenv(str(python_env.to_dict()) + (env_id or ""))

    # Create an environment
    env_root = Path(_get_mlflow_env_root())
    env_root.mkdir(parents=True, exist_ok=True)
    env_dir = env_root / env_name
    env_exists = env_dir.exists()
    if not env_exists:
        _logger.info("Creating a new environment %s", env_dir)
        _run_command(["virtualenv", "--python", python_bin_path, env_dir])
    else:
        _logger.info("Environment %s already exists", env_dir)

    # Construct a command to activate the environment
    is_windows = _is_windows()
    paths = ("Scripts", "activate.bat") if is_windows else ("bin", "activate")
    activator = env_dir.joinpath(*paths)
    activate_cmd = activator if is_windows else f"source {activator}"

    # Install dependencies
    if not env_exists:
        try:
            _logger.info("Installing dependencies")
            for deps in filter(None, [python_env.build_dependencies, python_env.dependencies]):
                tmp_req_file = model_path / f"requirements.{uuid.uuid4().hex}.txt"
                tmp_req_file.write_text("\n".join(deps))
                _run_multiple_commands(
                    # In windows `pip install pip==x.y.z` causes the following error:
                    # `[WinError 5] Access is denied: 'C:\path\to\pip.exe`
                    # We can avoid this issue by using `python -m`.
                    [activate_cmd, f"python -m pip install -r {tmp_req_file}"],
                    # Run `pip install` in the model directory to refer to the requirements
                    # file correctly
                    cwd=model_path,
                )
                tmp_req_file.unlink()
        except:
            _logger.warning(
                "Encountered an unexpected error while installing dependencies. Removing %s",
                env_dir,
            )
            shutil.rmtree(env_dir, ignore_errors=True)
            raise

    # Run the command
    commands = [activate_cmd]
    if install_mlflow:
        commands.append(_get_install_mlflow_command())
    _run_multiple_commands([*commands, command], env=command_env)
