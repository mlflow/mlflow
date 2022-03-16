import logging
import os
import hashlib
import shutil
from pathlib import Path
import uuid

import yaml
import subprocess
import posixpath
from mlflow.models import FlavorBackend
from mlflow.models.docker_utils import _build_image, DISABLE_ENV_CREATION
from mlflow.models.container import ENABLE_MLSERVER
from mlflow.pyfunc import ENV, scoring_server, mlserver
from mlflow.exceptions import MlflowException

from mlflow.utils.conda import (
    get_or_create_conda_env,
    get_conda_bin_executable,
    get_conda_command,
)
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.environment import (
    _PYTHON_ENV_FILE_NAME,
    _CONDA_ENV_FILE_NAME,
    PythonEnv,
    EnvManager,
    _contains_conda_packages,
)
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)

_MLFLOW_HOME_ENV_VAR = "MLFLOW_HOME"
_MLFLOW_ENV_ROOT_ENV_VAR = "MLFLOW_ENV_ROOT"


def _is_windows():
    return os.name == "nt"


def _get_and_operator():
    return " & " if _is_windows() else " && "


def _join(args, sep=" "):
    return sep.join(map(str, args))


def _get_entrypoint():
    return ["cmd", "/c"] if _is_windows() else ["bash", "-c"]


def _run_command(args, check=True, **kwargs):
    return subprocess.run([*_get_entrypoint(), _join(args)], check=check, **kwargs)


def _run_multiple_commands(commands, **kwargs):
    return _run_command([_join(commands, sep=_get_and_operator())], **kwargs)


def _get_stdout(args, **kwargs):
    prc = _run_command(args, check=False, capture_output=True, **kwargs)
    stdout = prc.stdout.decode("utf-8")
    stderr = prc.stderr.decode("utf-8")
    if prc.returncode != 0:
        raise Exception(
            "\n".join(
                [
                    f"Command {args} returned non-zero exit status {prc.returncode}",
                    "===== stdout =====",
                    stdout,
                    "===== stderr =====",
                    stderr,
                ]
            )
        )
    return stdout


def _get_install_mlflow_command():
    mlflow_home = os.getenv(_MLFLOW_HOME_ENV_VAR)
    return (
        "pip install -e {} 1>&2".format(mlflow_home)  # dev version
        if mlflow_home
        else "pip install mlflow=={} 1>&2".format(VERSION)
    )


class PyFuncBackend(FlavorBackend):
    """
    Flavor backend implementation for the generic python models.
    """

    def __init__(
        self,
        config,
        workers=1,
        env_manager=None,
        install_mlflow=False,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self._nworkers = workers or 1
        self._env_manager = env_manager
        self._install_mlflow = install_mlflow

    def prepare_env(self, model_uri):
        if self._should_use_local():
            return 0

        local_path = _download_artifact_from_uri(model_uri)
        command = 'python -c ""'
        if self._should_use_conda():
            conda_env_path = os.path.join(local_path, self._config[ENV])
            return _execute_in_conda_env(
                conda_env_path,
                command,
                self._install_mlflow,
                self._get_env_id(),
            )
        if self._should_use_virtualenv():
            return _execute_in_virtualenv(
                local_path,
                command,
                self._install_mlflow,
                self._get_env_id(),
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
        if self._should_use_local():
            return scoring_server._predict(
                local_uri, input_path, output_path, content_type, json_format
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
        if self._should_use_conda():
            conda_env_path = os.path.join(local_path, self._config[ENV])
            return _execute_in_conda_env(
                conda_env_path,
                command,
                self._install_mlflow,
                self._get_env_id(),
            )
        elif self._should_use_virtualenv():
            return _execute_in_virtualenv(
                local_path,
                command,
                self._install_mlflow,
                self._get_env_id(),
            )

    def _get_env_id(self):
        return os.environ.get("MLFLOW_HOME", VERSION) if self._install_mlflow else None

    def _should_use_conda(self):
        return self._env_manager is EnvManager.CONDA and ENV in self._config

    def _should_use_virtualenv(self):
        return self._env_manager is EnvManager.VIRTUALENV

    def _should_use_local(self):
        return self._env_manager is EnvManager.LOCAL

    def serve(self, model_uri, port, host, enable_mlserver):  # pylint: disable=W0221
        """
        Serve pyfunc model locally.
        """
        local_path = _download_artifact_from_uri(model_uri)

        server_implementation = mlserver if enable_mlserver else scoring_server
        command, command_env = server_implementation.get_cmd(local_path, port, host, self._nworkers)
        env_id = self._get_env_id()

        if self._should_use_virtualenv():
            _validate_pyenv_is_available()
            _validate_virtualenv_is_available()
            return _execute_in_virtualenv(
                local_path,
                command,
                self._install_mlflow,
                self._get_env_id(),
                command_env=command_env,
            )
        elif self._should_use_conda():
            conda_env_path = os.path.join(local_path, self._config[ENV])
            return _execute_in_conda_env(
                conda_env_path, command, self._install_mlflow, env_id, command_env=command_env
            )
        else:
            _logger.info("=== Running command '%s'", command)
            if os.name != "nt":
                subprocess.Popen(["bash", "-c", command], env=command_env).wait()
            else:
                subprocess.Popen(command, env=command_env).wait()

    def can_score_model(self):
        if self._no_conda:
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


def _execute_in_conda_env(conda_env_path, command, install_mlflow, env_id, command_env=None):
    if command_env is None:
        command_env = os.environ
    conda_env_name = get_or_create_conda_env(conda_env_path, env_id=env_id)
    activate_conda_env = get_conda_command(conda_env_name)
    if install_mlflow:
        activate_conda_env += [_get_install_mlflow_command()]
    _logger.info("=== Running command '%s'", command)
    _run_multiple_commands(
        [
            *activate_conda_env,
            command,
        ],
        env=command_env,
    )


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
                    [activate_cmd, f"pip install -r {tmp_req_file}"],
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
