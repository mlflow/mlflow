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

from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import FlavorBackend, Model, docker_utils
from mlflow.models.docker_utils import PYTHON_SLIM_BASE_IMAGE, UBUNTU_BASE_IMAGE
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.pyfunc import (
    ENV,
    _extract_conda_env,
    _mlflow_pyfunc_backend_predict,
    mlserver,
    scoring_server,
)
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import env_manager as em
from mlflow.utils.conda import get_conda_bin_executable, get_or_create_conda_env
from mlflow.utils.environment import Environment, _PythonEnv
from mlflow.utils.file_utils import (
    TempDir,
    get_or_create_nfs_tmp_dir,
    get_or_create_tmp_dir,
    path_to_local_file_uri,
)
from mlflow.utils.model_utils import _get_all_flavor_configurations
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.os import is_windows
from mlflow.utils.process import ShellCommandException, cache_return_value_per_process
from mlflow.utils.virtualenv import (
    _get_or_create_virtualenv,
    _get_pip_install_mlflow,
)
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)

_STDIN_SERVER_SCRIPT = Path(__file__).parent.joinpath("stdin_server.py")

# Flavors that require Java to be installed in the environment
JAVA_FLAVORS = {"johnsnowlabs", "h2o", "mleap", "spark"}

# Some flavor requires additional packages to be installed in the environment
FLAVOR_SPECIFIC_APT_PACKAGES = {
    "lightgbm": ["libgomp1"],
    "paddle": ["libgomp1"],
}

# Directory to store loaded model inside the Docker context directory
_MODEL_DIR_NAME = "model_dir"
LOCAL_ENV_MANAGER_ERROR_MESSAGE = "We cannot use 'LOCAL' environment manager "
"for your model configuration. Please specify a virtualenv or conda environment "
"manager instead with `--env-manager` argument."


class PyFuncBackend(FlavorBackend):
    """
    Flavor backend implementation for the generic python models.
    """

    def __init__(
        self,
        config,
        env_manager,
        workers=1,
        install_mlflow=False,
        create_env_root_dir=False,
        env_root_dir=None,
        **kwargs,
    ):
        """
        Args:
            env_manager: Environment manager to use for preparing the environment. If None,
                MLflow will automatically pick the env manager based on the model's flavor
                configuration for generate_dockerfile. It can't be None for other methods.
            env_root_dir: Root path for conda env. If None, use Conda's default environments
                directory. Note if this is set, conda package cache path becomes
                "{env_root_dir}/conda_cache_pkgs" instead of the global package cache
                path, and pip package cache path becomes
                "{env_root_dir}/pip_cache_pkgs" instead of the global package cache
                path.
        """
        super().__init__(config=config, **kwargs)
        self._nworkers = workers or 1
        if env_manager == em.CONDA and ENV not in config:
            warnings.warn(
                "Conda environment is not specified in config `env`. Using local environment."
            )
            env_manager = em.LOCAL
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

            envs_root_dir = os.path.join(root_tmp_dir, "envs")
            os.makedirs(envs_root_dir, exist_ok=True)
            return envs_root_dir

        local_path = _download_artifact_from_uri(model_uri)
        if self._create_env_root_dir:
            if self._env_root_dir is not None:
                raise Exception("env_root_dir can not be set when create_env_root_dir=True")
            nfs_root_dir = get_nfs_cache_root_dir()
            env_root_dir = _get_or_create_env_root_dir(nfs_root_dir is not None)
        else:
            env_root_dir = self._env_root_dir

        if self._env_manager == em.VIRTUALENV:
            activate_cmd = _get_or_create_virtualenv(
                local_path,
                self._env_id,
                env_root_dir=env_root_dir,
                capture_output=capture_output,
                pip_requirements_override=pip_requirements_override,
            )
            self._environment = Environment(activate_cmd)
        elif self._env_manager == em.CONDA:
            conda_env_path = os.path.join(local_path, _extract_conda_env(self._config[ENV]))
            self._environment = get_or_create_conda_env(
                conda_env_path,
                env_id=self._env_id,
                capture_output=capture_output,
                env_root_dir=env_root_dir,
                pip_requirements_override=pip_requirements_override,
            )

        elif self._env_manager == em.LOCAL:
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

        if self._env_manager != em.LOCAL:
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

            if pip_requirements_override and self._env_manager == em.CONDA:
                # Conda use = instead of == for version pinning
                pip_requirements_override = [
                    pip_req.replace("==", "=") for pip_req in pip_requirements_override
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

        if not is_windows():
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

        if self._env_manager != em.LOCAL:
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

            if not is_windows():
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
        if self._env_manager == em.LOCAL:
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
        self,
        model_uri,
        image_name,
        install_java=False,
        install_mlflow=False,
        mlflow_home=None,
        enable_mlserver=False,
    ):
        with TempDir() as tmp:
            cwd = tmp.path()
            self.generate_dockerfile(
                model_uri=model_uri,
                output_dir=cwd,
                install_java=install_java,
                install_mlflow=install_mlflow,
                mlflow_home=mlflow_home,
                enable_mlserver=enable_mlserver,
            )

            _logger.info("Building docker image with name %s", image_name)
            docker_utils.build_image_from_context(context_dir=cwd, image_name=image_name)

    def generate_dockerfile(
        self,
        model_uri,
        output_dir,
        install_java=False,
        install_mlflow=False,
        mlflow_home=None,
        enable_mlserver=False,
    ):
        os.makedirs(output_dir, exist_ok=True)
        _logger.debug("Created all folders in path", extra={"output_directory": output_dir})

        if model_uri:
            model_cwd = os.path.join(output_dir, _MODEL_DIR_NAME)
            pathlib.Path(model_cwd).mkdir(parents=True, exist_ok=True)
            model_path = _download_artifact_from_uri(model_uri, output_path=model_cwd)
            base_image = self._get_base_image(model_path, install_java)

            if base_image.startswith("python"):
                # we can directly use local env for python image
                env_manager = self._env_manager or em.LOCAL
                if env_manager in [em.CONDA, em.VIRTUALENV]:
                    # we can directly use ubuntu image for conda and virtualenv
                    base_image = UBUNTU_BASE_IMAGE
            elif base_image == UBUNTU_BASE_IMAGE:
                env_manager = self._env_manager or em.VIRTUALENV
                # installing python on ubuntu image is problematic and not recommended officially
                # , so we recommend using conda or virtualenv instead on ubuntu image
                if env_manager == em.LOCAL:
                    raise MlflowException.invalid_parameter_value(LOCAL_ENV_MANAGER_ERROR_MESSAGE)
            # shouldn't reach here but add this so we can validate base_image value above
            else:
                raise MlflowException(f"Unexpected base image value '{base_image}'")

            model_install_steps = self._model_installation_steps(
                model_path, env_manager, install_mlflow, enable_mlserver
            )
            entrypoint = f"from mlflow.models import container as C; C._serve('{env_manager}')"

        # if no model_uri specified, user must use virtualenv or conda env based on ubuntu image
        else:
            base_image = UBUNTU_BASE_IMAGE
            env_manager = self._env_manager or em.VIRTUALENV
            if env_manager == em.LOCAL:
                raise MlflowException.invalid_parameter_value(LOCAL_ENV_MANAGER_ERROR_MESSAGE)

            model_install_steps = ""
            # If model_uri is not specified, dependencies are installed at runtime
            entrypoint = (
                self._get_install_pyfunc_deps_cmd(env_manager, install_mlflow, enable_mlserver)
                + f" C._serve('{env_manager}')"
            )

        dockerfile_text = docker_utils.generate_dockerfile(
            output_dir=output_dir,
            base_image=base_image,
            model_install_steps=model_install_steps,
            entrypoint=entrypoint,
            env_manager=env_manager,
            mlflow_home=mlflow_home,
            enable_mlserver=enable_mlserver,
            # always disable env creation at runtime for pyfunc
            disable_env_creation_at_runtime=True,
        )
        _logger.debug("generated dockerfile at {output_dir}", extra={"dockerfile": dockerfile_text})

    def _get_base_image(self, model_path: str, install_java: bool) -> str:
        """
        Determine the base image to use for the Dockerfile.

        We use Python slim base image when all of the following conditions are met:
          1. Model URI is specified by the user
          2. Model flavor does not require Java
          3. Python version is specified in the model

        Returns:
            Either the Ubuntu base image or the Python slim base image.
        """
        # Check if the model requires Java
        if not install_java:
            flavors = _get_all_flavor_configurations(model_path).keys()
            if java_flavors := JAVA_FLAVORS & flavors:
                _logger.info(f"Detected java flavors {java_flavors}, installing Java in the image")
                install_java = True

        # Use ubuntu base image if Java is required
        if install_java:
            return UBUNTU_BASE_IMAGE

        # Get Python version from MLmodel
        model_config_path = os.path.join(model_path, MLMODEL_FILE_NAME)
        try:
            model = Model.load(model_config_path)

            conf = model.flavors[pyfunc.FLAVOR_NAME]
            env_conf = conf[pyfunc.ENV]
            python_env_config_path = os.path.join(model_path, env_conf[em.VIRTUALENV])

            python_env = _PythonEnv.from_yaml(python_env_config_path)
            return PYTHON_SLIM_BASE_IMAGE.format(version=python_env.python)
        except Exception as e:
            _logger.warning(
                f"Failed to determine Python version from {model_config_path}. "
                f"Defaulting to {UBUNTU_BASE_IMAGE}. Error: {e}"
            )
            return UBUNTU_BASE_IMAGE

    def _model_installation_steps(self, model_path, env_manager, install_mlflow, enable_mlserver):
        model_dir = str(posixpath.join(_MODEL_DIR_NAME, os.path.basename(model_path)))
        # Copy model to image if model_uri is specified
        steps = (
            "# Copy model to image and install dependencies\n"
            f"COPY {model_dir} /opt/ml/model\nRUN python -c "
        )
        steps += (
            f'"{self._get_install_pyfunc_deps_cmd(env_manager, install_mlflow, enable_mlserver)}"'
        )

        # Install flavor-specific dependencies if needed
        flavors = _get_all_flavor_configurations(model_path).keys()
        for flavor in flavors:
            if flavor in FLAVOR_SPECIFIC_APT_PACKAGES:
                packages = " ".join(FLAVOR_SPECIFIC_APT_PACKAGES[flavor])
                steps += f"\nRUN apt-get install -y --no-install-recommends {packages}"

        return steps

    def _get_install_pyfunc_deps_cmd(
        self, env_manager: str, install_mlflow: bool, enable_mlserver: bool
    ):
        return (
            "from mlflow.models import container as C; "
            f"C._install_pyfunc_deps('/opt/ml/model', install_mlflow={install_mlflow}, "
            f"enable_mlserver={enable_mlserver}, env_manager='{env_manager}');"
        )
