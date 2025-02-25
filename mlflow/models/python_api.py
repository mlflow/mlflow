import logging
import os
import shutil
from io import StringIO
from typing import ForwardRef, get_args, get_origin

from mlflow.exceptions import MlflowException
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import (
    is_databricks_connect,
)
from mlflow.utils.file_utils import TempDir

_logger = logging.getLogger(__name__)
UV_INSTALLATION_INSTRUCTIONS = (
    "Run `pip install uv` to install uv. See "
    "https://docs.astral.sh/uv/getting-started/installation for other installation methods."
)


def build_docker(
    model_uri=None,
    name="mlflow-pyfunc",
    env_manager=_EnvManager.VIRTUALENV,
    mlflow_home=None,
    install_java=False,
    install_mlflow=False,
    enable_mlserver=False,
    base_image=None,
):
    """
    Builds a Docker image whose default entrypoint serves an MLflow model at port 8080, using the
    python_function flavor. The container serves the model referenced by ``model_uri``, if
    specified. If ``model_uri`` is not specified, an MLflow Model directory must be mounted as a
    volume into the /opt/ml/model directory in the container.

    .. important::

        Since MLflow 2.10.1, the Docker image built with ``--model-uri`` does **not install Java**
        for improved performance, unless the model flavor is one of ``["johnsnowlabs", "h2o",
        "mleap", "spark"]``. If you need to install Java for other flavors, e.g. custom Python model
        that uses SparkML, please specify ``install-java=True`` to enforce Java installation.
        For earlier versions, Java is always installed to the image.


    .. warning::

        If ``model_uri`` is unspecified, the resulting image doesn't support serving models with
        the RFunc or Java MLeap model servers.

    NB: by default, the container will start nginx and gunicorn processes. If you don't need the
    nginx process to be started (for instance if you deploy your container to Google Cloud Run),
    you can disable it via the DISABLE_NGINX environment variable:

    .. code:: bash

        docker run -p 5001:8080 -e DISABLE_NGINX=true "my-image-name"

    See https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html for more information on the
    'python_function' flavor.

    Args:
        model_uri: URI to the model. A local path, a 'runs:/' URI, or a remote storage URI (e.g.,
            an 's3://' URI). For more information about supported remote URIs for model artifacts,
            see https://mlflow.org/docs/latest/tracking.html#artifact-stores
        name: Name of the Docker image to build. Defaults to 'mlflow-pyfunc'.
        env_manager: If specified, create an environment for MLmodel using the specified environment
            manager. The following values are supported: (1) virtualenv (default): use virtualenv
            and pyenv for Python version management (2) conda: use conda (3) local: use the local
            environment without creating a new one.
        mlflow_home: Path to local clone of MLflow project. Use for development only.
        install_java: If specified, install Java in the image. Default is False in order to
            reduce both the image size and the build time. Model flavors requiring Java will enable
            this setting automatically, such as the Spark flavor. (This argument is only available
            in MLflow 2.10.1 and later. In earlier versions, Java is always installed to the image.)
        install_mlflow: If specified and there is a conda or virtualenv environment to be activated
            mlflow will be installed into the environment after it has been activated.
            The version of installed mlflow will be the same as the one used to invoke this command.
        enable_mlserver: If specified, the image will be built with the Seldon MLserver as backend.
        base_image: Base image for the Docker image. If not specified, the default image is either
            UBUNTU_BASE_IMAGE = "ubuntu:20.04" or PYTHON_SLIM_BASE_IMAGE = "python:{version}-slim"
            Note: If custom image is used, there are no guarantees that the image will work. You
            may find greater compatibility by building your image on top of the ubuntu images. In
            addition, you must install Java and virtualenv to have the image work properly.
    """
    get_flavor_backend(model_uri, docker_build=True, env_manager=env_manager).build_image(
        model_uri,
        name,
        mlflow_home=mlflow_home,
        install_java=install_java,
        install_mlflow=install_mlflow,
        enable_mlserver=enable_mlserver,
        base_image=base_image,
    )


_CONTENT_TYPE_CSV = "csv"
_CONTENT_TYPE_JSON = "json"


@experimental
def predict(
    model_uri,
    input_data=None,
    input_path=None,
    content_type=_CONTENT_TYPE_JSON,
    output_path=None,
    env_manager=_EnvManager.VIRTUALENV,
    install_mlflow=False,
    pip_requirements_override=None,
    extra_envs=None,
    # TODO: add an option to force recreating the env
):
    """
    Generate predictions in json format using a saved MLflow model. For information about the input
    data formats accepted by this function, see the following documentation:
    https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools.

    Args:
        model_uri: URI to the model. A local path, a local or remote URI e.g. runs:/, s3://.
        input_data: Input data for prediction. Must be valid input for the PyFunc model. Refer
            to the :py:func:`mlflow.pyfunc.PyFuncModel.predict()` for the supported input types.

            .. note::
                If this API fails due to errors in input_data, use
                `mlflow.models.convert_input_example_to_serving_input` to manually validate
                your input data.
        input_path: Path to a file containing input data. If provided, 'input_data' must be None.
        content_type: Content type of the input data. Can be one of {‘json’, ‘csv’}.
        output_path: File to output results to as json. If not provided, output to stdout.
        env_manager: Specify a way to create an environment for MLmodel inference:

            - "virtualenv" (default): use virtualenv (and pyenv for Python version management)
            - "uv": use uv
            - "local": use the local environment
            - "conda": use conda

        install_mlflow: If specified and there is a conda or virtualenv environment to be activated
            mlflow will be installed into the environment after it has been activated. The version
            of installed mlflow will be the same as the one used to invoke this command.
        pip_requirements_override: If specified, install the specified python dependencies to the
            model inference environment. This is particularly useful when you want to add extra
            dependencies or try different versions of the dependencies defined in the logged model.

            .. tip::
                After validating the pip requirements override works as expected, you can update
                the logged model's dependency using `mlflow.models.update_model_requirements` API
                without re-logging it. Note that a registered model is immutable, so you need to
                register a new model version with the updated model.
        extra_envs: If specified, a dictionary of extra environment variables will be passed to the
            model inference environment. This is useful for testing what environment variables are
            needed for the model to run correctly. By default, environment variables existing in the
            current os.environ are passed, and this parameter can be used to override them.

            .. note::
                This parameter is only supported when `env_manager` is set to "virtualenv",
                "conda" or "uv".

    Code example:

    .. code-block:: python

        import mlflow

        run_id = "..."

        mlflow.models.predict(
            model_uri=f"runs:/{run_id}/model",
            input_data={"x": 1, "y": 2},
            content_type="json",
        )

        # Run prediction with "uv" as the environment manager
        mlflow.models.predict(
            model_uri=f"runs:/{run_id}/model",
            input_data={"x": 1, "y": 2},
            env_manager="uv",
        )

        # Run prediction with additional pip dependencies and extra environment variables
        mlflow.models.predict(
            model_uri=f"runs:/{run_id}/model",
            input_data={"x": 1, "y": 2},
            content_type="json",
            pip_requirements_override=["scikit-learn==0.23.2"],
            extra_envs={"OPENAI_API_KEY": "some_value"},
        )

    """
    # to avoid circular imports
    from mlflow.pyfunc import _PREBUILD_ENV_ROOT_LOCATION

    if content_type not in [_CONTENT_TYPE_JSON, _CONTENT_TYPE_CSV]:
        raise MlflowException.invalid_parameter_value(
            f"Content type must be one of {_CONTENT_TYPE_JSON} or {_CONTENT_TYPE_CSV}."
        )
    if extra_envs and env_manager not in (
        _EnvManager.VIRTUALENV,
        _EnvManager.CONDA,
        _EnvManager.UV,
    ):
        raise MlflowException.invalid_parameter_value(
            "Extra environment variables are only supported when env_manager is "
            f"set to '{_EnvManager.VIRTUALENV}', '{_EnvManager.CONDA}' or '{_EnvManager.UV}'."
        )
    if env_manager == _EnvManager.UV:
        if not shutil.which("uv"):
            raise MlflowException(
                f"Found '{env_manager}' as env_manager, but the 'uv' command is not found in the "
                f"PATH. {UV_INSTALLATION_INSTRUCTIONS} Alternatively, you can use 'virtualenv' or "
                "'conda' as the environment manager, but note their performances are not "
                "as good as 'uv'."
            )
    else:
        _logger.info(
            f"It is highly recommended to use `{_EnvManager.UV}` as the environment manager for "
            "predicting with MLflow models as its performance is significantly better than other "
            f"environment managers. {UV_INSTALLATION_INSTRUCTIONS}"
        )

    is_dbconnect_mode = is_databricks_connect()
    if is_dbconnect_mode:
        if env_manager not in (_EnvManager.VIRTUALENV, _EnvManager.UV):
            raise MlflowException(
                f"Databricks Connect only supports '{_EnvManager.VIRTUALENV}' or '{_EnvManager.UV}'"
                f" as the environment manager. Got {env_manager}."
            )
        pyfunc_backend_env_root_config = {
            "create_env_root_dir": False,
            "env_root_dir": _PREBUILD_ENV_ROOT_LOCATION,
        }
    else:
        pyfunc_backend_env_root_config = {"create_env_root_dir": True}

    def _predict(_input_path: str):
        return get_flavor_backend(
            model_uri,
            env_manager=env_manager,
            install_mlflow=install_mlflow,
            **pyfunc_backend_env_root_config,
        ).predict(
            model_uri=model_uri,
            input_path=_input_path,
            output_path=output_path,
            content_type=content_type,
            pip_requirements_override=pip_requirements_override,
            extra_envs=extra_envs,
        )

    if input_data is not None and input_path is not None:
        raise MlflowException.invalid_parameter_value(
            "Both input_data and input_path are provided. Only one of them should be specified."
        )
    elif input_data is not None:
        input_data = _serialize_input_data(input_data, content_type)

        # Write input data to a temporary file
        with TempDir() as tmp:
            input_path = os.path.join(tmp.path(), f"input.{content_type}")
            with open(input_path, "w") as f:
                f.write(input_data)

            _predict(input_path)
    else:
        _predict(input_path)


def _get_pyfunc_supported_input_types():
    # Importing here as the util module depends on optional packages not available in mlflow-skinny
    import mlflow.models.utils as base_module

    supported_input_types = []
    for input_type in get_args(base_module.PyFuncInput):
        if isinstance(input_type, type):
            supported_input_types.append(input_type)
        elif isinstance(input_type, ForwardRef):
            name = input_type.__forward_arg__
            if hasattr(base_module, name):
                cls = getattr(base_module, name)
                supported_input_types.append(cls)
        else:
            # typing instances like List, Dict, Tuple, etc.
            supported_input_types.append(get_origin(input_type))
    return tuple(supported_input_types)


def _serialize_input_data(input_data, content_type):
    # build-docker command is available in mlflow-skinny (which doesn't contain pandas)
    # so we shouldn't import pandas at the top level
    import pandas as pd

    # this introduces numpy as dependency, we shouldn't import it at the top level
    # as it is not available in mlflow-skinny
    from mlflow.models.utils import convert_input_example_to_serving_input

    valid_input_types = {
        _CONTENT_TYPE_CSV: (str, list, dict, pd.DataFrame),
        _CONTENT_TYPE_JSON: _get_pyfunc_supported_input_types(),
    }.get(content_type)

    if not isinstance(input_data, valid_input_types):
        raise MlflowException.invalid_parameter_value(
            f"Input data must be one of {valid_input_types} when content type is '{content_type}', "
            f"but got {type(input_data)}."
        )

    if content_type == _CONTENT_TYPE_CSV:
        if isinstance(input_data, str):
            _validate_csv_string(input_data)
            return input_data
        else:
            try:
                return pd.DataFrame(input_data).to_csv(index=False)
            except Exception as e:
                raise MlflowException.invalid_parameter_value(
                    "Failed to serialize input data to CSV format."
                ) from e

    try:
        # rely on convert_input_example_to_serving_input to validate
        # the input_data is valid type for the loaded pyfunc model
        return convert_input_example_to_serving_input(input_data)
    except Exception as e:
        raise MlflowException.invalid_parameter_value(
            "Invalid input data, please make sure the data is acceptable by the "
            "loaded pyfunc model. Use `mlflow.models.convert_input_example_to_serving_input` "
            "to manually validate your input data."
        ) from e


def _validate_csv_string(input_data: str):
    """
    Validate the string must be the path to a CSV file.
    """
    try:
        import pandas as pd

        pd.read_csv(StringIO(input_data))
    except Exception as e:
        raise MlflowException.invalid_parameter_value(
            message="Failed to deserialize input string data to Pandas DataFrame."
        ) from e
