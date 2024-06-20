import json
import logging
import os
from datetime import datetime
from io import StringIO
from typing import ForwardRef, get_args, get_origin

from mlflow.exceptions import MlflowException
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import dump_input_data

_logger = logging.getLogger(__name__)


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
            see https://mlflow.org/docs/latest/tracking.html#artifact-stores"
        name: Name of the Docker image to build. Defaults to 'mlflow-pyfunc'.
        install_java: If specified, install Java in the image. Default is False in order to
            reduce both the image size and the build time. Model flavors requiring Java will enable
            this setting automatically, such as the Spark flavor. (This argument is only available
            in MLflow 2.10.1 and later. In earlier versions, Java is always installed to the image.)

        enable_mlserver: If specified, the image will be built with the Seldon MLserver as backend.
        env_manager: If specified, create an environment for MLmodel using the specified environment
            manager. The following values are supported: (1) virtualenv (default): use virtualenv
            and pyenv for Python version management (2) conda: use conda (3) local: use the local
            environment without creating a new one.

        install_mlflow: If specified and there is a conda or virtualenv environment to be activated
            mlflow will be installed into the environment after it has been activated.
            The version of installed mlflow will be the same as the one used to invoke this command.

        mlflow_home: Path to local clone of MLflow project. Use for development only.

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


def predict(
    model_uri,
    input_data=None,
    input_path=None,
    content_type=_CONTENT_TYPE_JSON,
    output_path=None,
    env_manager=_EnvManager.VIRTUALENV,
    install_mlflow=False,
    pip_requirements_override=None,
):
    """
    Generate predictions in json format using a saved MLflow model. For information about the input
    data formats accepted by this function, see the following documentation:
    https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools.

    Args:
        model_uri: URI to the model. A local path, a local or remote URI e.g. runs:/, s3://.
        input_data: Input data for prediction. Must be valid input for the PyFunc model. Refer
            to the :py:func:`mlflow.pyfunc.PyFuncModel.predict()` for the supported input types.
        input_path: Path to a file containing input data. If provided, 'input_data' must be None.
        content_type: Content type of the input data. Can be one of {‘json’, ‘csv’}.
        output_path: File to output results to as json. If not provided, output to stdout.
        env_manager: Specify a way to create an environment for MLmodel inference:

            - "virtualenv" (default): use virtualenv (and pyenv for Python version management)
            - "local": use the local environment
            - "conda": use conda

        install_mlflow: If specified and there is a conda or virtualenv environment to be activated
            mlflow will be installed into the environment after it has been activated. The version
            of installed mlflow will be the same as the one used to invoke this command.
        pip_requirements_override: If specified, install the specified python dependencies to the
            model inference environment. This is particularly useful when you want to add extra
            dependencies or try different versions of the dependencies defined in the logged model.

    Code example:

    .. code-block:: python

        import mlflow

        run_id = "..."

        mlflow.models.predict(
            model_uri=f"runs:/{run_id}/model",
            input_data={"x": 1, "y": 2},
            content_type="json",
        )

        # Run prediction with additional pip dependencies
        mlflow.models.predict(
            model_uri=f"runs:/{run_id}/model",
            input_data='{"x": 1, "y": 2}',
            content_type="json",
            pip_requirements_override=["scikit-learn==0.23.2"],
        )

    """
    if content_type not in [_CONTENT_TYPE_JSON, _CONTENT_TYPE_CSV]:
        raise MlflowException.invalid_parameter_value(
            f"Content type must be one of {_CONTENT_TYPE_JSON} or {_CONTENT_TYPE_CSV}."
        )

    def _predict(_input_path: str):
        return get_flavor_backend(
            model_uri, env_manager=env_manager, install_mlflow=install_mlflow
        ).predict(
            model_uri=model_uri,
            input_path=_input_path,
            output_path=output_path,
            content_type=content_type,
            pip_requirements_override=pip_requirements_override,
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

    valid_input_types = {
        _CONTENT_TYPE_CSV: (str, list, dict, pd.DataFrame),
        _CONTENT_TYPE_JSON: _get_pyfunc_supported_input_types(),
    }.get(content_type)

    if not isinstance(input_data, valid_input_types):
        raise MlflowException.invalid_parameter_value(
            f"Input data must be one of {valid_input_types} when content type is '{content_type}', "
            f"but got {type(input_data)}."
        )

    # If the input is already string, check if the input string can be deserialized correctly
    if isinstance(input_data, str):
        _validate_string(input_data, content_type)
        return input_data

    try:
        if content_type == _CONTENT_TYPE_CSV:
            # We should not set header=False here because the scoring server expects the
            # first row to be the header
            return pd.DataFrame(input_data).to_csv(index=False)
        else:
            return _serialize_to_json(input_data)
    except Exception as e:
        raise MlflowException.invalid_parameter_value(
            message=f"Input data could not be serialized to {content_type}."
        ) from e


def _serialize_to_json(input_data):
    # imported inside function to avoid circular import
    from mlflow.pyfunc.scoring_server import SUPPORTED_FORMATS, SUPPORTED_LLM_FORMATS

    if isinstance(input_data, dict) and any(
        key in input_data for key in SUPPORTED_FORMATS | SUPPORTED_LLM_FORMATS
    ):
        return json.dumps(input_data)
    else:
        original_input_data = input_data

        if isinstance(input_data, (datetime, bool, bytes, float, int, str)):
            input_data = [input_data]

        input_data = dump_input_data(input_data)

        _logger.info(
            f"Your input data has been transformed to comply with the expected input format for "
            "the MLflow scoring server. If you want to deploy the model to online serving, make "
            "sure to apply the same preprocessing in your inference client. Please also refer to "
            "https://www.mlflow.org/docs/latest/deployment/deploy-model-locally.html#json-input "
            "for more details on the supported input format."
            f"\n\nOriginal input data:\n{original_input_data}"
            f"\n\nTransformed input data:\n{input_data}"
        )
        return input_data


def _validate_string(input_data: str, content_type: str):
    try:
        if content_type == _CONTENT_TYPE_CSV:
            import pandas as pd

            pd.read_csv(StringIO(input_data))
        else:
            json.loads(input_data)
    except Exception as e:
        target = "JSON" if content_type == _CONTENT_TYPE_JSON else "Pandas DataFrame"
        raise MlflowException.invalid_parameter_value(
            message=f"Failed to deserialize input string data to {target}."
        ) from e
