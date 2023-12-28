import json
import os
from io import StringIO
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from mlflow.exceptions import MlflowException
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.file_utils import TempDir


def build_docker(
    model_uri=None,
    name="mlflow-pyfunc",
    env_manager=_EnvManager.VIRTUALENV,
    mlflow_home=None,
    install_mlflow=False,
    enable_mlserver=False,
):
    """
    Builds a Docker image whose default entrypoint serves an MLflow model at port 8080, using the
    python_function flavor. The container serves the model referenced by ``model_uri``, if
    specified. If ``model_uri`` is not specified, an MLflow Model directory must be mounted as a
    volume into the /opt/ml/model directory in the container.

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
    """
    get_flavor_backend(model_uri, docker_build=True, env_manager=env_manager).build_image(
        model_uri,
        name,
        mlflow_home=mlflow_home,
        install_mlflow=install_mlflow,
        enable_mlserver=enable_mlserver,
    )


_CONTENT_TYPE_CSV = "csv"
_CONTENT_TYPE_JSON = "json"

_SUPPORTED_INPUT_DATA_TYPES = {
    _CONTENT_TYPE_CSV: (str, list, dict, pd.DataFrame),
    _CONTENT_TYPE_JSON: (str, dict),
}


def predict(
    model_uri: str,
    # Subset of PyfuncInput
    input_data_or_path: Union[str, Dict[str, Any], List[Any], pd.DataFrame, None],
    content_type: str = _CONTENT_TYPE_JSON,
    output_path: Optional[str] = None,
    env_manager: _EnvManager = _EnvManager.VIRTUALENV,
    install_mlflow: bool = False,
    pip_requirements_override: Optional[List[str]] = None,
):
    """
    Generate predictions in json format using a saved MLflow model. For information about the input
    data formats accepted by this function, see the following documentation:
    https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools.

    :param model_uri: URI to the model. A local path, a local or remote URI e.g. runs:/, s3://.
    :param input_data_or_path: Input data for prediction. It can be one of the following:

                  - A Python dictionary that contains either:
                     - single input payload, when content type is "json".
                     - Pandas DataFrame, when content type is "csv".
                  - A Python list. The content type has to be "csv".
                  - A Pandas DataFrame. The content type has to be "csv".
                  - A string represents serialized input data. e.g. '{"inputs": [1, 2]}'
                  - A path to a local file contains input data, either a JSON or a CSV file.
                  - None to input data from stdin.
    :param content_type: Content type of the input data. Can be one of {‘json’, ‘csv’}.
    :param output_path: File to output results to as json. If not provided, output to stdout.
    :param env_manager: Specify a way to create an environment for MLmodel inference:

                  - virtualenv (default): use virtualenv (and pyenv for Python version management)
                  - local: use the local environment
                  - conda: use conda
    :param install_mlflow: If specified and there is a conda or virtualenv environment to be
        activated mlflow will be installed into the environment after it has been activated.
        The version of installed mlflow will be the same as the one used to invoke this command.
    :param pip_requirements_override: If specified, install the specified python dependencies to
        the model inference environment. This is particularly useful when you want to add extra
        dependencies or try different versions of the dependencies defined in the logged model.

    Code example:

    .. code-block:: python

        import mlflow

        run_id = "..."

        mlflow.pyfunc.predict(
            model_uri=f"runs:/{run_id}/model",
            input_data={"x": 1, "y": 2},
            content_type="json",
        )

        # Run prediction with addtioanl pip dependencies
        mlflow.pyfunc.predict(
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

    if input_data_or_path is None or _is_filepath(input_data_or_path):
        _predict(input_data_or_path)
    else:
        input_data = _serialize_input_data(input_data_or_path, content_type)

        # Write input data to a temporary file
        with TempDir() as tmp:
            input_path = os.path.join(tmp.path(), f"input.{content_type}")
            with open(input_path, "w") as f:
                f.write(input_data)

            _predict(input_path)


def _is_filepath(x):
    return isinstance(x, str) and os.path.exists(x) and os.path.isfile(x)


def _serialize_input_data(input_data: Union[str, List, Dict, pd.DataFrame], content_type: str):
    valid_input_types = _SUPPORTED_INPUT_DATA_TYPES[content_type]
    if not isinstance(input_data, valid_input_types):
        raise MlflowException.invalid_parameter_value(
            f"Input data must be one of {valid_input_types} when content type is '{content_type}'."
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
            return json.dumps(input_data)
    except Exception as e:
        raise MlflowException.invalid_parameter_value(
            message="Input data could not be serialized to {content_type}."
        ) from e


def _validate_string(input_data: str, content_type: str):
    try:
        if content_type == _CONTENT_TYPE_CSV:
            pd.read_csv(StringIO(input_data))
        else:
            json.loads(input_data)
    except Exception as e:
        target = "JSON" if content_type == _CONTENT_TYPE_JSON else "Pandas Dataframe"
        raise MlflowException.invalid_parameter_value(
            message=f"Failed to deserialize input string data to {target}."
        ) from e
