"""
Exposes functionality for deploying MLflow models to custom serving tools.

Note: model deployment to AWS Sagemaker can currently be performed via the
:py:mod:`mlflow.sagemaker` module. Model deployment to Azure can be performed by using the
`azureml library <https://pypi.org/project/azureml-mlflow/>`_.

MLflow does not currently provide built-in support for any other deployment targets, but support
for custom targets can be installed via third-party plugins. See a list of known plugins
`here <https://mlflow.org/docs/latest/plugins.html#deployment-plugins>`_.

This page largely focuses on the user-facing deployment APIs. For instructions on implementing
your own plugin for deployment to a custom serving tool, see
`plugin docs <http://mlflow.org/docs/latest/plugins.html#writing-your-own-mlflow-plugins>`_.
"""

import contextlib
import json

from mlflow.deployments.base import BaseDeploymentClient
from mlflow.deployments.databricks import DatabricksDeploymentClient, DatabricksEndpoint
from mlflow.deployments.interface import get_deploy_client, run_local
from mlflow.deployments.openai import OpenAIDeploymentClient
from mlflow.deployments.utils import get_deployments_target, set_deployments_target
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

with contextlib.suppress(Exception):
    # MlflowDeploymentClient depends on optional dependencies and can't be imported
    # if they are not installed.
    from mlflow.deployments.mlflow import MlflowDeploymentClient


class PredictionsResponse(dict):
    """
    Represents the predictions and metadata returned in response to a scoring request, such as a
    REST API request sent to the ``/invocations`` endpoint of an MLflow Model Server.
    """

    def get_predictions(self, predictions_format="dataframe", dtype=None):
        """Get the predictions returned from the MLflow Model Server in the specified format.

        Args:
            predictions_format: The format in which to return the predictions. Either
                ``"dataframe"`` or ``"ndarray"``.
            dtype: The NumPy datatype to which to coerce the predictions. Only used when
                the "ndarray" predictions_format is specified.

        Raises:
            Exception: If the predictions cannot be represented in the specified format.

        Returns:
            The predictions, represented in the specified format.

        """
        import numpy as np
        import pandas as pd
        from pandas.core.dtypes.common import is_list_like

        if predictions_format == "dataframe":
            predictions = self["predictions"]
            if isinstance(predictions, str):
                return pd.DataFrame(data=[predictions])
            if isinstance(predictions, dict) and not any(
                is_list_like(p) and getattr(p, "ndim", 1) == 1 for p in predictions.values()
            ):
                return pd.DataFrame(data=predictions, index=[0])
            return pd.DataFrame(data=predictions)
        elif predictions_format == "ndarray":
            return np.array(self["predictions"], dtype)
        else:
            raise MlflowException(
                f"Unrecognized predictions format: '{predictions_format}'",
                INVALID_PARAMETER_VALUE,
            )

    def to_json(self, path=None):
        """Get the JSON representation of the MLflow Predictions Response.

        Args:
            path: If specified, the JSON representation is written to this file path.

        Returns:
            If ``path`` is unspecified, the JSON representation of the MLflow Predictions
            Response. Else, None.

        """
        if path is not None:
            with open(path, "w") as f:
                json.dump(dict(self), f)
        else:
            return json.dumps(dict(self))

    @classmethod
    def from_json(cls, json_str):
        try:
            parsed_response = json.loads(json_str)
        except Exception as e:
            raise MlflowException("Predictions response contents are not valid JSON") from e
        if not isinstance(parsed_response, dict) or "predictions" not in parsed_response:
            raise MlflowException(
                f"Invalid response. Predictions response contents must be a dictionary"
                f" containing a 'predictions' field. Instead, received: {parsed_response}"
            )
        return PredictionsResponse(parsed_response)


__all__ = [
    "get_deploy_client",
    "run_local",
    "BaseDeploymentClient",
    "DatabricksDeploymentClient",
    "OpenAIDeploymentClient",
    "DatabricksEndpoint",
    "MlflowDeploymentClient",
    "PredictionsResponse",
    "get_deployments_target",
    "set_deployments_target",
]
