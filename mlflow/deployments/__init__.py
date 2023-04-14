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
import json

from mlflow.exceptions import MlflowException
from mlflow.deployments.base import BaseDeploymentClient
from mlflow.deployments.interface import get_deploy_client, run_local
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class PredictionsResponse(dict):
    """
    Represents the predictions and metadata returned in response to a scoring request, such as a
    REST API request sent to the ``/invocations`` endpoint of an MLflow Model Server.
    """

    def get_predictions(self, predictions_format="dataframe", dtype=None):
        """
        Get the predictions returned from the MLflow Model Server in the specified format.

        :param predictions_format: The format in which to return the predictions. Either
                                   ``"dataframe"`` or ``"ndarray"``.
        :param dtype: The NumPy datatype to which to coerce the predictions. Only used when
                      the ``"ndarray"`` ``predictions_format`` is specified.
        :throws: Exception if the predictions cannot be represented in the specified format.
        :return: The predictions, represented in the specified format.
        """
        import numpy as np
        import pandas as pd

        if predictions_format == "dataframe":
            predictions = self["predictions"]
            if isinstance(predictions, str):
                return pd.DataFrame(data=[predictions])
            return pd.DataFrame(data=predictions)
        elif predictions_format == "ndarray":
            return np.array(self["predictions"], dtype)
        else:
            raise MlflowException(
                f"Unrecognized predictions format: '{predictions_format}'",
                INVALID_PARAMETER_VALUE,
            )

    def to_json(self, path=None):
        """
        Get the JSON representation of the MLflow Predictions Response.

        :param path: If specified, the JSON representation is written to this file path.
        :return: If ``path`` is unspecified, the JSON representation of the MLflow Predictions
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


__all__ = ["get_deploy_client", "run_local", "BaseDeploymentClient", "PredictionsResponse"]
