"""
Exposes functionality for deploying MLflow models to custom serving tools.

Note: model deployment to AWS Sagemaker and AzureML can currently be performed via the
:py:mod:`mlflow.sagemaker` and :py:mod:`mlflow.azureml` modules, respectively.

MLflow does not currently provide built-in support for any other deployment targets, but support
for custom targets can be installed via third-party plugins. See a list of known plugins
`here <https://mlflow.org/docs/latest/plugins.html#deployment-plugins>`_.

This page largely focuses on the user-facing deployment APIs. For instructions on implementing
your own plugin for deployment to a custom serving tool, see
`plugin docs <http://mlflow.org/docs/latest/plugins.html#writing-your-own-mlflow-plugins>`_.
"""

from mlflow.deployments.base import BaseDeploymentClient
from mlflow.deployments.interface import get_deploy_client, run_local


__all__ = ["get_deploy_client", "run_local", "BaseDeploymentClient"]
