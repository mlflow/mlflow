"""
Exposes experimental functionality for deploying mlflux models to custom serving tools.

Note: model deployment to AWS Sagemaker and AzureML can currently be performed via the
:py:mod:`mlflux.sagemaker` and :py:mod:`mlflux.azureml` modules, respectively.

mlflux does not currently provide built-in support for any other deployment targets, but support
for custom targets can be installed via third-party plugins. See a list of known plugins
`here <https://mlflux.org/docs/latest/plugins.html#deployment-plugins>`_.

This page largely focuses on the user-facing deployment APIs. For instructions on implementing
your own plugin for deployment to a custom serving tool, see
`plugin docs <http://mlflux.org/docs/latest/plugins.html#writing-your-own-mlflux-plugins>`_.
"""

from mlflux.deployments.base import BaseDeploymentClient
from mlflux.deployments.interface import get_deploy_client, run_local


__all__ = ["get_deploy_client", "run_local", "BaseDeploymentClient"]
