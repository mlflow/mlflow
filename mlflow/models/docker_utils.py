import os

import mlflow.sagemaker

from mlflow.sagemaker import _download_artifact_from_uri, _get_preferred_deployment_flavor, _validate_deployment_flavor, _get_deployment_config
from mlflow.models import Model

def _build_image(model_uri, image_name, mlflow_home=None, flavor=None):
    """
    Builds a Docker image containing the MLflow model (assumed to have the pyfunc flavor)
    at the specified URI.  The image's entry point serves the model with default settings. Note
    that the model is assumed to have the pyfunc flavor.
    """
    mlflow.sagemaker.build_image(
        image_name,
        mlflow_home=os.path.abspath(mlflow_home) if mlflow_home else None,
        model_uri=model_uri,
        flavor=flavor
    )
