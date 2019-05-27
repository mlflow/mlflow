import os

from mlflow import pyfunc
import mlflow.sagemaker

SUPPORTED_DEPLOYMENT_FLAVORS = [
    pyfunc.FLAVOR_NAME,
]

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
