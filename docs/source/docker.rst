Official MLflow Docker Image
============================

The official MLflow Docker image is available on GitHub Container Registry at https://ghcr.io/mlflow/mlflow.

There are two image variants.

mlflow:\<version\>
^^^^^^^^^^^^^^^^^^
This image contains mlflow plus all extra dependencies. It is intended to support "out of the box" all mlflow integrations.

Use this image if you just want to quickly have a working instance of mlflow, or if you don't want to bother rolling out your own image.

mlflow:\<version\>-slim
^^^^^^^^^^^^^^^^^^^^^^^
This image only contains the core mlflow dependencies, so most of the integrations (ie. backend stores databases,
artifact stores, etc.) will not work.

Use this image if you don't need any integration, or as the base image in case you want to have full control over
the extra dependencies installed.

.. code-block:: shell

    export CR_PAT=YOUR_TOKEN
    echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
    # Pull the latest version
    docker pull ghcr.io/mlflow/mlflow
    # Pull 2.0.1
    docker pull ghcr.io/mlflow/mlflow:v2.0.1
    # Pull 2.0.1 slim version
    docker pull ghcr.io/mlflow/mlflow:v2.0.1-slim
