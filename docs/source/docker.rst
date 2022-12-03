Official MLflow Docker Image
============================

The official MLflow Docker image is available on GitHub Container Registry at https://ghcr.io/mlflow/mlflow.

.. code-block:: shell

    export CR_PAT=YOUR_TOKEN
    echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
    # Pull the latest version
    docker pull ghcr.io/mlflow/mlflow
    # Pull 2.0.1
    docker pull ghcr.io/mlflow/mlflow:v2.0.1
