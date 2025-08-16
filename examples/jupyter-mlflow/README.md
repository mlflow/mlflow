# JupyterLab with MLflow Proxy

This directory contains a Docker setup for running JupyterLab with MLflow as a proxy.

## Prerequisites

- Docker installed on your machine.


## Building the Docker Image


```bash
docker build -t jupyter-mlflow-proxy .


## Running the Docker Container

Run the Docker container with the following command:

docker run -p 8888:8888 -p 5000:5000 jupyter-mlflow-proxy


Accessing JupyterLab

Open your web browser and navigate to http://localhost:8888. You should see the JupyterLab interface.

Verifying MLflow Integration

In a Jupyter notebook, run the following code to verify that MLflow is accessible:

import mlflow
print(mlflow.__version__)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.list_experiments()
