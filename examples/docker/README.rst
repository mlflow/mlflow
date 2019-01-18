Dockerized Model Training with MLflow
-------------------------------------
This directory contains an MLflow project that trains a linear regression model on the UC Irvine
Wine Quality Dataset. The project uses a docker image to capture the dependencies needed to run
training code. Running a project in a docker environment (as opposed to conda) allows for capturing
non-Python dependencies, e.g. Java libraries. In the future, we also hope to add tools to MLflow
for running dockerized projects e.g. on a Kubernetes cluster for scaleout.


Running this Example
^^^^^^^^^^^^^^^^^^^^

Install MLflow via `pip install mlflow` and `docker <https://www.docker.com/get-started>`_.
Then, build a docker image containing MLflow via `docker build examples/docker -t mlflow-docker-example`
and run the example project via `mlflow run examples/docker -P alpha=0.5`

What happens when the project is run?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Let's start by looking at the MLproject file, which specifies the docker image in which to run the
project via a docker_env field:

```
docker_env:
  image:  mlflow-docker-example
```

Here, `image` can be any valid argument to `docker run`, such as the tag, ID or
URL of a docker image (see `Docker docs <https://docs.docker.com/engine/reference/run/#general-form>`_).
The above example references a locally-stored image (mlflow-docker-example) by tag.

Running `mlflow run examples/docker` builds a new docker image based on `mlflow-docker-example`
but also containing our project code, then executes the default (main) project entry point
within the container via `docker run`.
This built image will be tagged as `mlflow-docker-example-<git-version>` where git-version is the git 
commit ID.

Environment variables such as MLFLOW_TRACKING_URI are
propagated inside the container during project execution. When running against a local tracking URI,
e.g. a local `mlruns` directory, MLflow will mount the host system's tracking directory inside the
container so that metrics and params logged during project execution are accessible afterwards.

