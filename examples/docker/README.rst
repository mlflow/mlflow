Dockerized Model Training with MLflow
-------------------------------------
This directory contains an MLflow project that trains a linear regression model on the UC Irvine
Wine Quality Dataset. The project uses a Docker image to capture the dependencies needed to run
training code. Running a project in a Docker environment (as opposed to conda) allows for capturing
non-Python dependencies, e.g. Java libraries. In the future, we also hope to add tools to MLflow
for running Dockerized projects e.g. on a Kubernetes cluster for scaleout.

Structure of this MLflow Project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This MLflow project contains a ``train.py`` file that trains a scikit-learn model and uses
MLflow's Tracking APIs to log the model and its metadata (e.g, hyperparameters and metrics)
for later use and reference. ``train.py`` operates on the Wine Quality Dataset, which is included
in ``wine-quality.csv``.

Most importantly, the project also includes an ``MLProject`` file, which specifies the Docker 
container environment in which to run the project via the ``docker_env`` field:

.. code-block:: yaml

  docker_env:
    image:  mlflow-docker-example

Here, ``image`` can be any valid argument to ``docker run``, such as the tag, ID or URL of a Docker 
image (see `Docker docs <https://docs.docker.com/engine/reference/run/#general-form>`_). The above 
example references a locally-stored image (mlflow-docker-example) by tag.

Finally, the project includes a ``Dockerfile`` that is used to build the image referenced by the
``MLProject`` file. The ``Dockerfile`` specifies library dependencies required by the project, such 
as ``mlflow`` and ``scikit-learn``.

Running this Example
^^^^^^^^^^^^^^^^^^^^

First, install MLflow (via ``pip install mlflow``) and install 
`Docker <https://www.docker.com/get-started>`_. 

Then, build the image for the project's Docker container environment via

.. code-block:: bash

  docker build -t mlflow-docker-example -f Dockerfile .

Note that the name if the image used in the ``docker build`` command, ``mlflow-docker-example``, 
matches the name of the image referenced in the ``MLProject`` file.

Finally, run the example project via ``mlflow run examples/docker -P alpha=0.5``.

What happens when the project is run?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running ``mlflow run examples/docker`` builds a new Docker image based on ``mlflow-docker-example``
but also containing our project code, then executes the default (main) project entry point
within the container via ``docker run``. The resulting image will be tagged as 
``mlflow-docker-example-<git-version>`` where git-version is the git commit ID.

Environment variables such as ``MLFLOW_TRACKING_URI`` are propagated inside the container during 
project execution. When running against a local tracking URI (e.g., a local ``mlruns`` directory), 
MLflow will mount the host system's tracking directory inside the container so that metrics and 
params logged during project execution are accessible afterwards.

