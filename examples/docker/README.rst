Dockerized Model Training with MLflow
-------------------------------------
This directory contains an MLflow project that trains a linear regression model on the UC Irvine
Wine Quality Dataset. The project uses a Docker image to capture the dependencies needed to run
training code. Running a project in a Docker environment (as opposed to Conda) allows for capturing
non-Python dependencies, e.g. Java libraries. In the future, we also hope to add tools to MLflow
for running Dockerized projects e.g. on a Kubernetes cluster for scale out.

Structure of this MLflow Project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This MLflow project contains a ``train.py`` file that trains a scikit-learn model and uses
MLflow Tracking APIs to log the model and its metadata (e.g., hyperparameters and metrics)
for later use and reference. ``train.py`` operates on the Wine Quality Dataset, which is included
in ``wine-quality.csv``.

Most importantly, the project also includes an ``MLproject`` file, which specifies the Docker
container environment in which to run the project using the ``docker_env`` field:

.. code-block:: yaml

  docker_env:
    image:  mlflow-docker-example

Here, ``image`` can be any valid argument to ``docker run``, such as the tag, ID or URL of a Docker
image (see `Docker docs <https://docs.docker.com/engine/reference/run/#general-form>`_). The above
example references a locally-stored image (``mlflow-docker-example``) by tag.

Finally, the project includes a ``Dockerfile`` that is used to build the image referenced by the
``MLproject`` file. The ``Dockerfile`` specifies library dependencies required by the project, such
as ``mlflow`` and ``scikit-learn``.

Running this Example
^^^^^^^^^^^^^^^^^^^^

First, install MLflow (via ``pip install mlflow``) and install
`Docker <https://www.docker.com/get-started>`_.

Then, build the image for the project's Docker container environment. You must use the same image
name that is given by the ``docker_env.image`` field of the MLproject file. In this example, the
image name is ``mlflow-docker-example``. Issue the following command to build an image with this
name:

.. code-block:: bash

  docker build -t mlflow-docker-example -f Dockerfile .

Note that the name if the image used in the ``docker build`` command, ``mlflow-docker-example``,
matches the name of the image referenced in the ``MLproject`` file.

Finally, run the example project using ``mlflow run examples/docker -P alpha=0.5``.

.. note::
    If running this example on a Mac with Apple silicon, ensure that Docker Desktop is running and
    that you are logged in to the Docker Desktop service.
    If you are modifying the example ``DockerFile`` to specify older versions of ``scikit-learn``,
    you should enable `Rosetta compatibility <https://docs.docker.com/desktop/settings/mac/#features-in-development>`_
    in the Docker Desktop configuration settings to ensure that the appropriate ``cython`` compiler is used.

What happens when the project is run?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running ``mlflow run examples/docker`` builds a new Docker image based on ``mlflow-docker-example``
that also contains our project code. The resulting image is tagged as
``mlflow-docker-example-<git-version>`` where ``<git-version>`` is the git commit ID. After the image is
built, MLflow executes the default (main) project entry point within the container using ``docker run``.

Environment variables, such as ``MLFLOW_TRACKING_URI``, are propagated inside the container during
project execution. When running against a local tracking URI, MLflow mounts the host system's
tracking directory (e.g., a local ``mlruns`` directory) inside the container so that metrics and
params logged during project execution are accessible afterwards.
