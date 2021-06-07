Kubernetes-based CRD Model Training with MLflow
-------------------------------------
This directory contains an MLflow project that builds on the "docker" example project, by extending it to run in the context of a Kubernetes Custom Resource Definition (CRD). This example assumes that the user has an existing CRD installed in their Kubernetes cluster called "CustomJob".

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
    image:  mlflow-crd-example

Here, ``image`` can be any valid argument to ``docker run``, such as the tag, ID or URL of a Docker 
image (see `Docker docs <https://docs.docker.com/engine/reference/run/#general-form>`_). The above 
example references a locally-stored image (``mlflow-crd-example``) by tag.

Finally, the project includes a ``Dockerfile`` that is used to build the image referenced by the
``MLproject`` file. The ``Dockerfile`` specifies library dependencies required by the project, such 
as ``mlflow`` and ``scikit-learn``.

Running this Example
^^^^^^^^^^^^^^^^^^^^

First, install MLflow (via ``pip install mlflow``) and install 
`Docker <https://www.docker.com/get-started>`_.

Then, build the image for the project's Docker container environment. You must use the same image
name that is given by the ``docker_env.image`` field of the MLproject file. In this example, the
image name is ``mlflow-crd-example``. Issue the following command to build an image with this
name:

.. code-block:: bash

  docker build -t mlflow-crd-example -f Dockerfile .

Note that the name if the image used in the ``docker build`` command, ``mlflow-crd-example``, 
matches the name of the image referenced in the ``MLproject`` file.

Finally, run the example project using ``mlflow run examples/kubernetes_crd -P alpha=0.5``.

What happens when the project is run?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running ``mlflow run examples/kubernetes_crd`` builds a new Docker image based on ``mlflow-crd-example``
that also contains our project code. The resulting image is tagged as 
``mlflow-crd-example-<git-version>`` where ``<git-version>`` is the git commit ID. After the image is
built, MLflow executes the default (main) project entry point within the container using ``docker run``.

Environment variables, such as ``MLFLOW_TRACKING_URI``, are propagated inside the container during 
project execution. When running against a local tracking URI, MLflow mounts the host system's 
tracking directory (e.g., a local ``mlruns`` directory) inside the container so that metrics and 
params logged during project execution are accessible afterwards.
