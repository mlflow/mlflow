Singularity Model Training with MLflow
-------------------------------------
This directory contains an MLflow project that trains a linear regression model on the UC Irvine
Wine Quality Dataset. It is a replication of the same [docker example](../docker) but leverages Singularity
containers. The project uses a Singularity image to capture the dependencies needed to run
training code. Running a project in a Singularity container (as opposed to Conda or Docker) allows for capturing
non-Python dependencies, e.g. Java libraries, and is robustly reproducible in that the container is read only. 

Dependencies
^^^^^^^^^^^^
You'll need to install additional dependencies, spython "singularity python" to
interact with Singularity containers. Any of the following will work.

.. code-block:: console

  pip install mlflow[singularity]
  pip install .[singularity]
  pip install spython

Structure of this MLflow Project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This MLflow project contains a ``train.py`` file that trains a scikit-learn model and uses
MLflow Tracking APIs to log the model and its metadata (e.g., hyperparameters and metrics)
for later use and reference. ``train.py`` operates on the Wine Quality Dataset, which is included
in ``wine-quality.csv``.

Most importantly, the project also includes an ``MLproject`` file, which specifies the Singularity 
container environment in which to run the project using the ``singularity_env`` field:

.. code-block:: yaml

  singularity_env:
    image:  mlflow.sif

Here, ``image`` can be any valid argument to ``singularity run``, such as a library, or Docker
unique resource identifier, or a binary file on the filesystem (typically extension .sif). 
The above example references a local file (``mlflow.sif``).

Finally, the project includes a ``Singularity`` recipe that is used to build the image referenced by the
``MLproject`` file. The recipe specifies library dependencies required by the project, such 
as ``mlflow`` and ``scikit-learn``.

Running this Example
^^^^^^^^^^^^^^^^^^^^

First, install MLflow (via ``pip install mlflow``) and install 
`Singularity <https://sylabs.io/guides/3.3/user-guide/installation.html>`_.

Then, build the image for the project's Singularity recipe. You must use the same image
name that is given by the ``singularity_env.image`` field of the MLproject file. In this example, the
image name is ``mlflow.sif``. Issue the following command to build an image with this
name:

.. code-block:: bash

  sudo singularity build mlflow.sif Singularity

Note that the path of the image used in the ``singularity build`` command, ``mlflow.sif``, 
matches the name of the image referenced in the ``MLproject`` file.

Finally, run the example project using ``mlflow run examples/singularity -P alpha=0.5``.

What happens when the project is run?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running ``mlflow run examples/singularity`` uses the Singularity image ``mlflow.sif``
that also contains our project code. 

Environment variables, such as ``MLFLOW_TRACKING_URI``, are propagated inside the container during 
project execution. When running against a local tracking URI, MLflow mounts the host system's 
tracking directory (e.g., a local ``mlruns`` directory) inside the container so that metrics and 
params logged during project execution are accessible afterwards.
