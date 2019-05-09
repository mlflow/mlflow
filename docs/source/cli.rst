.. _cli:

Command-Line Interface
======================

The MLflow command-line interface (CLI) provides a simple interface to various functionality in MLflow. You can use the CLI to run projects, start the tracking UI, create and list experiments, download run artifacts,
serve MLflow Python Function and scikit-learn models, and serve models on
`Microsoft Azure Machine Learning <https://azure.microsoft.com/en-us/overview/machine-learning/>`_ and
`Amazon SageMaker <https://aws.amazon.com/sagemaker/>`_.

.. .. code-block:: bash
..
..    $ mlflow --help
..    Usage: mlflow [OPTIONS] COMMAND [ARGS]...
..
..    Options:
..      --version  Show the version and exit.
..      --help     Show this message and exit.
..
..    Commands:
..      azureml      Serve models on Azure Machine Learning.
..      download     Download the artifact at the specified DBFS or S3 URI. 
..      experiments  Manage experiments.
..      pyfunc       Serve Python models locally.
..      run          Run an MLflow project from the given URI.
..      sagemaker    Serve models on Amazon SageMaker.
..      sklearn      Serve scikit-learn models.
..      ui           Run the MLflow tracking UI.

Each individual command has a detailed help screen accessible via ``mlflow command_name --help``.

.. contents:: Table of Contents
  :local:
  :depth: 2

.. click:: mlflow.cli:cli
  :prog: mlflow
  :show-nested:
