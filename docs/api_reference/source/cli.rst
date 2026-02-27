.. _cli:

Command-Line Interface
======================

The MLflow command-line interface (CLI) provides a simple interface to various functionality in MLflow. You can use the CLI to run projects, start the tracking UI, create and list experiments, download run artifacts,
serve MLflow Python Function and scikit-learn models, serve MLflow Python Function and scikit-learn models, and serve models on
`Microsoft Azure Machine Learning <https://azure.microsoft.com/en-us/services/machine-learning-service/>`_
and `Amazon SageMaker <https://aws.amazon.com/sagemaker/>`_.

Each individual command has a detailed help screen accessible via ``mlflow command_name --help``.

.. attention::
    It is advisable to set the ``MLFLOW_TRACKING_URI`` environment variable by default, 
    as the CLI does not automatically connect to a tracking server. Without this, 
    the CLI will default to using the local filesystem where the command is executed, 
    rather than connecting to a localhost or remote HTTP server. 
    Setting ``MLFLOW_TRACKING_URI`` to the URL of your desired tracking server is required for most of the commands below.


.. contents:: Table of Contents
  :local:
  :depth: 2

.. click:: mlflow.cli:cli
  :prog: mlflow
  :show-nested:
