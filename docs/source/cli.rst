.. _cli:

Command-Line Interface
======================

The MLflow command-line interface (CLI) provides a simple interface to various functionality in MLflow. You can use the CLI to run projects, start the tracking UI, create and list experiments, download run artifacts,
serve MLflow Python Function and scikit-learn models, and serve models on
`Microsoft Azure Machine Learning <https://azure.microsoft.com/en-us/services/machine-learning-service/>`_
and `Amazon SageMaker <https://aws.amazon.com/sagemaker/>`_.

Each individual command has a detailed help screen accessible via ``mlflow command_name --help``.

.. contents:: Table of Contents
  :local:
  :depth: 2

.. click:: mlflow.cli:cli
  :prog: mlflow
  :show-nested:


Configuration File
------------------

Additionally to the command line arguments, MLflow can read configuration from a JSON file which is passed to ``mlflow server`` or ``mlflow ui`` via the ``--mlflow-config`` option. The configuration options:

* **model_stages**: 
  Configure the available Model Stages. The order within the JSON array reflects their ordering within the MLflow UI. 
  
  Each model stage consists of the following:

    - **name**: The name of the model stage. Can be any combination of strings, numbers and underscores.

    - **color**: Define the color of the stage badge within the MLflow UI. Provided as lowercased `HTML named colors <https://www.w3schools.com/tags/ref_colornames.asp>`_. Default: grey

The **default configuration** is:
    
.. literalinclude:: ../../mlflow/server/default-config.json
   :language: json

