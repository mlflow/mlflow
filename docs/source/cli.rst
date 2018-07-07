.. _cli:

Command-Line Interface
======================

The MLflow command-line interface (CLI) provides a simple interface to various functionality in MLflow. You can use the CLI to
start the tracking UI, run projects and runs, serve models to
`Microsoft Azure ML <https://azure.microsoft.com/en-us/overview/machine-learning/>`_ or
`Amazon SageMaker <https://aws.amazon.com/sagemaker/>`_, create
and list experiments, and download artifacts.

.. code::

    $ mlflow --help
    Usage: mlflow [OPTIONS] COMMAND [ARGS]...

    Options:
      --version  Show the version and exit.
      --help     Show this message and exit.

    Commands:
      azureml      Serve models on Azure ML.
      download     Download the artifact at the specified DBFS or S3 URI. 
      experiments  Run and list experiments.
      pyfunc       Serve Python models locally.
      run          Run an MLflow project from the given URI.
      sagemaker    Serve models on Amazon SageMaker.
      sklearn      Serve scikit-learn models.
      ui           Run the MLflow tracking UI.


Each individual command has a detailed help screen accessible via ``mlflow command_name --help``


Azure ML
--------

Subcommands to serve models on Azure ML.


Download
--------

Download the artifact at the specified DBFS or S3 URI into the specified
local output path, or the current directory if no output path is
specified.


Experiments
-----------

Subcommands to create and list experiments.


Python Function
---------------

Subcommands to serve Python models and apply them for inference.


SageMaker
---------

Subcommands to serve models on SageMaker.


scikit-learn Models
-------------------

Subcommands to serve scikit-learn models and apply them for inference.


Run
---

Run an MLflow project from the given URI.

If running locally (the default), the URI can be either a Git repository
URI or a local path. If running on Databricks, the URI must be a Git
repository.

By default, Git projects will run in a new working directory with the
given parameters, while local projects will run from the project's root
directory.


UI
--

Run the MLflow tracking UI. The UI is served at http://localhost:5000.
