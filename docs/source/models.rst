.. _models:

MLflow Models
=============

An MLflow Model is a standard format for packaging machine learning models that can be used in a
variety of downstream tools---for example, real-time serving through a REST API or batch inference
on Apache Spark. They provide a convention to save a model in different "flavors" that can be
understood by different downstream tools.

.. contents:: Table of Contents
  :local:
  :depth: 1


Storage Format
--------------

Each MLflow Model is simply a directory containing arbitrary files, together with an ``MLmodel``
file in the root of the directory that can define multiple *flavors* that the model can be viewed
in.

Flavors are the key concept that makes MLflow Models powerful: they are a convention that deployment
tools can use to understand the model, which makes it possible to write tools that work with models
from any ML library without having to integrate each tool with each library. MLflow defines
several "standard" flavors that all of its built-in deployment tools support, such as a "Python
function" flavor that describes how to run the model as a Python function. However, libraries can
also define and use other flavors. For example, MLflow's :py:mod:`mlflow.sklearn` library allows
loading models back as a scikit-learn ``Pipeline`` object for use in code that is aware of
scikit-learn, or as a generic Python function for use in tools that just need to apply the model
(for example, the ``mlflow sagemaker`` tool for deploying models to Amazon SageMaker).

All of the flavors that a particular model supports are defined in its ``MLmodel`` file in YAML
format. For example, :py:mod:`mlflow.sklearn` outputs models as follows:

.. code:: bash

    # Directory written by mlflow.sklearn.save_model(model, "my_model")
    my_model/
    ├── MLmodel
    └── model.pkl

And its ``MLmodel`` file describes two flavors:

.. code:: yaml

    time_created: 2018-05-25T17:28:53.35

    flavors:
      sklearn:
        sklearn_version: 0.19.1
        pickled_model: model.pkl
      python_function:
        loader_module: mlflow.sklearn

This model can then be used with any tool that supports *either* the ``sklearn`` or
``python_function`` model flavor. For example, the ``mlflow sklearn`` command can serve a
model with the ``sklearn`` flavor

.. code::

    mlflow sklearn serve my_model

In addition, the ``mlflow sagemaker`` command-line tool can package and deploy models to AWS
SageMaker as long as they support the ``python_function`` flavor:

.. code:: bash

    mlflow sagemaker deploy -m my_model [other options]

Fields in the MLmodel Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Apart from a **flavors** field listing the model flavors, the MLmodel YAML format can contain
the following fields:

time_created
    Date and time when the model was created, in UTC ISO 8601 format.

run_id
    ID of the run that created the model, if the model was saved using :ref:`tracking`.

Model API
---------

You can save and load MLflow Models in multiple ways. First, MLflow includes integrations with
several common libraries. For example, :py:mod:`mlflow.sklearn` contains
:py:func:`save_model <mlflow.sklearn.save_model>`, :py:func:`log_model <mlflow.sklearn.log_model>`
and :py:func:`load_model <mlflow.sklearn.load_model>` functions for Scikit-learn models. Second,
you can use the more general :py:class:`mlflow.models.Model` class to create and write models. This
class has four key functions:

* :py:func:`add_flavor <mlflow.sklearn.Model.add_flavor>` to add a flavor to the model. Each flavor
  has a string name and a dictionary of key-value attributes, where the values can be any object
  that can be serialized to YAML.
* :py:func:`save <mlflow.sklearn.Model.save>` saves the model to a local directory.
* :py:func:`log_artifact <mlflow.sklearn.Model.log_artifact>` logs the model as an artifact in the
  current run using MLflow Tracking.
* :py:func:`Model.load <mlflow.sklearn.Model.load>` loads a model from a local directory or
  from an artifact in a previous run.

Built-In Model Flavors
----------------------

MLflow provides several standard flavors that might be useful in your applications. Specifically,
many of its deployment tools support these flavors, so you can export your own model in one of these
flavors to benefit from all these tools.

Python Function (``python_function``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Python Function flavor defines a generic filesystem format for Python models and provides utilities
for saving and loading models to and from this format. The format is self-contained in the sense
that it includes all the information necessary to load and use a model. Dependencies
are stored either directly with the model or referenced via Conda environment.

The convention for Pyfunc models is to have a ``predict`` method or function with the following
signature:

.. code:: python

    predict(data: pandas.DataFrame) -> pandas.DataFrame | numpy.array

Other MLflow components expect Pyfunc models to follow this convention.

The Pyfunc model format is defined as a directory structure containing all required data, code and
configuration:

.. code:: bash

    ./dst-path/
        ./MLmodel - config
        <code> - any code packaged with the model (specified in the conf file, see below)
        <data> - any data packaged with the model (specified in the conf file, see below)
        <env>  - conda environment definition (specified in the conf file, see below)

A Pyfunc model directory must contain an ``MLmodel`` file in its root with "python_function" format and the following
parameters:

.. code:: bash

   - loader_module [required]:
         Python module that can load the model. Expected to be a module identifier
         (e.g. ``mlflow.sklearn``) importable via ``importlib.import_module``.
         The imported module must contain a function with the following signature:

              load_pyfunc(path: string) -> <pyfunc model>

         The path argument is specified by the data parameter and may refer to a file or directory.

   - code [optional]:
         A relative path to a directory containing the code packaged with this model.
         All files and directories inside this directory are added to the Python path
         prior to importing the model loader.

   - data [optional]:
         A relative path to a file or directory containing model data.
         the path is passed to the model loader.

   - env [optional]:
         A relative path to an exported Conda environment. If present this environment
         will be activated prior to running the model.

Example:

.. code:: bash

    >tree example/sklearn_iris/mlruns/run1/outputs/linear-lr
    ├── MLmodel
    ├── code
    │   ├── sklearn_iris.py
    │  
    ├── data
    │   └── model.pkl
    └── mlflow_env.yml

    >cat example/sklearn_iris/mlruns/run1/outputs/linear-lr/MLmodel
    python_function:
      code: code
      data: data/model.pkl
      env: mlflow_env.yml
      main: sklearn_iris

For more detail see docs at :py:mod:`mlflow.pyfunc`:

Scikit-learn (``sklearn``)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The sklearn model flavor provides an easy to use interface for handling scikit-learn models with no
external dependencies. It saves and loads models using Python's pickle module and also generates a valid
``Python Function`` flavor. For more information, see :py:mod:`mlflow.sklearn`.

TensorFlow (``tensorflow``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The TensorFlow model flavor enables logging TensorFlow ``Saved Models`` and loading them back as ``Python Function`` models for inference on Pandas DataFrames. Given a directory containing a saved model, you can log the model to MLflow via ``log_saved_model``. The saved model can then be loaded for inference via ``load_pyfunc()``. For more information, see :py:mod:`mlflow.tensorflow`. 

Custom Flavors
--------------

In general, you can add any flavor you'd like in MLmodel files, either by writing them directly or
building them with the :py:class:`mlflow.models.Model` class. Just choose an arbitrary string name
for your flavor. MLflow's tools will ignore flavors that they do not understand in the MLmodel file.

Built-In Deployment Tools
-------------------------

MLflow provides tools for deployment on a local machine and several production environments.
You can use these tools to easily apply your models in a production environment. Not all deployment
methods are available for all model flavors. Deployment is currently supported mostly for the
python function format and all compatible formats.

Local
^^^^^
MLflow can deploy models locally as a local REST API endpoint or to directly score csv files.
This functionality is a convenient way of testing models before uploading to remote.

Python function flavor can be deployed locally via :py:mod:`mlflow.pyfunc` module as

* :py:func:`serve <mlflow.pyfunc.cli.serve>`
  deploys model as a local REST api server
* :py:func:`predict <mlflow.pyfunc.cli.predict>` uses the model to generate prediction for local
  csv file.

For more info, see:

.. code:: bash

    mlflow pyfunc --help
    mlflow pyfunc serve --help
    mlflow pyfunc predict --help

Microsoft AzureML
^^^^^^^^^^^^^^^^^
MLflow's :py:mod:`mlflow.azureml` module can export ``Python Function`` models as Azure ML compatible models. It
can also be used to directly deploy and serve models on Azure ML, provided the environment has
been correctly set up.

* :py:func:`export <mlflow.azureml.cli.export>` exports the model in Azure ML-compatible format.
  MLFlow will output a directory with the dependencies necessary to deploy the model.

* :py:func:`deploy <mlflow.azureml.cli.deploy>` deploys the model directly to Azure ML.
  You first need to set up your environment to work with the Azure ML CLI. You can do this by
  starting a shell from the Azure ML Workbench application. You also have to set up all accounts
  required to run and deploy on Azure ML. Where the model is deployed is dependent on your
  active Azure ML environment. If the active environment is set up for local deployment, the model
  will be deployed locally in a Docker container (Docker is required).

Model export example:

.. code:: bash

    mlflow azureml export -m <path-to-model> -o test-output
    tree test-output
    test-output
    ├── create_service.sh  - you can use this script to upload the model to Azure ML
    ├── score.py - main module required by Azure ML
    └── test-output - dir containing MLFlow model in Python Function flavor

Example model workflow for deployment:

.. code:: bash

    az ml set env <local-env> - set environment to local deployment
    mlflow azureml deploy <parameters> - deploy locally to test the model
    az ml set env <cluster-env> - set environment to cluster
    mlflow azureml deploy <parameters> - deploy to the cloud

For more info, see:

.. code:: bash

    mlflow azureml --help
    mlflow azureml export --help
    mlflow azureml deploy --help

Amazon Sagemaker
^^^^^^^^^^^^^^^^
MLflow's :py:mod:`mlflow.sagemaker` module can deploy ``Python Function`` models on Sagemaker
or locally in a docker container with Sagemaker compatible environment (Docker is required).
Similarly to Azure ML, you have to set up your environment and user accounts first in order to
deploy to Sagemaker with MLflow. Also, in order to export a custom model to Sagemaker, you need a
MLflow-compatible Docker image to be available on Amazon ECR. MLflow provides a default Docker
image definition, however, it is up to you to build the actual image and upload it to ECR.
MLflow includes a utility function to perform this step. Once built and uploaded, the MLflow
container can be used for all MLflow models.

* :py:func:`build-and-push-container <mlflow.sagemaker.cli.build_and_push_container>` builds an MLFLow
  Docker image and uploads it to ECR. The calling user has to have the correct permissions set up. The image
  is built locally and requires Docker to be present on the machine that performs this step.


* :py:func:`run_local <mlflow.sagemaker.cli.run_local>` deploys the model locally in a Docker
  container. The image and the environment should be identical to how the model would be run
  remotely and it is therefore useful for testing the model prior to deployment.

* :py:func:`deploy <mlflow.sagemaker.cli.deploy>` deploys the model on Amazon Sagemaker. MLflow
  will upload the Python Function model into S3 and start an Amazon Sagemaker endpoint serving
  the model.

Example workflow:

.. code:: bash

    mlflow sagemaker build-and-push-container  - build the container (only needs to be called once)
    mlflow sagemaker run-local -m <path-to-yourmodel>  - test the model locally
    mlflow sagemaker deploy <parameters> - deploy the model to the cloud


For more info, see:

.. code:: bash

    mlflow sagemaker --help
    mlflow sagemaker build-and-push-container --help
    mlflow sagemaker run-local --help
    mlflow sagemaker deploy --help


Spark
^^^^^
MLFLow can output python function model as a Spark UDF, which can be uploaded to a Spark cluster and
used to score the model.

Example:

.. code:: python

    pyfunc_udf = mlflow.pyfunc.spark_udf(<path-to-model>)
    df = spark_df.withColumn("prediction", pyfunc_udf(<features>))
