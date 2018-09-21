.. _models:

MLflow Models
=============

An MLflow Model is a standard format for packaging machine learning models that can be used in a
variety of downstream tools---for example, real-time serving through a REST API or batch inference
on Apache Spark. The format defines a convention that lets you save a model in different "flavors" 
that can be understood by different downstream tools.

.. contents:: Table of Contents
  :local:
  :depth: 1


Storage Format
--------------

Each MLflow Model is a directory containing arbitrary files, together with an ``MLmodel``
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

::

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
model with the ``sklearn`` flavor:

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
:py:func:`save_model <mlflow.sklearn.save_model>`, :py:func:`log_model <mlflow.sklearn.log_model>`,
and :py:func:`load_model <mlflow.sklearn.load_model>` functions for scikit-learn models. Second,
you can use the :py:class:`mlflow.models.Model` class to create and write models. This
class has four key functions:

* :py:func:`add_flavor <mlflow.models.Model.add_flavor>` to add a flavor to the model. Each flavor
  has a string name and a dictionary of key-value attributes, where the values can be any object
  that can be serialized to YAML.
* :py:func:`save <mlflow.models.Model.save>` to save the model to a local directory.
* :py:func:`log <mlflow.models.Model.log>` to log the model as an artifact in the
  current run using MLflow Tracking.
* :py:func:`load <mlflow.models.Model.load>` to load a model from a local directory or
  from an artifact in a previous run.

Built-In Model Flavors
----------------------

MLflow provides several standard flavors that might be useful in your applications. Specifically,
many of its deployment tools support these flavors, so you can export your own model in one of these
flavors to benefit from all these tools.

Python Function (``python_function``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``python_function`` model flavor defines a generic filesystem format for Python models and provides utilities
for saving and loading models to and from this format. The format is self-contained in the sense
that it includes all the information necessary to load and use a model. Dependencies
are stored either directly with the model or referenced via Conda environment.

The convention for ``python_function`` models is to have a ``predict`` method or function with the following
signature:

.. code:: python

    predict(data: pandas.DataFrame) -> [pandas.DataFrame | numpy.array]

Other MLflow components expect ``python_function`` models to follow this convention.

The ``python_function`` model format is defined as a directory structure containing all required data, code, and
configuration:

.. code:: bash

  ./dst-path/
          ./MLmodel: configuration
          <code>: code packaged with the model (specified in the MLmodel file)
          <data>: data packaged with the model (specified in the MLmodel file)
          <env>: Conda environment definition (specified in the MLmodel file)

A ``python_function`` model directory must contain an ``MLmodel`` file in its root with "python_function" format and the following parameters:

- loader_module [required]:
     Python module that can load the model. Expected to be a module identifier
     (for example, ``mlflow.sklearn``) importable via ``importlib.import_module``.
     The imported module must contain a function with the following signature:

          _load_pyfunc(path: string) -> <pyfunc model>

     The path argument is specified by the ``data`` parameter and may refer to a file or directory.

- code [optional]:
     A relative path to a directory containing the code packaged with this model.
     All files and directories inside this directory are added to the Python path
     prior to importing the model loader.

- data [optional]:
     A relative path to a file or directory containing model data.
     The path is passed to the model loader.

- env [optional]:
     A relative path to an exported Conda environment. If present this environment
     is activated prior to running the model.

.. rubric:: Example

.. code:: bash

   tree example/sklearn_iris/mlruns/run1/outputs/linear-lr
   
::
 
   ├── MLmodel
   ├── code
   │   ├── sklearn_iris.py
   │  
   ├── data
   │   └── model.pkl
   └── mlflow_env.yml

.. code:: bash

   cat example/sklearn_iris/mlruns/run1/outputs/linear-lr/MLmodel
   
::
  
   python_function:
     code: code
     data: data/model.pkl
     loader_module: mlflow.sklearn
     env: mlflow_env.yml
     main: sklearn_iris

For more information, see :py:mod:`mlflow.pyfunc`.

H\ :sub:`2`\ O (``h2o``)
^^^^^^^^^^^^^^^^^^^^^^^^

The H2O model flavor enables logging and loading H2O models. These models will be saved by using the :py:mod:`mlflow.h2o.save_model`. Using :py:mod:`mlflow.h2o.log_model` will also enable a valid ``Python Function`` flavor.

When loading a H2O model as a PyFunc model, :py:mod:`h2o.init(...)` will be called. Therefore, the right version of h2o(-py) has to be in the environment. The arguments given to :py:mod:`h2o.init(...)` can be customized in ``model.h2o/h2o.yaml`` under the key ``init``. For more information, see :py:mod:`mlflow.h2o`.

Keras (``keras``)
^^^^^^^^^^^^^^^^^

The ``keras`` model flavor enables logging and loading Keras models. This model will be saved in a HDF5 file format, via the model_save functionality provided by Keras. Additionally, model can be loaded back as ``Python Function``. For more information, see :py:mod:`mlflow.keras`.

MLeap (``mleap``)
^^^^^^^^^^^^^^^^^

The ``mleap`` model flavor supports saving models using the MLeap persistence mechanism. A companion module for loading MLflow models with the MLeap flavor format is available in the ``mlflow/java`` package. For more information, see :py:mod:`mlflow.mleap`.

PyTorch (``pytorch``)
^^^^^^^^^^^^^^^^^^^^^

The ``pytorch`` model flavor enables logging and loading PyTorch models. Model is completely stored in `.pth` format using `torch.save(model)` method. Given a directory containing a saved model, you can log the model to MLflow via ``log_saved_model``. The saved model can then be loaded for inference via ``mlflow.pyfunc.load_pyfunc()``. For more information, see :py:mod:`mlflow.pytorch`.

Scikit-learn (``sklearn``)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``sklearn`` model flavor provides an easy to use interface for handling scikit-learn models with no
external dependencies. It saves and loads models using Python's pickle module and also generates a valid
``python_function`` flavor model. For more information, see :py:mod:`mlflow.sklearn`. 


Spark MLlib (``spark``)
^^^^^^^^^^^^^^^^^^^^^^^

The ``spark`` model flavor enables exporting Spark MLlib models as MLflow models. Exported models are
saved using Spark MLLib's native serialization, and can then be loaded back as MLlib models or
deployed as ``python_function`` models. When deployed as a ``python_function``, the model creates its own
SparkContext and converts pandas DataFrame input to a Spark DataFrame before scoring. While this is not
the most efficient solution, especially for real-time scoring, it enables you to easily deploy any MLlib PipelineModel
(as long as the PipelineModel has no external JAR dependencies) to any endpoint supported by
MLflow. For more information, see :py:mod:`mlflow.spark`.

TensorFlow (``tensorflow``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``tensorflow`` model flavor enables logging TensorFlow ``Saved Models`` and loading them back as ``Python Function`` models for inference on pandas DataFrames. Given a directory containing a saved model, you can log the model to MLflow via ``log_saved_model`` and then load the saved model for inference using ``mlflow.pyfunc.load_pyfunc``. For more information, see :py:mod:`mlflow.tensorflow`.

Custom Flavors
--------------
You can add a flavor in MLmodel files, either by writing it directly or
building it with the :py:class:`mlflow.models.Model` class. Choose an arbitrary string name
for your flavor. MLflow tools ignore flavors in the MLmodel file that they do not understand.

Built-In Deployment Tools
-------------------------

MLflow provides tools for deploying models on a local machine and several production environments.
You can use these tools to easily apply your models in a production environment. Not all deployment
methods are available for all model flavors. Deployment is supported for the Python function format and all compatible formats.

Local
^^^^^
MLflow can deploy models locally as local REST API endpoints or to directly score CSV files.
This functionality is a convenient way of testing models before uploading to a remote model server.
You deploy the Python Function flavor locally via the CLI interface to the :py:mod:`mlflow.pyfunc` module.

* :py:func:`serve <mlflow.pyfunc.cli.serve>` deploys the model as a local REST API server.
* :py:func:`predict <mlflow.pyfunc.cli.predict>` uses the model to generate a prediction for a local
  CSV file.

For more info, see:

.. code:: bash

    mlflow pyfunc --help
    mlflow pyfunc serve --help
    mlflow pyfunc predict --help

Microsoft AzureML
^^^^^^^^^^^^^^^^^
The :py:mod:`mlflow.azureml` module can export ``python_function`` models as Azure ML compatible models. It
can also be used to directly deploy and serve models on Azure ML, provided the environment has
been correctly set up.

* :py:func:`export <mlflow.azureml.export>` exports the model in Azure ML-compatible format.
  MLflow will output a directory with the dependencies necessary to deploy the model.

* :py:func:`deploy <mlflow.azureml.deploy>` deploys the model directly to Azure ML.
  You first need to set up your environment to work with the Azure ML CLI. You can do this by
  starting a shell from the Azure ML Workbench application. You also have to set up all accounts
  required to run and deploy on Azure ML. Where the model is deployed is dependent on your
  active Azure ML environment. If the active environment is set up for local deployment, the model
  will be deployed locally in a Docker container (Docker is required).

Model export example:

.. code:: bash

    mlflow azureml export -m <path-to-model> -o test-output
    tree test-output

::
  
    test-output
    ├── create_service.sh  - use this script to upload the model to Azure ML
    ├── score.py - main module required by Azure ML
    └── test-output - directory containing MLflow model in Python Function flavor

.. rubric:: Example workflow using the MLflow CLI

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

Amazon SageMaker
^^^^^^^^^^^^^^^^
The :py:mod:`mlflow.sagemaker` module can deploy ``python_function`` models on SageMaker
or locally in a Docker container with SageMaker compatible environment.
You have to set up your environment and user accounts first in order to
deploy to SageMaker with MLflow. Also, in order to export a custom model to SageMaker, you need a
MLflow-compatible Docker image to be available on Amazon ECR. MLflow provides a default Docker
image definition; however, it is up to you to build the actual image and upload it to ECR.
MLflow includes a utility function to perform this step. Once built and uploaded, the MLflow
container can be used for all MLflow models.

* The :py:func:`build-and-push-container <mlflow.sagemaker.cli.build_and_push_container>` CLI command builds an MLfLow
  Docker image and uploads it to ECR. The caller must have the correct permissions set up. The image
  is built locally and requires Docker to be present on the machine that performs this step.

* :py:func:`run-local <mlflow.sagemaker.run_local>` deploys the model locally in a Docker
  container. The image and the environment should be identical to how the model would be run
  remotely and it is therefore useful for testing the model prior to deployment.

* :py:func:`deploy <mlflow.sagemaker.deploy>` deploys the model on Amazon SageMaker. MLflow
  uploads the Python Function model into S3 and starts an Amazon SageMaker endpoint serving
  the model.

.. rubric:: Example workflow using the MLflow CLI

.. code:: bash

    mlflow sagemaker build-and-push-container  - build the container (only needs to be called once)
    mlflow sagemaker run-local -m <path-to-model>  - test the model locally
    mlflow sagemaker deploy <parameters> - deploy the model to the cloud


For more info, see:

.. code:: bash

    mlflow sagemaker --help
    mlflow sagemaker build-and-push-container --help
    mlflow sagemaker run-local --help
    mlflow sagemaker deploy --help


Apache Spark
^^^^^^^^^^^^
MLfLow can output a ``python_function`` model as an Apache Spark UDF, which can be uploaded to a Spark cluster and
used to score the model.

.. rubric:: Example

.. code:: python

    pyfunc_udf = mlflow.pyfunc.spark_udf(<path-to-model>)
    df = spark_df.withColumn("prediction", pyfunc_udf(<features>))
