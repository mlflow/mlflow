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


.. _model-storage-format:

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

.. code-block:: yaml

    time_created: 2018-05-25T17:28:53.35

    flavors:
      sklearn:
        sklearn_version: 0.19.1
        pickled_model: model.pkl
      python_function:
        loader_module: mlflow.sklearn

This model can then be used with any tool that supports either the ``sklearn`` or
``python_function`` model flavor. For example, the ``mlflow models serve`` command
can serve a model with the ``python_function`` or the ``crate`` (R Function) flavor:

.. code-block:: bash

    mlflow models serve -m my_model

In addition, the ``mlflow sagemaker`` command-line tool can package and deploy models to AWS
SageMaker as long as they support the ``python_function`` flavor:

.. code-block:: bash

    mlflow sagemaker deploy -m my_model [other options]

Fields in the MLmodel Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Apart from a **flavors** field listing the model flavors, the MLmodel YAML format can contain
the following fields:

time_created
    Date and time when the model was created, in UTC ISO 8601 format.

run_id
    ID of the run that created the model, if the model was saved using :ref:`tracking`.

.. _model-api:

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
flavors to benefit from all these tools:

.. contents::
  :local:
  :depth: 1

Python Function (``python_function``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``python_function`` model flavor defines a generic filesystem format for Python models and provides utilities
for saving and loading models to and from this format. The format is self-contained in the sense
that it includes all the information necessary to load and use a model. Dependencies
are stored either directly with the model or referenced via Conda environment.

Many MLflow Model persistence modules, such as :mod:`mlflow.sklearn`, :mod:`mlflow.keras`,
and :mod:`mlflow.pytorch`, produce models with the ``python_function`` (``pyfunc``) flavor. This
means that they adhere to the :ref:`python_function filesystem format <pyfunc-filesystem-format>`
and can be interpreted as generic Python classes that implement the specified
:ref:`inference API <pyfunc-inference-api>`. Therefore, any tool that operates on these ``pyfunc``
classes can operate on any MLflow Model containing the ``pyfunc`` flavor, regardless of which
persistence module or framework was used to produce the model. This interoperability is very
powerful because it allows any Python model to be productionized in a variety of environments.

The convention for ``python_function`` models is to have a ``predict`` method or function with the following
signature:

.. code-block:: py

    predict(model_input: pandas.DataFrame) -> [numpy.ndarray | pandas.Series | pandas.DataFrame]

Other MLflow components expect ``python_function`` models to follow this convention.

The ``python_function`` :ref:`model format <pyfunc-filesystem-format>` is defined as a directory
structure containing all required data, code, and configuration.

The :py:mod:`mlflow.pyfunc` module defines functions for saving and loading MLflow Models with the
``python_function`` flavor. This module also includes utilities for creating custom Python models.
For more information, see the :ref:`custom Python models documentation <custom-python-models>`
and the :mod:`mlflow.pyfunc` documentation.

R Function (``crate``)
^^^^^^^^^^^^^^^^^^^^^^

The ``crate`` model flavor defines a generic model format for representing an arbitrary R prediction
function as an MLflow model. The prediction function is expected to take a dataframe as input and
produce a dataframe, a vector or a list with the predictions as output.

This flavor requires R to be installed in order to be used.

H\ :sub:`2`\ O (``h2o``)
^^^^^^^^^^^^^^^^^^^^^^^^

The ``h2o`` model flavor enables logging and loading H2O models.

The :py:mod:`mlflow.h2o` module defines :py:func:`save_model() <mlflow.h2o.save_model>` and
:py:func:`log_model() <mlflow.h2o.log_model>` methods for saving H2O models in MLflow Model format.
These methods produce MLflow Models with the ``python_function`` flavor, allowing you to load them
as generic Python functions for inference via :py:func:`mlflow.pyfunc.load_model()`. When you load
MLflow Models with the ``h2o`` flavor using :py:func:`mlflow.pyfunc.load_model()`,
the `h2o.init() <http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/h2o.html#h2o.init>`_ method is
called. Therefore, the correct version of ``h2o(-py)`` must be installed in the loader's
environment. You can customize the arguments given to
`h2o.init() <http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/h2o.html#h2o.init>`_ by modifying the
``init`` entry of the persisted H2O model's YAML configuration file: ``model.h2o/h2o.yaml``.

Finally, you can use the :py:func:`mlflow.h2o.load_model()` method to load MLflow Models with the
``h2o`` flavor as H2O model objects.

For more information, see :py:mod:`mlflow.h2o`.

Keras (``keras``)
^^^^^^^^^^^^^^^^^

The ``keras`` model flavor enables logging and loading Keras models. It is available in both Python
and R clients. The :py:mod:`mlflow.keras` module defines :py:func:`save_model()<mlflow.keras.save_model>`
and :py:func:`log_model() <mlflow.keras.log_model>` functions that you can use to save Keras models
in MLflow Model format in Python. Similarly, in R, you can save or log the model using
`mlflow_save_model <R-api.rst#mlflow-save-model>`__ and `mlflow_log_model <R-api.rst#mlflow-log-model>`__. These functions serialize Keras
models as HDF5 files using the Keras library's built-in model persistence functions. MLflow Models
produced by these functions also contain the ``python_function`` flavor, allowing them to be interpreted
as generic Python functions for inference via :py:func:`mlflow.pyfunc.load_model()`. Finally, you
can use the :py:func:`mlflow.keras.load_model()` function in Python or `mlflow_load_model <R-api.rst#mlflow-load-model>`__
function in R to load MLflow Models with the ``keras`` flavor as
`Keras Model objects <https://keras.io/models/about-keras-models/>`_.

For more information, see :py:mod:`mlflow.keras`.

MLeap (``mleap``)
^^^^^^^^^^^^^^^^^

The ``mleap`` model flavor supports saving Spark models in MLflow format using the
`MLeap <http://mleap-docs.combust.ml/>`_ persistence mechanism. MLeap is an inference-optimized
format and execution engine for Spark models that does not depend on
`SparkContext <https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.SparkContext>`_
to evaluate inputs.

You can save Spark models in MLflow format with the ``mleap`` flavor by specifying the
``sample_input`` argument of the :py:func:`mlflow.spark.save_model()` or
:py:func:`mlflow.spark.log_model()` method (recommended). The :py:mod:`mlflow.mleap` module also
defines :py:func:`save_model() <mlflow.mleap.save_model>` and
:py:func:`log_model() <mlflow.mleap.log_model>` methods for saving MLeap models in MLflow format,
but these methods do not include the ``python_function`` flavor in the models they produce.

A companion module for loading MLflow Models with the MLeap flavor is available in the
``mlflow/java`` package.

For more information, see :py:mod:`mlflow.spark`, :py:mod:`mlflow.mleap`, and the
`MLeap documentation <http://mleap-docs.combust.ml/>`_.

PyTorch (``pytorch``)
^^^^^^^^^^^^^^^^^^^^^

The ``pytorch`` model flavor enables logging and loading PyTorch models.

The :py:mod:`mlflow.pytorch` module defines utilities for saving and loading MLflow Models with the
``pytorch`` flavor. You can use the :py:func:`mlflow.pytorch.save_model()` and
:py:func:`mlflow.pytorch.log_model()` methods to save PyTorch models in MLflow format; both of these
functions use the `torch.save() <https://pytorch.org/docs/stable/torch.html#torch.save>`_ method to
serialize PyTorch models. Additionally, you can use the :py:func:`mlflow.pytorch.load_model()`
method to load MLflow Models with the ``pytorch`` flavor as PyTorch model objects. Finally, models
produced by :py:func:`mlflow.pytorch.save_model()` and :py:func:`mlflow.pytorch.log_model()` contain
the ``python_function`` flavor, allowing you to load them as generic Python functions for inference
via :py:func:`mlflow.pyfunc.load_model()`.

For more information, see :py:mod:`mlflow.pytorch`.

Scikit-learn (``sklearn``)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``sklearn`` model flavor provides an easy-to-use interface for saving and loading scikit-learn
models. The :py:mod:`mlflow.sklearn` module defines
:py:func:`save_model() <mlflow.sklearn.save_model>` and
:py:func:`log_model() <mlflow.sklearn.log_model>` functions that save scikit-learn models in
MLflow format, using either Python's pickle module (Pickle) or CloudPickle for model serialization.
These functions produce MLflow Models with the ``python_function`` flavor, allowing them to
be loaded as generic Python functions for inference via :py:func:`mlflow.pyfunc.load_model()`.
Finally, you can use the :py:func:`mlflow.sklearn.load_model()` method to load MLflow Models with
the ``sklearn`` flavor as scikit-learn model objects.

For more information, see :py:mod:`mlflow.sklearn`.

Spark MLlib (``spark``)
^^^^^^^^^^^^^^^^^^^^^^^

The ``spark`` model flavor enables exporting Spark MLlib models as MLflow Models.

The :py:mod:`mlflow.spark` module defines :py:func:`save_model() <mlflow.spark.save_model>` and
:py:func:`log_model() <mlflow.spark.log_model>` methods that save Spark MLlib pipelines in MLflow
model format. MLflow Models produced by these functions contain the ``python_function`` flavor,
allowing you to load them as generic Python functions via :py:func:`mlflow.pyfunc.load_model()`.
When a model with the ``spark`` flavor is loaded as a Python function via
:py:func:`mlflow.pyfunc.load_model()`, a new
`SparkContext <https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.SparkContext>`_
is created for model inference; additionally, the function converts all Pandas DataFrame inputs to
Spark DataFrames before scoring. While this initialization overhead and format translation latency
is not ideal for high-performance use cases, it enables you to easily deploy any
`MLlib PipelineModel <http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=
pipelinemodel#pyspark.ml.Pipeline>`_ to any production environment supported by MLflow
(SageMaker, AzureML, etc).

Finally, the :py:func:`mlflow.spark.load_model()` method is used to load MLflow Models with
the ``spark`` flavor as Spark MLlib pipelines.

For more information, see :py:mod:`mlflow.spark`.

TensorFlow (``tensorflow``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``tensorflow`` model flavor allows serialized TensorFlow models in
`SavedModel format <https://www.tensorflow.org/guide/saved_model#save_and_restore_models>`_
to be logged in MLflow format via the :py:func:`mlflow.tensorflow.save_model()` and
:py:func:`mlflow.tensorflow.log_model()` methods. These methods also add the ``python_function``
flavor to the MLflow Models that they produce, allowing the models to be interpreted as generic
Python functions for inference via :py:func:`mlflow.pyfunc.load_model()`. Finally, you can use the
:py:func:`mlflow.tensorflow.load_model()` method to load MLflow Models with the ``tensorflow``
flavor as TensorFlow graphs.

For more information, see :py:mod:`mlflow.tensorflow`.

ONNX (``onnx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``onnx`` model flavor enables logging of `ONNX models <http://onnx.ai/>`_ in MLflow format via
the :py:func:`mlflow.onnx.save_model()` and :py:func:`mlflow.onnx.log_model()` methods. These
methods also add the ``python_function`` flavor to the MLflow Models that they produce, allowing the
models to be interpreted as generic Python functions for inference via
:py:func:`mlflow.pyfunc.load_model()`. The ``python_function`` representation of an MLflow
ONNX model uses the `ONNX Runtime execution engine <https://github.com/microsoft/onnxruntime>`_ for
evaluation. Finally, you can use the :py:func:`mlflow.onnx.load_model()` method to load MLflow
Models with the ``onnx`` flavor in native ONNX format.

For more information, see :py:mod:`mlflow.onnx` and `<http://onnx.ai/>`_.

MXNet Gluon (``gluon``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``gluon`` model flavor enables logging of `Gluon models
<https://mxnet.incubator.apache.org/api/python/docs/api/gluon/index.html>`_ in MLflow format via
the :py:func:`mlflow.gluon.save_model()` and :py:func:`mlflow.gluon.log_model()` methods. These
methods also add the ``python_function`` flavor to the MLflow Models that they produce, allowing the
models to be interpreted as generic Python functions for inference via
:py:func:`mlflow.pyfunc.load_model()`. You can also use the :py:func:`mlflow.gluon.load_model()`
method to load MLflow Models with the ``gluon`` flavor in native Gluon format.

For more information, see :py:mod:`mlflow.gluon`.

XGBoost (``xgboost``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``xgboost`` model flavor enables logging of `XGBoost models
<https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster>`_
in MLflow format via the :py:func:`mlflow.xgboost.save_model()` and :py:func:`mlflow.xgboost.log_model()` methods.
These methods also add the ``python_function`` flavor to the MLflow Models that they produce, allowing the
models to be interpreted as generic Python functions for inference via
:py:func:`mlflow.pyfunc.load_model()`. You can also use the :py:func:`mlflow.xgboost.load_model()`
method to load MLflow Models with the ``xgboost`` model flavor in native XGBoost format.

Note that the ``xgboost`` model flavor only supports an instance of `xgboost.Booster
<https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster>`_,
not models that implement the `scikit-learn API
<https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn>`__.

For more information, see :py:mod:`mlflow.xgboost`.

LightGBM (``lightgbm``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``lightgbm`` model flavor enables logging of `LightGBM models
<https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm-booster>`_
in MLflow format via the :py:func:`mlflow.lightgbm.save_model()` and :py:func:`mlflow.lightgbm.log_model()` methods.
These methods also add the ``python_function`` flavor to the MLflow Models that they produce, allowing the
models to be interpreted as generic Python functions for inference via
:py:func:`mlflow.pyfunc.load_model()`. You can also use the :py:func:`mlflow.lightgbm.load_model()`
method to load MLflow Models with the ``lightgbm`` model flavor in native LightGBM format.

Note that the ``lightgbm`` model flavor only supports an instance of `lightgbm.Booster
<https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm-booster>`__,
not models that implement the `scikit-learn API
<https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api>`_.

For more information, see :py:mod:`mlflow.lightgbm`.

Model Customization
-------------------

While MLflow's built-in model persistence utilities are convenient for packaging models from various
popular ML libraries in MLflow Model format, they do not cover every use case. For example, you may
want to use a model from an ML library that is not explicitly supported by MLflow's built-in
flavors. Alternatively, you may want to package custom inference code and data to create an
MLflow Model. Fortunately, MLflow provides two solutions that can be used to accomplish these
tasks: :ref:`custom-python-models` and :ref:`custom-flavors`.

.. contents:: In this section:
  :local:
  :depth: 2

.. _custom-python-models:

Custom Python Models
^^^^^^^^^^^^^^^^^^^^
The :py:mod:`mlflow.pyfunc` module provides :py:func:`save_model() <mlflow.pyfunc.save_model>` and
:py:func:`log_model() <mlflow.pyfunc.log_model>` utilities for creating MLflow Models with the
``python_function`` flavor that contain user-specified code and *artifact* (file) dependencies.
These artifact dependencies may include serialized models produced by any Python ML library.

Because these custom models contain the ``python_function`` flavor, they can be deployed
to any of MLflow's supported production environments, such as SageMaker, AzureML, or local
REST endpoints.

The following examples demonstrate how you can use the :py:mod:`mlflow.pyfunc` module to create
custom Python models. For additional information about model customization with MLflow's
``python_function`` utilities, see the
:ref:`python_function custom models documentation <pyfunc-create-custom>`.

Example: Creating a custom "add n" model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example defines a class for a custom model that adds a specified numeric value, ``n``, to all
columns of a Pandas DataFrame input. Then, it uses the :py:mod:`mlflow.pyfunc` APIs to save an
instance of this model with ``n = 5`` in MLflow Model format. Finally, it loads the model in
``python_function`` format and uses it to evaluate a sample input.

.. code-block:: py

    import mlflow.pyfunc

    # Define the model class
    class AddN(mlflow.pyfunc.PythonModel):

        def __init__(self, n):
            self.n = n

        def predict(self, context, model_input):
            return model_input.apply(lambda column: column + self.n)

    # Construct and save the model
    model_path = "add_n_model"
    add5_model = AddN(n=5)
    mlflow.pyfunc.save_model(path=model_path, python_model=add5_model)

    # Load the model in `python_function` format
    loaded_model = mlflow.pyfunc.load_model(model_path)

    # Evaluate the model
    import pandas as pd
    model_input = pd.DataFrame([range(10)])
    model_output = loaded_model.predict(model_input)
    assert model_output.equals(pd.DataFrame([range(5, 15)]))

Example: Saving an XGBoost model in MLflow format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example begins by training and saving a gradient boosted tree model using the XGBoost
library. Next, it defines a wrapper class around the XGBoost model that conforms to MLflow's
``python_function`` :ref:`inference API <pyfunc-inference-api>`. Then, it uses the wrapper class and
the saved XGBoost model to construct an MLflow Model that performs inference using the gradient
boosted tree. Finally, it loads the MLflow Model in ``python_function`` format and uses it to
evaluate test data.

.. code-block:: py

    # Load training and test datasets
    import xgboost as xgb
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()
    x = iris.data[:, 2:]
    y = iris.target
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(x_train, label=y_train)

    # Train and save an XGBoost model
    xgb_model = xgb.train(params={'max_depth': 10}, dtrain=dtrain, num_boost_round=10)
    xgb_model_path = "xgb_model.pth"
    xgb_model.save_model(xgb_model_path)

    # Create an `artifacts` dictionary that assigns a unique name to the saved XGBoost model file.
    # This dictionary will be passed to `mlflow.pyfunc.save_model`, which will copy the model file
    # into the new MLflow Model's directory.
    artifacts = {
        "xgb_model": xgb_model_path
    }

    # Define the model class
    import mlflow.pyfunc
    class XGBWrapper(mlflow.pyfunc.PythonModel):

        def load_context(self, context):
            import xgboost as xgb
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(context.artifacts["xgb_model"])

        def predict(self, context, model_input):
            input_matrix = xgb.DMatrix(model_input.values)
            return self.xgb_model.predict(input_matrix)

    # Create a Conda environment for the new MLflow Model that contains the XGBoost library
    # as a dependency, as well as the required CloudPickle library
    import cloudpickle
    conda_env = {
        'channels': ['defaults'],
        'dependencies': [
          'xgboost={}'.format(xgb.__version__),
          'cloudpickle={}'.format(cloudpickle.__version__),
        ],
        'name': 'xgb_env'
    }

    # Save the MLflow Model
    mlflow_pyfunc_model_path = "xgb_mlflow_pyfunc"
    mlflow.pyfunc.save_model(
            path=mlflow_pyfunc_model_path, python_model=XGBWrapper(), artifacts=artifacts,
            conda_env=conda_env)

    # Load the model in `python_function` format
    loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

    # Evaluate the model
    import pandas as pd
    test_predictions = loaded_model.predict(pd.DataFrame(x_test))
    print(test_predictions)

.. _custom-flavors:

Custom Flavors
^^^^^^^^^^^^^^
You can also create custom MLflow Models by writing a custom *flavor*.

As discussed in the :ref:`model-api` and :ref:`model-storage-format` sections, an MLflow Model
is defined by a directory of files that contains an ``MLmodel`` configuration file. This ``MLmodel``
file describes various model attributes, including the flavors in which the model can be
interpreted. The ``MLmodel`` file contains an entry for each flavor name; each entry is
a YAML-formatted collection of flavor-specific attributes.

To create a new flavor to support a custom model, you define the set of flavor-specific attributes
to include in the ``MLmodel`` configuration file, as well as the code that can interpret the
contents of the model directory and the flavor's attributes.

As an example, let's examine the :py:mod:`mlflow.pytorch` module corresponding to MLflow's
``pytorch`` flavor. In the :py:func:`mlflow.pytorch.save_model()` method, a PyTorch model is saved
to a specified output directory. Additionally, :py:func:`mlflow.pytorch.save_model()` leverages the
:py:func:`mlflow.models.Model.add_flavor()` and :py:func:`mlflow.models.Model.save()` functions to
produce an ``MLmodel`` configuration containing the ``pytorch`` flavor. The resulting configuration
has several flavor-specific attributes, such as ``pytorch_version``, which denotes the version of the
PyTorch library that was used to train the model. To interpret model directories produced by
:py:func:`save_model() <mlflow.pytorch.save_model>`, the :py:mod:`mlflow.pytorch` module also
defines a :py:mod:`load_model() <mlflow.pytorch.load_model>` method.
:py:mod:`mlflow.pytorch.load_model()` reads the ``MLmodel`` configuration from a specified
model directory and uses the configuration attributes of the ``pytorch`` flavor to load
and return a PyTorch model from its serialized representation.

Built-In Deployment Tools
-------------------------

MLflow provides tools for deploying MLflow models on a local machine and to several production environments.
Not all deployment methods are available for all model flavors.

.. contents:: In this section:
  :local:
  :depth: 1

.. _local_model_deployment:

Deploy MLflow models
^^^^^^^^^^^^^^^^^^^^
MLflow can deploy models locally as local REST API endpoints or to directly score files. In addition,
MLflow can package models as self-contained Docker images with the REST API endpoint. The image can
be used to safely deploy the model to various environments such as Kubernetes.

You deploy MLflow model locally or generate a Docker image using the CLI interface to the
:py:mod:`mlflow.models` module.

The REST API server accepts the following data formats as POST input to the ``/invocations`` path:

* JSON-serialized pandas DataFrames in the ``split`` orientation. For example,
  ``data = pandas_df.to_json(orient='split')``. This format is specified using a ``Content-Type``
  request header value of ``application/json`` or ``application/json; format=pandas-split``.

* JSON-serialized pandas DataFrames in the ``records`` orientation. *We do not recommend using
  this format because it is not guaranteed to preserve column ordering.* This format is
  specified using a ``Content-Type`` request header value of
  ``application/json; format=pandas-records``.

* CSV-serialized pandas DataFrames. For example, ``data = pandas_df.to_csv()``. This format is
  specified using a ``Content-Type`` request header value of ``text/csv``.

Example requests:

.. code-block:: bash

    # split-oriented
    curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
        "columns": ["a", "b", "c"],
        "data": [[1, 2, 3], [4, 5, 6]]
    }'

    # record-oriented (fine for vector rows, loses ordering for JSON records)
    curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json; format=pandas-records' -d '[
        {"a": 1,"b": 2,"c": 3},
        {"a": 4,"b": 5,"c": 6}
    ]'


For more information about serializing pandas DataFrames, see
`pandas.DataFrame.to_json <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_json.html>`_.

The predict command accepts the same input formats. The format is specified as command line arguments.

Commands
~~~~~~~~

* `serve <cli.html#mlflow-models-serve>`_ deploys the model as a local REST API server.
* `build_docker <cli.html#mlflow-models-build-docker>`_ packages a REST API endpoint serving the
  model as a docker image.
* `predict <cli.html#mlflow-models-predict>`_ uses the model to generate a prediction for a local
  CSV or JSON file.

For more info, see:

.. code-block:: bash

    mlflow models --help
    mlflow models serve --help
    mlflow models predict --help
    mlflow models build-docker --help

.. _azureml_deployment:

Deploy a ``python_function`` model on Microsoft Azure ML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:mod:`mlflow.azureml` module can package ``python_function`` models into Azure ML container images and deploy them as a webservice. Models can be deployed to Azure Kubernetes Service (AKS) and the Azure Container Instances (ACI)
platform for real-time serving. The resulting Azure ML ContainerImage contains a web server that
accepts the following data formats as input:

* JSON-serialized pandas DataFrames in the ``split`` orientation. For example, ``data = pandas_df.to_json(orient='split')``. This format is specified using a ``Content-Type`` request header value of ``application/json``.

* :py:func:`mlflow.azureml.deploy` registers an MLflow Model with an existing Azure ML workspace, builds an Azure ML container image and deploys the model to AKS and ACI. The `Azure ML SDK`_ is required in order to use this function. *The Azure ML SDK requires Python 3. It cannot be installed with earlier versions of Python.*

.. _Azure ML SDK: https://docs.microsoft.com/python/api/overview/azure/ml/intro?view=azure-ml-py

.. rubric:: Example workflow using the Python API

.. code-block:: py

    import mlflow.azureml

    from azureml.core import Workspace
    from azureml.core.webservice import AciWebservice, Webservice


    # Create or load an existing Azure ML workspace. You can also load an existing workspace using
    # Workspace.get(name="<workspace_name>")
    workspace_name = "<Name of your Azure ML workspace>"
    subscription_id = "<Your Azure subscription ID>"
    resource_group = "<Name of the Azure resource group in which to create Azure ML resources>"
    location = "<Name of the Azure location (region) in which to create Azure ML resources>"
    azure_workspace = Workspace.create(name=workspace_name,
                                       subscription_id=subscription_id,
                                       resource_group=resource_group,
                                       location=location,
                                       create_resource_group=True,
                                       exist_okay=True)
    # Create a deployment config
    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)                                       

    # Register and deploy model to Azure Container Instance (ACI)
    (webservice, model) = mlflow.azureml.deploy(model_uri='<your-model-uri>',
                                                workspace=azure_workspace,
                                                model_name='mymodelname', 
                                                service_name='myservice', 
                                                deployment_config=aci_config)
    # After the model deployment completes, requests can be posted via HTTP to the new ACI
    # webservice's scoring URI. The following example posts a sample input from the wine dataset
    # used in the MLflow ElasticNet example:
    # https://github.com/mlflow/mlflow/tree/master/examples/sklearn_elasticnet_wine
    print("Scoring URI is: %s", webservice.scoring_uri)

    import requests
    import json

    # `sample_input` is a JSON-serialized pandas DataFrame with the `split` orientation
    sample_input = {
        "columns": [
            "alcohol",
            "chlorides",
            "citric acid",
            "density",
            "fixed acidity",
            "free sulfur dioxide",
            "pH",
            "residual sugar",
            "sulphates",
            "total sulfur dioxide",
            "volatile acidity"
        ],
        "data": [
            [8.8, 0.045, 0.36, 1.001, 7, 45, 3, 20.7, 0.45, 170, 0.27]
        ]
    }
    response = requests.post(
                  url=webservice.scoring_uri, data=json.dumps(sample_input),
                  headers={"Content-type": "application/json"})
    response_json = json.loads(response.text)
    print(response_json)

.. _sagemaker_deployment:

Deploy a ``python_function`` model on Amazon SageMaker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:mod:`mlflow.sagemaker` module can deploy ``python_function`` models locally in a Docker
container with SageMaker compatible environment and remotely on SageMaker.
To deploy remotely to SageMaker you need to set up your environment and user accounts.
To export a custom model to SageMaker, you need a MLflow-compatible Docker image to be available on Amazon ECR.
MLflow provides a default Docker image definition; however, it is up to you to build the image and upload it to ECR.
MLflow includes the utility function ``build_and_push_container`` to perform this step. Once built and uploaded, you can use the MLflow container for all MLflow Models. Model webservers deployed using the :py:mod:`mlflow.sagemaker`
module accept the following data formats as input, depending on the deployment flavor:

* ``python_function``: For this deployment flavor, the endpoint accepts the same formats described
  in the :ref:`local model deployment documentation <local_model_deployment>`.

* ``mleap``: For this deployment flavor, the endpoint accepts `only`
  JSON-serialized pandas DataFrames in the ``split`` orientation. For example,
  ``data = pandas_df.to_json(orient='split')``. This format is specified using a ``Content-Type``
  request header value of ``application/json``.

Commands
~~~~~~~~~

* :py:func:`run-local <mlflow.sagemaker.run_local>` deploys the model locally in a Docker
  container. The image and the environment should be identical to how the model would be run
  remotely and it is therefore useful for testing the model prior to deployment.

* `build-and-push-container <cli.html#mlflow-sagemaker-build-and-push-container>`_ builds an MLfLow
  Docker image and uploads it to ECR. The caller must have the correct permissions set up. The image
  is built locally and requires Docker to be present on the machine that performs this step.

* :py:func:`deploy <mlflow.sagemaker.deploy>` deploys the model on Amazon SageMaker. MLflow
  uploads the Python Function model into S3 and starts an Amazon SageMaker endpoint serving
  the model.

.. rubric:: Example workflow using the MLflow CLI

.. code-block:: bash

    mlflow sagemaker build-and-push-container  - build the container (only needs to be called once)
    mlflow sagemaker run-local -m <path-to-model>  - test the model locally
    mlflow sagemaker deploy <parameters> - deploy the model remotely


For more info, see:

.. code-block:: bash

    mlflow sagemaker --help
    mlflow sagemaker build-and-push-container --help
    mlflow sagemaker run-local --help
    mlflow sagemaker deploy --help


Export a ``python_function`` model as an Apache Spark UDF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can output a ``python_function`` model as an Apache Spark UDF, which can be uploaded to a
Spark cluster and used to score the model.

.. rubric:: Example

.. code-block:: py

    pyfunc_udf = mlflow.pyfunc.spark_udf(<path-to-model>)
    df = spark_df.withColumn("prediction", pyfunc_udf(<features>))

The resulting UDF is based on Spark's Pandas UDF and is currently limited to producing either a single
value or an array of values of the same type per observation. By default, we return the first
numeric column as a double. You can control what result is returned by supplying ``result_type``
argument. The following values are supported:

* ``'int'`` or IntegerType_: The leftmost integer that can fit in
  ``int32`` result is returned or exception is raised if there is none.
* ``'long'`` or LongType_: The leftmost long integer that can fit in ``int64``
  result is returned or exception is raised if there is none.
* ArrayType_ (IntegerType_ | LongType_): Return all integer columns that can fit
  into the requested size.
* ``'float'`` or FloatType_: The leftmost numeric result cast to
  ``float32`` is returned or exception is raised if there is no numeric column.
* ``'double'`` or DoubleType_: The leftmost numeric result cast to
  ``double`` is returned or exception is raised if there is no numeric column.
* ArrayType_ ( FloatType_ | DoubleType_ ): Return all numeric columns cast to the
  requested. type. Exception is raised if there are numeric columns.
* ``'string'`` or StringType_: Result is the leftmost column converted to string.
* ArrayType_ ( StringType_ ): Return all columns converted to string.

.. _IntegerType: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.types.IntegerType
.. _LongType: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.types.LongType
.. _FloatType: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.types.FloatType
.. _DoubleType: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.types.DoubleType
.. _StringType: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.types.StringType
.. _ArrayType: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.types.ArrayType

.. rubric:: Example

.. code-block:: py

    from pyspark.sql.types import ArrayType, FloatType
    pyfunc_udf = mlflow.pyfunc.spark_udf(<path-to-model>, result_type=ArrayType(FloatType()))
    # The prediction column will contain all the numeric columns returned by the model as floats
    df = spark_df.withColumn("prediction", pyfunc_udf(<features>))
