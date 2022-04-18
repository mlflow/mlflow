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
    ├── model.pkl
    ├── conda.yaml
    └── requirements.txt
    

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

signature
  :ref:`model signature <model-signature>` in JSON format.

input_example
  reference to an artifact with :ref:`input example <input-example>`.

databricks_runtime
    Databricks runtime version and type, if the model was trained in a Databricks notebook or job.




Additional Logged Files
^^^^^^^^^^^^^^^^^^^^^^^
For environment recreation, we automatically log ``conda.yaml`` and ``requirements.txt`` files whenever a model is logged. These files can then be used to reinstall dependencies using either ``conda`` or ``pip``.

.. note::
    Anaconda Inc. updated their `terms of service <https://www.anaconda.com/terms-of-service>`_ for anaconda.org channels. Based on the new terms of service you may require a commercial license if you rely on Anaconda’s packaging and distribution. See `Anaconda Commercial Edition FAQ <https://www.anaconda.com/blog/anaconda-commercial-edition-faq>`_ for more information. Your use of any Anaconda channels is governed by their terms of service.

    MLflow models logged before `v1.18 <https://mlflow.org/news/2021/06/18/1.18.0-release/index.html>`_ were by default logged with the conda ``defaults`` channel (`https://repo.anaconda.com/pkgs/ <https://repo.anaconda.com/pkgs/>`_) as a dependency. Because of this license change, MLflow has stopped the use of the ``defaults`` channel for models logged using MLflow v1.18 and above. The default channel logged is now ``conda-forge``, which points at the community managed `https://conda-forge.org/ <https://conda-forge.org/>`_.

    If you logged a model before MLflow v1.18 without excluding the ``defaults`` channel from the conda environment for the model, that model may have a dependency on the ``defaults`` channel that you may not have intended.
    To manually confirm whether a model has this dependency, you can examine ``channel`` value in the ``conda.yaml`` file that is packaged with the logged model. For example, a model’s ``conda.yaml`` with a ``defaults`` channel dependency may look like this:

    .. code-block:: yaml

        name: mlflow-env
        channels:
        - defaults
        dependencies:
        - python=3.8.8
        - pip
        - pip:
            - mlflow
            - scikit-learn==0.23.2
            - cloudpickle==1.6.0

    If you would like to change the channel used in a model’s environment, you can re-register the model to the model registry with a new ``conda.yaml``. You can do this by specifying the channel in the ``conda_env`` parameter of ``log_model()``.

    For more information on the ``log_model()`` API, see the MLflow documentation for the model flavor you are working with, for example, :py:func:`mlflow.sklearn.log_model() <mlflow.sklearn.log_model>`.

conda.yaml
    When saving a model, MLflow provides the option to pass in a conda environment parameter that can contain dependencies used by the model. If no conda environment is provided, a default environment is created based on the flavor of the model. This conda environment is then saved in ``conda.yaml``.
requirements.txt
    The requirements file is created from the `pip portion <https://www.anaconda.com/blog/using-pip-in-a-conda-environment>`_ of the ``conda.yaml`` environment specification. Additional pip dependencies can be added to ``requirements.txt`` by including them as a pip dependency in a conda environment and logging the model with the environment. 

The following shows an example of saving a model with a manually specified conda environment and the corresponding content of the generated ``conda.yaml`` and ``requirements.txt`` files.

.. code-block:: py

    conda_env = {
        'channels': ['conda-forge'],
        'dependencies': [
            'python=3.8.8',
            'pip'],
        'pip': [
            'mlflow',
            'scikit-learn==0.23.2',
            'cloudpickle==1.6.0'
        ],
        'name': 'mlflow-env'
    }
    mlflow.sklearn.log_model(model, "my_model", conda_env=conda_env)

The written ``conda.yaml`` file:

.. code-block:: yaml

    name: mlflow-env
    channels:
      - conda-forge
    dependencies:
    - python=3.8.8
    - pip
    - pip:
      - mlflow
      - scikit-learn==0.23.2
      - cloudpickle==1.6.0

The written ``requirements.txt`` file:

.. code-block:: text

    mlflow
    scikit-learn==0.23.2
    cloudpickle==1.6.0

.. _model-metadata:

Model Signature And Input Example
---------------------------------
When working with ML models you often need to know some basic functional properties of the model
at hand, such as "What inputs does it expect?" and "What output does it produce?". MLflow models can
include the following additional metadata about model inputs and outputs that can be used by
downstream tooling:

* :ref:`Model Signature <model-signature>` - description of a model's inputs and outputs.
* :ref:`Model Input Example <input-example>` - example of a valid model input.

.. _model-signature:

Model Signature
^^^^^^^^^^^^^^^
The Model signature defines the schema of a model's inputs and outputs. Model inputs and outputs can
be either column-based or tensor-based. Column-based inputs and outputs can be described as a
sequence of (optionally) named columns with type specified as one of the
:py:class:`MLflow data types <mlflow.types.DataType>`. Tensor-based inputs and outputs can be
described as a sequence of (optionally) named tensors with type specified as one of the
`numpy data types <https://numpy.org/devdocs/user/basics.types.html>`_.

To include a signature with your model, pass a :py:class:`signature object
<mlflow.models.ModelSignature>` as an argument to the appropriate log_model call, e.g.
:py:func:`sklearn.log_model() <mlflow.sklearn.log_model>`. More details are in the :ref:`How to log models with signatures <how-to-log-models-with-signatures>` section. The signature is stored in
JSON format in the :ref:`MLmodel file <pyfunc-model-config>`, together with other model metadata.

Model signatures are recognized and enforced by standard :ref:`MLflow model deployment tools
<built-in-deployment>`. For example, the :ref:`mlflow models serve <local_model_deployment>` tool,
which deploys a model as a REST API, validates inputs based on the model's signature.


Column-based Signature Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All flavors support column-based signatures.

Each column-based input and output is represented by a type corresponding to one of 
:py:class:`MLflow data types <mlflow.types.DataType>` and an optional name. The following example
displays an MLmodel file excerpt containing the model signature for a classification model trained on
the `Iris dataset <https://archive.ics.uci.edu/ml/datasets/iris>`_. The input has 4 named, numeric columns.
The output is an unnamed integer specifying the predicted class.

.. code-block:: yaml

  signature:
      inputs: '[{"name": "sepal length (cm)", "type": "double"}, {"name": "sepal width
        (cm)", "type": "double"}, {"name": "petal length (cm)", "type": "double"}, {"name":
        "petal width (cm)", "type": "double"}]'
      outputs: '[{"type": "integer"}]'
      
Tensor-based Signature Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Only DL flavors support tensor-based signatures (i.e TensorFlow, Keras, PyTorch, Onnx, and Gluon).

Each tensor-based input and output is represented by a dtype corresponding to one of
`numpy data types <https://numpy.org/devdocs/user/basics.types.html>`_, shape and an optional name.
When specifying the shape, -1 is used for axes that may be variable in size.
The following example displays an MLmodel file excerpt containing the model signature for a
classification model trained on the `MNIST dataset <http://yann.lecun.com/exdb/mnist/>`_.
The input has one named tensor where input sample is an image represented by a 28 × 28 × 1 array
of float32 numbers. The output is an unnamed tensor that has 10 units specifying the
likelihood corresponding to each of the 10 classes. Note that the first dimension of the input
and the output is the batch size and is thus set to -1 to allow for variable batch sizes. 

.. code-block:: yaml

  signature:
      inputs: '[{"name": "images", "dtype": "uint8", "shape": [-1, 28, 28, 1]}]'
      outputs: '[{"shape": [-1, 10], "dtype": "float32"}]'

Signature Enforcement
~~~~~~~~~~~~~~~~~~~~~
Schema enforcement checks the provided input against the model's signature
and raises an exception if the input is not compatible. This enforcement is applied in MLflow before
calling the underlying model implementation. Note that this enforcement only applies when using :ref:`MLflow
model deployment tools <built-in-deployment>` or when loading models as ``python_function``. In
particular, it is not applied to models that are loaded in their native format (e.g. by calling
:py:func:`mlflow.sklearn.load_model() <mlflow.sklearn.load_model>`).

Name Ordering Enforcement
"""""""""""""""""""""""""
The input names are checked against the model signature. If there are any missing inputs,
MLflow will raise an exception. Extra inputs that were not declared in the signature will be
ignored. If the input schema in the signature defines input names, input matching is done by name
and the inputs are reordered to match the signature. If the input schema does not have input
names, matching is done by position (i.e. MLflow will only check the number of inputs).

Input Type Enforcement
"""""""""""""""""""""""
The input types are checked against the signature.

For models with column-based signatures (i.e DataFrame inputs), MLflow will perform safe type conversions
if necessary. Generally, only conversions that are guaranteed to be lossless are allowed. For
example, int -> long or int -> double conversions are ok, long -> double is not. If the types cannot
be made compatible, MLflow will raise an error.

For models with tensor-based signatures, type checking is strict (i.e an exception will be thrown if
the input type does not match the type specified by the schema). 

Handling Integers With Missing Values
"""""""""""""""""""""""""""""""""""""
Integer data with missing values is typically represented as floats in Python. Therefore, data
types of integer columns in Python can vary depending on the data sample. This type variance can
cause schema enforcement errors at runtime since integer and float are not compatible types. For
example, if your training data did not have any missing values for integer column c, its type will
be integer. However, when you attempt to score a sample of the data that does include a missing
value in column c, its type will be float. If your model signature specified c to have integer type,
MLflow will raise an error since it can not convert float to int. Note that MLflow uses python to
serve models and to deploy models to Spark, so this can affect most model deployments. The best way
to avoid this problem is to declare integer columns as doubles (float64) whenever there can be
missing values.

Handling Date and Timestamp
"""""""""""""""""""""""""""
For datetime values, Python has precision built into the type. For example, datetime values with
day precision have NumPy type ``datetime64[D]``, while values with nanosecond precision have
type ``datetime64[ns]``. Datetime precision is ignored for column-based model signature but is
enforced for tensor-based signatures.

.. _how-to-log-models-with-signatures:

How To Log Models With Signatures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To include a signature with your model, pass :py:class:`signature object
<mlflow.models.ModelSignature>` as an argument to the appropriate log_model call, e.g.
:py:func:`sklearn.log_model() <mlflow.sklearn.log_model>`. The model signature object can be created
by hand or :py:func:`inferred <mlflow.models.infer_signature>` from datasets with valid model inputs
(e.g. the training dataset with target column omitted) and valid model outputs (e.g. model
predictions generated on the training dataset).

Column-based Signature Example
""""""""""""""""""""""""""""""
The following example demonstrates how to store a model signature for a simple classifier trained
on the ``Iris dataset``:

.. code-block:: python

    import pandas as pd
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature

    iris = datasets.load_iris()
    iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
    clf = RandomForestClassifier(max_depth=7, random_state=0)
    clf.fit(iris_train, iris.target)
    signature = infer_signature(iris_train, clf.predict(iris_train))
    mlflow.sklearn.log_model(clf, "iris_rf", signature=signature)

The same signature can be created explicitly as follows:

.. code-block:: python

    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Schema, ColSpec

    input_schema = Schema([
      ColSpec("double", "sepal length (cm)"),
      ColSpec("double", "sepal width (cm)"),
      ColSpec("double", "petal length (cm)"),
      ColSpec("double", "petal width (cm)"),
    ])
    output_schema = Schema([ColSpec("long")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

Tensor-based Signature Example
""""""""""""""""""""""""""""""
The following example demonstrates how to store a model signature for a simple classifier trained
on the ``MNIST dataset``:

.. code-block:: python

    from keras.datasets import mnist
    from keras.utils import to_categorical
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
    from keras.optimizers import SGD
    import mlflow
    import mlflow.keras
    from mlflow.models.signature import infer_signature

    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    trainX = train_X.reshape((train_X.shape[0], 28, 28, 1))
    testX = test_X.reshape((test_X.shape[0], 28, 28, 1))
    trainY = to_categorical(train_Y)
    testY = to_categorical(test_Y)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY))

    signature = infer_signature(testX, model.predict(testX))
    mlflow.keras.log_model(model, "mnist_cnn", signature=signature)

The same signature can be created explicitly as follows:

.. code-block:: python

    import numpy as np
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Schema, TensorSpec

    input_schema = Schema([
      TensorSpec(np.dtype(np.uint8), (-1, 28, 28, 1)),
    ])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

.. _input-example:

Model Input Example
^^^^^^^^^^^^^^^^^^^
Similar to model signatures, model inputs can be column-based (i.e DataFrames) or tensor-based
(i.e numpy.ndarrays). A model input example provides an instance of a valid model input.
Input examples are stored with the model as separate artifacts and are referenced in the the
:ref:`MLmodel file <pyfunc-model-config>`.

To include an input example with your model, add it to the appropriate log_model call, e.g.
:py:func:`sklearn.log_model() <mlflow.sklearn.log_model>`.

How To Log Model With Column-based Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For models accepting column-based inputs, an example can be a single record or a batch of records. The
sample input can be passed in as a Pandas DataFrame, list or dictionary. The given
example will be converted to a Pandas DataFrame and then serialized to json using the Pandas split-oriented
format. Bytes are base64-encoded. The following example demonstrates how you can log a column-based
input example with your model:

.. code-block:: python

    input_example = {
      "sepal length (cm)": 5.1,
      "sepal width (cm)": 3.5,
      "petal length (cm)": 1.4,
      "petal width (cm)": 0.2
    }
    mlflow.sklearn.log_model(..., input_example=input_example)

How To Log Model With Tensor-based Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For models accepting tensor-based inputs, an example must be a batch of inputs. By default, the axis 0
is the batch axis unless specified otherwise in the model signature. The sample input can be passed in as
a numpy ndarray or a dictionary mapping a string to a numpy array. The following example demonstrates how
you can log a tensor-based input example with your model:

.. code-block:: python

    # each input has shape (4, 4)
    input_example = np.array([
       [[  0,   0,   0,   0],
	[  0, 134,  25,  56],
	[253, 242, 195,   6],
	[  0,  93,  82,  82]],
       [[  0,  23,  46,   0],
	[ 33,  13,  36, 166],
	[ 76,  75,   0, 255],
	[ 33,  44,  11,  82]]
    ], dtype=np.uint8)
    mlflow.keras.log_model(..., input_example=input_example)

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

.. _pyfunc-model-flavor:

Python Function (``python_function``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``python_function`` model flavor serves as a default model interface for MLflow Python models.
Any MLflow Python model is expected to be loadable as a ``python_function`` model. This enables
other MLflow tools to work with any python model regardless of which persistence module or
framework was used to produce the model. This interoperability is very powerful because it allows
any Python model to be productionized in a variety of environments.

In addition, the ``python_function`` model flavor defines a generic filesystem :ref:`model format
<pyfunc-filesystem-format>` for Python models and provides utilities for saving and loading models
to and from this format. The format is self-contained in the sense that it includes all the
information necessary to load and use a model. Dependencies are stored either directly with the
model or referenced via conda environment. This model format allows other tools to integrate
their models with MLflow.

How To Save Model As Python Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Most ``python_function`` models are saved as part of other model flavors - for example, all mlflow
built-in flavors include the ``python_function`` flavor in the exported models. In addition, the
:py:mod:`mlflow.pyfunc` module defines functions for creating ``python_function`` models explicitly.
This module also includes utilities for creating custom Python models, which is a convenient way of
adding custom python code to ML models. For more information, see the :ref:`custom Python models
documentation <custom-python-models>`.


How To Load And Score Python Function Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can load ``python_function`` models in Python by calling the :py:func:`mlflow.pyfunc.load_model()`
function. Note that the ``load_model`` function assumes that all dependencies are already available
and *will not* check nor install any dependencies (
see :ref:`model deployment section <built-in-deployment>` for tools to deploy models with
automatic dependency management).

Once loaded, you can score the model by calling the :py:func:`predict <mlflow.pyfunc.PyFuncModel.predict>`
method, which has the following signature::

  predict(model_input: [pandas.DataFrame, numpy.ndarray, Dict[str, np.ndarray]]) -> [numpy.ndarray | pandas.(Series | DataFrame)]
  
All PyFunc models will support `pandas.DataFrame` as an input. In addition to `pandas.DataFrame`,
DL PyFunc models will also support tensor inputs in the form of `numpy.ndarrays`. To verify
whether a model flavor supports tensor inputs, please check the flavor's documentation.
  
For models with a column-based schema, inputs are typically provided in the form of a `pandas.DataFrame`.
If a dictionary mapping column name to values is provided as input for schemas with named columns or if a
python `List` or a `numpy.ndarray` is provided as input for schemas with unnamed columns, MLflow will cast the
input to a DataFrame. Schema enforcement and casting with respect to the expected data types is performed against
the DataFrame.

For models with a tensor-based schema, inputs are typically provided in the form of a `numpy.ndarray` or a
dictionary mapping the tensor name to its np.ndarray value. Schema enforcement will check the provided input's
shape and type against the shape and type specified in the model's schema and throw an error if they do not match.

For models where no schema is defined, no changes to the model inputs and outputs are made. MLflow will
propogate any errors raised by the model if the model does not accept the provided input type.


The python environment that a PyFunc model is loaded into for prediction or inference may differ from the environment
in which it was trained. In the case of an environment mismatch, a warning message will be printed when calling
:py:func:`mlflow.pyfunc.load_model`. This warning statement will identify the packages that have a version mismatch
between those used during training and the current environment.  In order to get the full dependencies of the
environment in which the model was trained, you can call :py:func:`mlflow.pyfunc.get_model_dependencies`.
Furthermore, if you want to run model inference in the same environment used in model training, you can call
:py:func:`mlflow.pyfunc.spark_udf` with the `env_manager` argument set as "conda". This will generate the environment
from the `conda.yaml` file, ensuring that the python UDF will execute with the exact package versions that were used
during training.


R Function (``crate``)
^^^^^^^^^^^^^^^^^^^^^^

The ``crate`` model flavor defines a generic model format for representing an arbitrary R prediction
function as an MLflow model using the ``crate`` function from the
`carrier <https://github.com/r-lib/carrier>`_ package. The prediction function is expected to take a dataframe as input and
produce a dataframe, a vector or a list with the predictions as output.

This flavor requires R to be installed in order to be used.

H\ :sub:`2`\ O (``h2o``)
^^^^^^^^^^^^^^^^^^^^^^^^

The ``h2o`` model flavor enables logging and loading H2O models.

The :py:mod:`mlflow.h2o` module defines :py:func:`save_model() <mlflow.h2o.save_model>` and
:py:func:`log_model() <mlflow.h2o.log_model>` methods in python, and
`mlflow_save_model <R-api.html#mlflow-save-model-h2o>`__ and
`mlflow_log_model <R-api.html#mlflow-log-model>`__ in R for saving H2O models in MLflow Model
format.
These methods produce MLflow Models with the ``python_function`` flavor, allowing you to load them
as generic Python functions for inference via :py:func:`mlflow.pyfunc.load_model()`.
This loaded PyFunc model can be scored with only DataFrame input. When you load
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
as generic Python functions for inference via :py:func:`mlflow.pyfunc.load_model()`. This loaded PyFunc model can be
scored with both DataFrame input and numpy array input. Finally, you can use the :py:func:`mlflow.keras.load_model()`
function in Python or `mlflow_load_model <R-api.rst#mlflow-load-model>`__ function in R to load MLflow Models
with the ``keras`` flavor as `Keras Model objects <https://keras.io/models/about-keras-models/>`_.

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
Similarly, ``mleap`` models can be saved in R with `mlflow_save_model <R-api.rst#mlflow-save-model>`__
and loaded with `mlflow_load_model <R-api.rst#mlflow-load-model>`__, with
`mlflow_save_model <R-api.rst#mlflow-save-model>`__ requiring `sample_input` to be specified as a
sample Spark dataframe containing input data to the model is required by MLeap for data schema
inference.

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
method to load MLflow Models with the ``pytorch`` flavor as PyTorch model objects. This loaded
PyFunc model can be scored with both DataFrame input and numpy array input. Finally, models
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
This loaded PyFunc model can only be scored with DataFrame input. Finally, you can use the
:py:func:`mlflow.sklearn.load_model()` method to load MLflow Models with the ``sklearn`` flavor as
scikit-learn model objects.

For more information, see :py:mod:`mlflow.sklearn`.

Spark MLlib (``spark``)
^^^^^^^^^^^^^^^^^^^^^^^

The ``spark`` model flavor enables exporting Spark MLlib models as MLflow Models.

The :py:mod:`mlflow.spark` module defines :py:func:`save_model() <mlflow.spark.save_model>` and
:py:func:`log_model() <mlflow.spark.log_model>` methods that save Spark MLlib pipelines in MLflow
model format. MLflow Models produced by these functions contain the ``python_function`` flavor,
allowing you to load them as generic Python functions via :py:func:`mlflow.pyfunc.load_model()`.
This loaded PyFunc model can only be scored with DataFrame input.
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
Python functions for inference via :py:func:`mlflow.pyfunc.load_model()`. This loaded PyFunc model
can be scored with both DataFrame input and numpy array input. Finally, you can use the
:py:func:`mlflow.tensorflow.load_model()` method to load MLflow Models with the ``tensorflow``
flavor as TensorFlow graphs.

For more information, see :py:mod:`mlflow.tensorflow`.

ONNX (``onnx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``onnx`` model flavor enables logging of `ONNX models <http://onnx.ai/>`_ in MLflow format via
the :py:func:`mlflow.onnx.save_model()` and :py:func:`mlflow.onnx.log_model()` methods. These
methods also add the ``python_function`` flavor to the MLflow Models that they produce, allowing the
models to be interpreted as generic Python functions for inference via
:py:func:`mlflow.pyfunc.load_model()`. This loaded PyFunc model can be scored with
both DataFrame input and numpy array input. The ``python_function`` representation of an MLflow
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
:py:func:`mlflow.pyfunc.load_model()`. This loaded PyFunc model can be scored with
both DataFrame input and numpy array input. You can also use the :py:func:`mlflow.gluon.load_model()`
method to load MLflow Models with the ``gluon`` flavor in native Gluon format.

For more information, see :py:mod:`mlflow.gluon`.

XGBoost (``xgboost``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``xgboost`` model flavor enables logging of `XGBoost models
<https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster>`_
in MLflow format via the :py:func:`mlflow.xgboost.save_model()` and :py:func:`mlflow.xgboost.log_model()` methods in python and `mlflow_save_model <R-api.html#mlflow-save-model-crate>`__ and `mlflow_log_model <R-api.html#mlflow-log-model>`__ in R respectively.
These methods also add the ``python_function`` flavor to the MLflow Models that they produce, allowing the
models to be interpreted as generic Python functions for inference via
:py:func:`mlflow.pyfunc.load_model()`. This loaded PyFunc model can only be scored with DataFrame input.
You can also use the :py:func:`mlflow.xgboost.load_model()`
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
:py:func:`mlflow.pyfunc.load_model()`. This loaded PyFunc model can only be scored with DataFrame input.
You can also use the :py:func:`mlflow.lightgbm.load_model()`
method to load MLflow Models with the ``lightgbm`` model flavor in native LightGBM format.

Note that the ``lightgbm`` model flavor only supports an instance of `lightgbm.Booster
<https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm-booster>`__,
not models that implement the `scikit-learn API
<https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api>`_.

For more information, see :py:mod:`mlflow.lightgbm`.

CatBoost (``catboost``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``catboost`` model flavor enables logging of `CatBoost models
<https://catboost.ai/docs/concepts/python-reference_catboost.html>`_
in MLflow format via the :py:func:`mlflow.catboost.save_model()` and :py:func:`mlflow.catboost.log_model()` methods.
These methods also add the ``python_function`` flavor to the MLflow Models that they produce, allowing the
models to be interpreted as generic Python functions for inference via
:py:func:`mlflow.pyfunc.load_model()`. You can also use the :py:func:`mlflow.catboost.load_model()`
method to load MLflow Models with the ``catboost`` model flavor in native CatBoost format.

For more information, see :py:mod:`mlflow.catboost`.

Spacy(``spaCy``)
^^^^^^^^^^^^^^^^^^^^
The ``spaCy`` model flavor enables logging of `spaCy models <https://spacy.io/models>`_ in MLflow format via
the :py:func:`mlflow.spacy.save_model()` and :py:func:`mlflow.spacy.log_model()` methods. Additionally, these
methods add the ``python_function`` flavor to the MLflow Models that they produce, allowing the models to be
interpreted as generic Python functions for inference via :py:func:`mlflow.pyfunc.load_model()`.
This loaded PyFunc model can only be scored with DataFrame input. You can
also use the :py:func:`mlflow.spacy.load_model()` method to load MLflow Models with the ``spacy`` model flavor
in native spaCy format.

For more information, see :py:mod:`mlflow.spacy`.

Fastai(``fastai``)
^^^^^^^^^^^^^^^^^^^^^^
The ``fastai`` model flavor enables logging of `fastai Learner models <https://docs.fast.ai/training.html>`_ in MLflow format via
the :py:func:`mlflow.fastai.save_model()` and :py:func:`mlflow.fastai.log_model()` methods. Additionally, these
methods add the ``python_function`` flavor to the MLflow Models that they produce, allowing the models to be
interpreted as generic Python functions for inference via :py:func:`mlflow.pyfunc.load_model()`. This loaded PyFunc model can
only be scored with DataFrame input. You can also use the :py:func:`mlflow.fastai.load_model()` method to
load MLflow Models with the ``fastai`` model flavor in native fastai format.

For more information, see :py:mod:`mlflow.fastai`.

Statsmodels (``statsmodels``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``statsmodels`` model flavor enables logging of `Statsmodels models
<https://www.statsmodels.org/stable/api.html>`_ in MLflow format via the :py:func:`mlflow.statsmodels.save_model()`
and :py:func:`mlflow.statsmodels.log_model()` methods.
These methods also add the ``python_function`` flavor to the MLflow Models that they produce, allowing the
models to be interpreted as generic Python functions for inference via
:py:func:`mlflow.pyfunc.load_model()`. This loaded PyFunc model can only be scored with DataFrame input.
You can also use the :py:func:`mlflow.statsmodels.load_model()`
method to load MLflow Models with the ``statsmodels`` model flavor in native statsmodels format.

As for now, automatic logging is restricted to parameters, metrics and models generated by a call to `fit`
on a ``statsmodels`` model.

For more information, see :py:mod:`mlflow.statsmodels`.

Prophet (``prophet``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``prophet`` model flavor enables logging of `Prophet models
<https://facebook.github.io/prophet/>`_ in MLflow format via the :py:func:`mlflow.prophet.save_model()`
and :py:func:`mlflow.prophet.log_model()` methods.
These methods also add the ``python_function`` flavor to the MLflow Models that they produce, allowing the
models to be interpreted as generic Python functions for inference via
:py:func:`mlflow.pyfunc.load_model()`. This loaded PyFunc model can only be scored with DataFrame input.
You can also use the :py:func:`mlflow.prophet.load_model()`
method to load MLflow Models with the ``prophet`` model flavor in native prophet format.

For more information, see :py:mod:`mlflow.prophet`.

Pmdarima (``pmdarima``) (Experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``pmdarima`` model flavor enables logging of `pmdarima models <http://alkaline-ml.com/pmdarima/>`_ in MLflow
format via the :py:func:`mlflow.pmdarima.save_model()` and :py:func:`mlflow.pmdarima.log_model()` methods.
These methods also add the ``python_function`` flavor to the MLflow Models that they produce, allowing the
model to be interpreted as generic Python functions for inference via :py:func:`mlflow.pyfunc.load_model()`.
This loaded PyFunc model can only be scored with a DataFrame input.
You can also use the :py:func:`mlflow.pmdarima.load_model()` method to load MLflow Models with the ``pmdarima``
model flavor in native pmdarima formats.

The interface for utilizing a ``pmdarima`` model loaded as a ``pyfunc`` type for generating forecast predictions uses
a *single-row* ``Pandas DataFrame`` configuration argument. The following columns in this configuration
``Pandas DataFrame`` are supported:

* ``n_periods`` (required) - specifies the number of future periods to generate starting from the last datetime value
    of the training dataset, utilizing the frequency of the input training series when the model was trained.
    (for example, if the training data series elements represent one value per hour, in order to forecast 3 days of
    future data, set the column ``n_periods`` to ``72``.
* ``X`` (optional) - exogenous regressor values (*only supported in pmdarima version >= 1.8.0*) a 2D array of values for
    future time period events. For more information, read the underlying library
    `explanation <https://www.statsmodels.org/stable/endog_exog.html>`_.
* ``return_conf_int`` (optional) - a boolean (Default: ``False``) for whether to return confidence interval values.
    See above note.
* ``alpha`` (optional) - the significance value for calculating confidence intervals. (Default: ``0.05``)

An example configuration for the ``pyfunc`` predict of a ``pmdarima`` model is shown below, with a future period
prediction count of 100, a confidence interval calculation generation, no exogenous regressor elements, and a default
alpha of ``0.05``:

====== ========= ===============
Index  n_periods return_conf_int
====== ========= ===============
0      100       True
====== ========= ===============

.. warning::
    The ``Pandas DataFrame`` passed to a ``pmdarima`` ``pyfunc`` flavor must only contain 1 row.

.. note::
    When predicting a ``pmdarima`` flavor, the ``predict`` method's ``DataFrame`` configuration column
    ``return_conf_int``'s value controls the output format. When the column's value is set to ``False`` or ``None``
    (which is the default if this column is not supplied in the configuration ``DataFrame``), the schema of the
    returned ``Pandas DataFrame`` is a single column: ``["yhat"]``. When set to ``True``, the schema of the returned
    ``DataFrame`` is: ``["yhat", "yhat_lower", "yhat_upper"]`` with the respective lower (``yhat_lower``) and
    upper (``yhat_upper``) confidence intervals added to the forecast predictions (``yhat``).

Example usage of pmdarima artifact loaded as a pyfunc with confidence intervals calculated:

.. code-block:: py

    import pmdarima
    import mlflow
    import pandas as pd

    data = pmdarima.datasets.load_airpassengers()

    with mlflow.start_run():

        model = pmdarima.auto_arima(data, seasonal=True)
        mlflow.pmdarima.save_model(model, "/tmp/model.pmd")

    loaded_pyfunc = mlflow.pyfunc.load_model("/tmp/model.pmd")

    prediction_conf = pd.DataFrame([{"n_periods": 4, "return_conf_int": True, "alpha": 0.1}])

    predictions = loaded_pyfunc.predict(prediction_conf)

Output (``Pandas DataFrame``):

====== ========== ========== ==========
Index  yhat       yhat_lower yhat_upper
====== ========== ========== ==========
0      467.573731 423.30995  511.83751
1      490.494467 416.17449  564.81444
2      509.138684 420.56255  597.71117
3      492.554714 397.30634  587.80309
====== ========== ========== ==========

.. warning::
    Signature logging for ``pmdarima`` will not function correctly if ``return_conf_int`` is set to ``True`` from
    a non-pyfunc artifact. The output of the native ``ARIMA.predict()`` when returning confidence intervals is not
    a recognized signature type.

Diviner (``diviner``) (Experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``diviner`` model flavor enables logging of
`diviner models <https://databricks-diviner.readthedocs.io/en/latest/index.html>`_ in MLflow format via the
:py:func:`mlflow.diviner.save_model()` and :py:func:`mlflow.diviner.log_model()` methods. These methods also add the
``python_function`` flavor to the MLflow Models that they produce, allowing the model to be interpreted as generic
Python functions for inference via :py:func:`mlflow.pyfunc.load_model()`.
This loaded PyFunc model can only be scored with a DataFrame input.
You can also use the :py:func:`mlflow.diviner.load_model()` method to load MLflow Models with the ``diviner``
model flavor in native diviner formats.

Diviner Types
~~~~~~~~~~~~~
Diviner is a library that provides an orchestration framework for performing time series forecasting on groups of
related series. Forecasting in ``diviner`` is accomplished through wrapping popular open source libraries such as
`prophet <https://facebook.github.io/prophet/>`_ and `pmdarima <http://alkaline-ml.com/pmdarima/>`_. The ``diviner``
library offers a simplified set of APIs to simultaneously generate distinct time series forecasts for multiple data
groupings using a single input DataFrame and a unified high-level API.

Metrics and Parameters logging for Diviner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unlike other flavors that are supported in MLflow, Diviner has the concept of grouped models. As a collection of many
(perhaps thousands) of individual forecasting models, the burden to the tracking server to log individual metrics
and parameters for each of these models is significant. For this reason, metrics and parameters are exposed for
retrieval from Diviner's APIs as ``Pandas`` ``DataFrames``, rather than discrete primitive values.

To illustrate, let us assume we are forecasting hourly electricity consumption from major cities around the world.
A sample of our input data looks like this:

======= ========== =================== =======
country city       datetime            watts
======= ========== =================== =======
US      NewYork    2022-03-01 00:01:00 23568.9
US      NewYork    2022-03-01 00:02:00 22331.7
US      Boston     2022-03-01 00:01:00 14220.1
US      Boston     2022-03-01 00:02:00 14183.4
CA      Toronto    2022-03-01 00:01:00 18562.2
CA      Toronto    2022-03-01 00:02:00 17681.6
MX      MexicoCity 2022-03-01 00:01:00 19946.8
MX      MexicoCity 2022-03-01 00:02:00 19444.0
======= ========== =================== =======

If we were to ``fit`` a model on this data, supplying the grouping keys as:

.. code-block:: py

    grouping_keys = ["country", "city"]

We will have a model generated for each of the grouping keys that have been supplied:

.. code-block:: py

    [("US", "NewYork"),
     ("US", "Boston"),
     ("CA", "Toronto"),
     ("MX", "MexicoCity")]

With a model constructed for each of these, entering each of their metrics and parameters wouldn't be an issue for the
MLflow tracking server. What would become a problem, however, is if we modeled each major city on the planet and ran
this forecasting scenario every day. If we were to adhere to the conditions of the World Bank, that would mean just
over 10,000 models as of 2022. After a mere few weeks of running this forecasting every day we would have a very large
metrics table.

To eliminate this issue for large-scale forecasting, the metrics and parameters for ``diviner`` are extracted as a
grouping key indexed ``Pandas DataFrame``, as shown below for example (float values truncated for visibility):

===================== ======= ========== ========== ====== ====== ==== ===== =====
grouping_key_columns  country city       mse        rmse   mae    mape mdape smape
===================== ======= ========== ========== ====== ====== ==== ===== =====
"('country', 'city')" CA      Toronto    8276851.6  2801.7 2417.7 0.16 0.16  0.159
"('country', 'city')" MX      MexicoCity 3548872.4  1833.8 1584.5 0.15 0.16  0.159
"('country', 'city')" US      NewYork    3167846.4  1732.4 1498.2 0.15 0.16  0.158
"('country', 'city')" US      Boston     14082666.4 3653.2 3156.2 0.15 0.16  0.159
===================== ======= ========== ========== ====== ====== ==== ===== =====

There are two recommended means of logging the metrics and parameters from a ``diviner`` model :


* Writing the DataFrames to local storage and using :py:func:`mlflow.log_artifacts`


.. code-block:: py

    import os
    import mlflow
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        params = model.extract_model_params()
        metrics = model.cross_validate_and_score(
            horizon="72 hours",
            period="240 hours",
            initial="480 hours",
            parallel="threads",
            rolling_window=0.1,
            monthly=False,
        )
        params.to_csv(f"{tmpdir}/params.csv", index=False, header=True)
        metrics.to_csv(f"{tmpdir}/metrics.csv", index=False, header=True)

        mlflow.log_artifacts(tmpdir, artifact_path="data")


* Writing directly as a JSON artifact using :py:func:`mlflow.log_dict`


.. note::
    The parameters extract from ``diviner`` models *may require* casting (or dropping of columns) if using the
    ``pd.DataFrame.to_dict()`` approach due to the inability of this method to serialize objects.

.. code-block:: py

    import mlflow

    params = model.extract_model_params()
    metrics = model.cross_validate_and_score(
        horizon="72 hours",
        period="240 hours",
        initial="480 hours",
        parallel="threads",
        rolling_window=0.1,
        monthly=False,
    )
    params["t_scale"] = params["t_scale"].astype(str)
    params["start"] = params["start"].astype(str)
    params = params.drop("stan_backend", axis=1)

    mlflow.log_dict(params.to_dict(), "params.json")
    mlflow.log_dict(metrics.to_dict(), "metrics.json")

Logging of the model artifact is shown in the ``pyfunc`` example below.

Diviner pyfunc usage
~~~~~~~~~~~~~~~~~~~~
The MLflow Diviner flavor includes an implementation of the ``pyfunc`` interface for Diviner models. To control
prediction behavior, you can specify configuration arguments in the first row of a Pandas DataFrame input.

As this configuration is dependent upon the underlying model type (i.e., the ``diviner.GroupedProphet.forecast()``
method has a different signature than does ``diviner.GroupedPmdarima.predict()``), the Diviner pyfunc implementation
attempts to coerce arguments to the types expected by the underlying model.

.. note::
    Diviner models support both "full group" and "partial group" forecasting. If a column named ``"groups"`` is present
    in the configuration ``DataFrame`` submitted to the ``pyfunc`` flavor, the grouping key values in the first row
    will be used to generate a subset of forecast predictions. This functionality removes the need to filter a subset
    from the full output of all groups forecasts if the results of only a few (or one) groups are needed.

For a ``GroupedPmdarima`` model, an example configuration for the ``pyfunc`` ``predict()`` method is:

.. code-block:: py

    import mlflow
    import pandas as pd
    from pmdarima.arima.auto import AutoARIMA
    from diviner import GroupedPmdarima

    with mlflow.start_run():
        base_model = AutoARIMA(out_of_sample_size=96, maxiter=200)
        model = GroupedPmdarima(model_template=base_model).fit(
            df=df,
            group_key_columns=["country", "city"],
            y_col="watts",
            datetime_col="datetime",
            silence_warnings=True,
        )

        mlflow.diviner.save_model(diviner_model=model, path="/tmp/diviner_model")

    diviner_pyfunc = mlflow.pyfunc.load_model(model_uri="/tmp/diviner_model")

    predict_conf = pd.DataFrame(
        {"n_periods": 120,
         "groups": [("US", "NewYork"), ("CA", "Toronto"), ("MX", "MexicoCity")],  # NB: List of tuples required.
         "predict_col": "wattage_forecast",
         "alpha": 0.1,
         "return_conf_int": True,
         "on_error": "warn",
        },
        index=[0],
    )

    subset_forecasts = diviner_pyfunc.predict(predict_conf)

.. note::
    There are several instances in which a configuration ``DataFrame`` submitted to the ``pyfunc`` ``predict()`` method
    will cause an ``MlflowException`` to be raised:

        * If neither ``horizon`` or ``n_periods`` are provided.
        * The value of ``n_periods`` or ``horizon`` is not an integer.
        * If the model is of type ``GroupedProphet``, ``frequency`` as a string type must be provided.
        * If both ``horizon`` and ``n_periods`` are provided with different values.

.. _model-evaluation:

Model Evaluation
----------------
After building and training your MLflow Model, you can use the :py:func:`mlflow.evaluate()` API to
evaluate its performance on one or more datasets of your choosing. :py:func:`mlflow.evaluate()`
currently supports evaluation of MLflow Models with the
:ref:`python_function (pyfunc) model flavor <pyfunc-model-flavor>` for classification and regression
tasks, computing a variety of task-specific performance metrics, model performance plots, and
model explanations. Evaluation results are logged to :ref:`MLflow Tracking <tracking>`.

The following `example from the MLflow GitHub Repository
<https://github.com/mlflow/mlflow/blob/master/examples/evaluation/evaluate_on_binary_classifier.py>`_
uses :py:func:`mlflow.evaluate()` to evaluate the performance of a classifier
on the `UCI Adult Data Set <https://archive.ics.uci.edu/ml/datasets/adult>`_, logging a
comprehensive collection of MLflow Metrics and Artifacts that provide insight into model performance
and behavior:

.. code-block:: py

    import xgboost
    import shap
    import mlflow
    from sklearn.model_selection import train_test_split

    # load UCI Adult Data Set; segment it into training and test sets
    X, y = shap.datasets.adult()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # train XGBoost model
    model = xgboost.XGBClassifier().fit(X_train, y_train)

    # construct an evaluation dataset from the test set
    eval_data = X_test
    eval_data["label"] = y_test

    with mlflow.start_run() as run:
        model_info = mlflow.sklearn.log_model(model, "model")
        result = mlflow.evaluate(
            model_info.model_uri,
            eval_data,
            targets="label",
            model_type="classifier",
            dataset_name="adult",
            evaluators=["default"],
        )

|eval_metrics_img| |eval_importance_img|

.. |eval_metrics_img| image:: _static/images/model_evaluation_metrics.png
   :width: 30%

.. |eval_importance_img| image:: _static/images/model_evaluation_feature_importance.png
   :width: 69%


Evaluating with Custom Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the default set of metrics is insufficient, you can specify a list of ``custom_metrics`` functions to 
:py:func:`mlflow.evaluate()` to produce custom performance metrics for the model(s) that you're evaluating. Custom metric
functions should accept at least two arguments: a DataFrame containing ``prediction`` and ``target`` columns,
and a dictionary containing the default set of metrics. For a full list of default metrics, refer to the documentation 
of :py:func:`mlflow.evaluate()`. If the custom metric function produces artifacts in the form of files, it should also
accept an additional string argument representing the path to the temporary directory that can be used to store such
artifacts.

The following `short example from the MLflow GitHub Repository
<https://github.com/mlflow/mlflow/blob/master/examples/evaluation/evaluate_with_custom_metrics.py>`_ 
uses :py:func:`mlflow.evaluate()` with a custom metric function to evaluate the performance of a regressor on the
`California Housing Dataset <https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html>`_.
Note that custom metric functions can return both metrics and artifacts. They can either return a single
dictionary of metrics, or two dictionaries representing metrics and artifacts.

.. code-block:: py

    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    import numpy as np
    import mlflow
    import os
    import matplotlib.pyplot as plt

    # loading the California housing dataset
    cali_housing = fetch_california_housing(as_frame=True)

    # split the dataset into train and test partitions
    X_train, X_test, y_train, y_test = train_test_split(
        cali_housing.data, cali_housing.target, test_size=0.2, random_state=123
    )

    # train the model
    lin_reg = LinearRegression().fit(X_train, y_train)

    # creating the evaluation dataframe
    eval_data = X_test.copy()
    eval_data["target"] = y_test


    def example_custom_metric_fn(eval_df, builtin_metrics, artifacts_dir):
        """
        This example custom metric function creates a metric based on the ``prediction`` and
        ``target`` columns in ``eval_df`` and a metric derived from existing metrics in
        ``builtin_metrics``. It also generates and saves a scatter plot to ``artifacts_dir`` that
        visualizes the relationship between the predictions and targets for the given model to a
        file as an image artifact.
        """
        metrics = {
            "squared_diff_plus_one": np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2),
            "sum_on_label_divided_by_two": builtin_metrics["sum_on_label"] / 2,
        }
        plt.scatter(eval_df["prediction"], eval_df["target"])
        plt.xlabel("Targets")
        plt.ylabel("Predictions")
        plt.title("Targets vs. Predictions")
        plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
        plt.savefig(plot_path)
        artifacts = {"example_scatter_plot_artifact": plot_path}
        return metrics, artifacts


    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(lin_reg, "model")
        model_uri = mlflow.get_artifact_uri("model")
        result = mlflow.evaluate(
            model=model_uri,
            data=eval_data,
            targets="target",
            model_type="regressor",
            dataset_name="cali_housing",
            evaluators=["default"],
            custom_metrics=[example_custom_metric_fn],
        )


For a more comprehensive custom metrics usage example, refer to `this example from the MLflow GitHub Repository
<https://github.com/mlflow/mlflow/blob/master/examples/evaluation/evaluate_with_custom_metrics_comprehensive.py>`_.

Additional information about model evaluation behaviors and outputs is available in the
:py:func:`mlflow.evaluate()` API docs.

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
    from sys import version_info
    import xgboost as xgb
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                      minor=version_info.minor,
                                                      micro=version_info.micro)
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

    # Create a Conda environment for the new MLflow Model that contains all necessary dependencies.
    import cloudpickle
    conda_env = {
        'channels': ['defaults'],
        'dependencies': [
          'python={}'.format(PYTHON_VERSION),
          'pip',
          {
            'pip': [
              'mlflow',
              'xgboost=={}'.format(xgb.__version__),
              'cloudpickle=={}'.format(cloudpickle.__version__),
            ],
          },
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

.. _built-in-deployment:

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

* Tensor input formatted as described in `TF Serving's API docs
  <https://www.tensorflow.org/tfx/serving/api_rest#request_format_2>`_ where the provided inputs
  will be cast to Numpy arrays. This format is specified using a ``Content-Type`` request header
  value of ``application/json`` and the ``instances`` or ``inputs`` key in the request body dictionary.

If the ``Content-Type`` request header has a value of ``application/json``, MLflow will infer whether
the input format is a pandas DataFrame or TF serving (i.e tensor) input based on the data in the request
body. For pandas DataFrame input, the orient can  also be provided explicitly by specifying the format
in the request header as shown in the record-oriented example below.

.. note:: Since JSON loses type information, MLflow will cast the JSON input to the input type specified
    in the model's schema if available. If your model is sensitive to input types, it is recommended that
    a schema is provided for the model to ensure that type mismatch errors do not occur at inference time.
    In particular, DL models are typically strict about input types and will need model schema in order
    for the model to score correctly. For complex data types, see :ref:`encoding-complex-data` below.

Example requests:

.. code-block:: bash

    # split-oriented DataFrame input
    curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
        "columns": ["a", "b", "c"],
        "data": [[1, 2, 3], [4, 5, 6]]
    }'

    # record-oriented DataFrame input (fine for vector rows, loses ordering for JSON records)
    curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json; format=pandas-records' -d '[
        {"a": 1,"b": 2,"c": 3},
        {"a": 4,"b": 5,"c": 6}
    ]'

    # numpy/tensor input using TF serving's "instances" format
    curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
        "instances": [
            {"a": "s1", "b": 1, "c": [1, 2, 3]},
            {"a": "s2", "b": 2, "c": [4, 5, 6]},
            {"a": "s3", "b": 3, "c": [7, 8, 9]}
        ]
    }'

    # numpy/tensor input using TF serving's "inputs" format
    curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
        "inputs": {"a": ["s1", "s2", "s3"], "b": [1, 2, 3], "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
    }'


For more information about serializing pandas DataFrames, see
`pandas.DataFrame.to_json <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_json.html>`_.

For more information about serializing tensor inputs using the TF serving format, see
`TF serving's request format docs <https://www.tensorflow.org/tfx/serving/api_rest#request_format_2>`_.

.. _serving_with_mlserver:

Serving with MLServer (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python models can be deployed using `Seldon's MLServer
<https://mlserver.readthedocs.io/en/latest/>`_ as alternative inference server. 
MLServer is integrated with two leading open source model deployment tools,
`Seldon Core
<https://docs.seldon.io/projects/seldon-core/en/latest/graph/protocols.html#v2-kfserving-protocol>`_
and `KServe (formerly known as KFServing)
<https://kserve.github.io/website/modelserving/v1beta1/sklearn/v2/>`_, and can
be used to test and deploy models using these frameworks. 
This is especially powerful when building docker images since the docker image
built with MLServer can be deployed directly with both of these frameworks. 

MLServer exposes the same scoring API through the ``/invocations`` endpoint.
In addition, it supports the standard `V2 Inference Protocol
<https://github.com/kubeflow/kfserving/tree/master/docs/predict-api/v2>`_.

.. note::
   To use MLServer with MLflow, please install ``mlflow`` as:

   .. code-block:: bash

       pip install mlflow[extras]

To serve a MLflow model using MLServer, you can use the ``--enable-mlserver`` flag,
such as:

.. code-block:: bash

    mlflow models serve -m my_model --enable-mlserver

Similarly, to build a Docker image built with MLServer you can use the
``--enable-mlserver`` flag, such as:

.. code-block:: bash

    mlflow models build -m my_model --enable-mlserver -n my-model

To read more about the integration between MLflow and MLServer, please check
the `end-to-end example in the MLServer documentation
<https://mlserver.readthedocs.io/en/latest/examples/mlflow/README.html>`_ or
visit the `MLServer docs <https://mlserver.readthedocs.io/en/latest/>`_.

.. note::
    - This feature is experimental and is subject to change.
    - MLServer requires Python 3.7 or above.

.. _encoding-complex-data:

Encoding complex data
~~~~~~~~~~~~~~~~~~~~~

Complex data types, such as dates or binary, do not have a native JSON representation. If you include a model
signature, MLflow can automatically decode supported data types from JSON. The following data type conversions
are supported:

* binary: data is expected to be base64 encoded, MLflow will automatically base64 decode.

* datetime: data is expected as string according to
  `ISO 8601 specification <https://www.iso.org/iso-8601-date-and-time-format.html>`_.
  MLflow will parse this into the appropriate datetime representation on the given platform.

Example requests:

.. code-block:: bash

    # record-oriented DataFrame input with binary column "b"
    curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json; format=pandas-records' -d '[
        {"a": 0, "b": "dGVzdCBiaW5hcnkgZGF0YSAw"},
        {"a": 1, "b": "dGVzdCBiaW5hcnkgZGF0YSAx"},
        {"a": 2, "b": "dGVzdCBiaW5hcnkgZGF0YSAy"}
    ]'

    # record-oriented DataFrame input with datetime column "b"
    curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json; format=pandas-records' -d '[
        {"a": 0, "b": "2020-01-01T00:00:00Z"},
        {"a": 1, "b": "2020-02-01T12:34:56Z"},
        {"a": 2, "b": "2021-03-01T00:00:00Z"}
    ]'


Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

MLflow also has a CLI that supports the following commands:

* `serve <cli.html#mlflow-models-serve>`_ deploys the model as a local REST API server.
* `build_docker <cli.html#mlflow-models-build-docker>`_ packages a REST API endpoint serving the
  model as a docker image.
* `predict <cli.html#mlflow-models-predict>`_ uses the model to generate a prediction for a local
  CSV or JSON file. Note that this method only supports DataFrame input.

For more info, see:

.. code-block:: bash

    mlflow models --help
    mlflow models serve --help
    mlflow models predict --help
    mlflow models build-docker --help

.. _azureml_deployment:

Deploy a ``python_function`` model on Microsoft Azure ML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MLflow plugin `azureml-mlflow <https://pypi.org/project/azureml-mlflow/>`_ can deploy models to Azure ML, either to Azure Kubernetes Service (AKS) or Azure Container Instances (ACI) for real-time serving. 

The resulting deployment accepts the following data formats as input:

* JSON-serialized pandas DataFrames in the ``split`` orientation. For example, ``data = pandas_df.to_json(orient='split')``. This format is specified using a ``Content-Type`` request header value of ``application/json``.

.. warning::
    The ``TensorSpec`` input format is not fully supported for deployments on Azure Machine Learning at the moment. Be aware that many ``autolog()`` implementations may use ``TensorSpec`` for model's signatures when logging models and hence those deployments will fail in Azure ML.

Deployments can be generated using both the Python API or MLflow CLI. In both cases, a ``JSON`` configuration file can be indicated with the details of the deployment you want to achieve. If not indicated, then a default deployment is done using Azure Container Instances (ACI) and a minimal configuration. The full specification of this configuration file can be checked at `Deployment configuration schema <https://docs.microsoft.com/en-us/azure/machine-learning/reference-azure-machine-learning-cli#deployment-configuration-schema>`_. Also, you will also need the Azure ML MLflow Tracking URI of your particular Azure ML Workspace where you want to deploy your model. You can obtain this URI in several ways:

* Through the `Azure ML Studio <https://ml.azure.com>`_:

  * Navigate to `Azure ML Studio <https://ml.azure.com>`_ and select the workspace you are working on.
  * Click on the name of the workspace at the upper right corner of the page.
  * Click "View all properties in Azure Portal" on the pane popup.
  * Copy the ``MLflow tracking URI`` value from the properties section.

* Programmatically, using Azure ML SDK with the method `Woskspace.get_mlflow_tracking_uri() <https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.workspace.workspace?view=azure-ml-py#azureml-core-workspace-workspace-get-mlflow-tracking-uri>`_. If you are running inside Azure ML Compute, like for instance a Compute Instace, you can get this value also from the environment variable ``os.environ["MLFLOW_TRACKING_URI"]``.
* Manually, for a given Subscription ID, Resource Group and Azure ML Workspace, the URI is as follows: ``azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP_NAME>/providers/Microsoft.MachineLearningServices/workspaces/<WORKSPACE_NAME>``


.. rubric:: Configuration example for ACI deployment

.. code-block:: json

    {
      "computeType": "aci",
      "containerResourceRequirements": 
      {
        "cpu": 1,
        "memoryInGB": 1
      },
      "location": "eastus2",
    }

Remarks:
 * If ``containerResourceRequirements`` is not indicated, a deployment with minimal compute configuration is applied (``cpu: 0.1`` and ``memory: 0.5``).
 * If ``location`` is not indicated, it defaults to the location of the workspace.

.. rubric:: Configuration example for an AKS deployment

.. code-block:: json

    {
      "computeType": "aks",
      "computeTargetName": "aks-mlflow"
    }

Remarks:
  * In above exmaple, ``aks-mlflow`` is the name of an Azure Kubernetes Cluster registered/created in Azure Machine Learning.

The following examples show how to create a deployment in ACI. Please, ensure you have `azureml-mlflow <https://pypi.org/project/azureml-mlflow/>`_ installed before continuing.

.. rubric:: Example: Workflow using the Python API

.. code-block:: py

    import json
    from mlflow.deployments import get_deploy_client

    # Create the deployment configuration.
    # If no deployment configuration is provided, then the deployment happens on ACI.
    deploy_config = {
        "computeType": "aci"
    }

    # Write the deployment configuration into a file.
    deployment_config_path = "deployment_config.json"
    with open(deployment_config_path, "w") as outfile:
        outfile.write(json.dumps(deploy_config))

    # Set the tracking uri in the deployment client.
    client = get_deploy_client("<azureml-mlflow-tracking-url>")

    # MLflow requires the deployment configuration to be passed as a dictionary.
    config = {'deploy-config-file': deployment_config_path}
    model_name = "mymodel"
    model_version = 1

    # define the model path and the name is the service name
    # if model is not registered, it gets registered automatically and a name is autogenerated using the "name" parameter below 
    client.create_deployment(model_uri=f'models:/{model_name}/{model_version}',
                            config=config,
                            name="mymodel-aci-deployment")

    # After the model deployment completes, requests can be posted via HTTP to the new ACI
    # webservice's scoring URI. 
    print("Scoring URI is: %s", webservice.scoring_uri)

    # The following example posts a sample input from the wine dataset
    # used in the MLflow ElasticNet example:
    # https://github.com/mlflow/mlflow/tree/master/examples/sklearn_elasticnet_wine

    # `sample_input` is a JSON-serialized pandas DataFrame with the `split` orientation
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

.. rubric:: Example: Workflow using the MLflow CLI

.. code-block:: bash
    
    echo "{ computeType: aci }" > deployment_config.json
    mlflow deployments create --name <deployment-name> -m models:/<model-name>/<model-version> -t <azureml-mlflow-tracking-url> --deploy-config-file deployment_config.json

    # After the deployment completes, requests can be posted via HTTP to the new ACI
    # webservice's scoring URI.

    scoring_uri=$(az ml service show --name <deployment-name> -v | jq -r ".scoringUri")

    # The following example posts a sample input from the wine dataset
    # used in the MLflow ElasticNet example:
    # https://github.com/mlflow/mlflow/tree/master/examples/sklearn_elasticnet_wine

    # `sample_input` is a JSON-serialized pandas DataFrame with the `split` orientation
    sample_input='
    {
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
    }'

    echo $sample_input | curl -s -X POST $scoring_uri\
    -H 'Cache-Control: no-cache'\
    -H 'Content-Type: application/json'\
    -d @-

You can also test your deployments locally first using the option `run-local`:

.. code-block:: bash

    mlflow deployments run-local --name <deployment-name> -m models:/<model-name>/<model-version> -t <azureml-mlflow-tracking-url>

For more info, see:

.. code-block:: bash

    mlflow deployments help -t azureml


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

    from pyspark.sql.functions import struct
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    pyfunc_udf = mlflow.pyfunc.spark_udf(spark, <path-to-model>)
    df = spark_df.withColumn("prediction", pyfunc_udf(struct(<feature-names>)))

If a model contains a signature, the UDF can be called without specifying column name arguments.
In this case, the UDF will be called with column names from signature, so the evaluation
dataframe's column names must match the model signature's column names.

.. rubric:: Example

.. code-block:: py

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    pyfunc_udf = mlflow.pyfunc.spark_udf(spark, <path-to-model-with-signature>)
    df = spark_df.withColumn("prediction", pyfunc_udf())

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

.. _IntegerType: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.types.IntegerType.html#pyspark.sql.types.IntegerType
.. _LongType: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.types.LongType.html#pyspark.sql.types.LongType
.. _FloatType: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.types.FloatType.html#pyspark.sql.types.FloatType
.. _DoubleType: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.types.DoubleType.html#pyspark.sql.types.DoubleType
.. _StringType: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.types.StringType.html#pyspark.sql.types.StringType
.. _ArrayType: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.types.ArrayType.html#pyspark.sql.types.ArrayType

.. rubric:: Example

.. code-block:: py

    from pyspark.sql.types import ArrayType, FloatType
    from pyspark.sql.functions import struct
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    pyfunc_udf = mlflow.pyfunc.spark_udf(
        spark,
        "path/to/model",
        result_type=ArrayType(FloatType())
    )
    # The prediction column will contain all the numeric columns returned by the model as floats
    df = spark_df.withColumn("prediction", pyfunc_udf(struct("name", "age")))


If you want to use conda to restore the python environment that was used to train the model,
set the `env_manager` argument when calling :py:func:`mlflow.pyfunc.spark_udf`.


.. rubric:: Example

.. code-block:: py

    from pyspark.sql.types import ArrayType, FloatType
    from pyspark.sql.functions import struct
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    pyfunc_udf = mlflow.pyfunc.spark_udf(
        spark,
        "path/to/model",
        result_type=ArrayType(FloatType()),
        env_manager="conda"  # Use conda to restore the environment used in training
    )
    df = spark_df.withColumn("prediction", pyfunc_udf(struct("name", "age")))



.. _deployment_plugin:

Deployment to Custom Targets
----------------------------
In addition to the built-in deployment tools, MLflow provides a pluggable
`mlflow.deployments Python API <python_api/mlflow.deployments.html#mlflow.deployments>`_ and
`mlflow deployments CLI <cli.html#mlflow-deployments>`_ for deploying
models to custom targets and environments. To deploy to a custom target, you must first install an
appropriate third-party Python plugin. See the list of known community-maintained plugins
`here <plugins.html#deployment-plugins>`_.


.. Note::
    APIs for deployment to custom targets are experimental, and may be altered in a future release.


Commands
^^^^^^^^
The `mlflow deployments` CLI contains the following commands, which can also be invoked programmatically
using the `mlflow.deployments Python API <python_api/mlflow.deployments.html#mlflow.deployments>`_:

* `Create <cli.html#mlflow-deployments-create>`_: Deploy an MLflow model to a specified custom target
* `Delete <cli.html#mlflow-deployments-delete>`_: Delete a deployment
* `Update <cli.html#mlflow-deployments-update>`_: Update an existing deployment, for example to
  deploy a new model version or change the deployment's configuration (e.g. increase replica count)
* `List <cli.html#mlflow-deployments-list>`_: List IDs of all deployments
* `Get <cli.html#mlflow-deployments-get>`_: Print a detailed description of a particular deployment
* `Run Local <cli.html#mlflow-deployments-run-local>`_: Deploy the model locally for testing
* `Help <cli.html#mlflow-deployments-help>`_: Show the help string for the specified target


For more info, see:

.. code-block:: bash

    mlflow deployments --help
    mlflow deployments create --help
    mlflow deployments delete --help
    mlflow deployments update --help
    mlflow deployments list --help
    mlflow deployments get --help
    mlflow deployments run-local --help
    mlflow deployments help --help


Community Model Flavors
-----------------------

MLflow VizMod
^^^^^^^^^^^^^

The `mlflow-vizmod <https://github.com/JHibbard/mlflow-vizmod/>`_ project allows data scientists
to be more productive with their visualizations. We treat visualizations as models - just like ML
models - thus being able to use the same infrastructure as MLflow to track, create projects,
register, and deploy visualizations.

Installation:

.. code-block:: bash

    pip install mlflow-vizmod

Example:

.. code-block:: python

    from sklearn.datasets import load_iris
    import altair as alt
    import mlflow_vismod

    df_iris = load_iris(as_frame=True)

    viz_iris = (
        alt.Chart(df_iris)
          .mark_circle(size=60)
          .encode(x="x", y="y", color="z:N")
          .properties(height=375, width=575)
          .interactive()
    )

    mlflow_vismod.log_model(
        model=viz_iris,
        artifact_path="viz",
        style="vegalite",
        input_example=df_iris.head(5),
    )
