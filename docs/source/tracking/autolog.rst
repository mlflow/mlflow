.. _automatic-logging:

======================================
Automatic Logging with MLflow Tracking
======================================
Auto logging is a powerful feature that allows you to log metrics, parameters, and models without the need for explicit log statements. All you need to do is to call
:py:func:`mlflow.autolog` before your training code. 

.. code-block:: python

    import mlflow

    mlflow.autolog()

    with mlflow.start_run():
        # your training code goes here
        ...

This will enable MLflow to automatically log various information about your run, including:

* **Metrics** - MLflow pre-selects a set of metrics to log, based on what model and library you use
* **Parameters** - hyper params specified for the training, plus default values provided by the library if not explicitly set
* **Model Signature** - logs :ref:`Model signature <model-signature>` instance, which describes input and output schema of the model
* **Artifacts** -  e.g. model checkpoints
* **Dataset** - dataset object used for training (if applicable), such as `tensorflow.data.Dataset`


How to Get started
==================

Step 1 - Get MLflow
-------------------

MLflow is available on PyPI. If you don't already have it installed on your system, you can install it with:

.. code-section::

    .. code-block:: bash
        :name: download-mlflow

        pip install mlflow

Step 2 - Insert ``mlflow.autolog`` in Your Code
-----------------------------------------------

For example, following code snippet shows how to enable autologging for a scikit-learn model:

.. code-block:: python

    import mlflow

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor

    mlflow.autolog()

    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    # MLflow triggers logging automatically upon model fitting
    rf.fit(X_train, y_train)

Step 3 - Execute Your Code
--------------------------

.. code-section::

    .. code-block:: bash
        :name: execute code

        python YOUR_ML_CODE.py


Step 4 - View Your Results in the MLflow UI
-------------------------------------------

Once your training job finishes, you can run following command to launch the MLflow UI:

.. code-section::

    .. code-block:: bash
        :name: view-results

        mlflow ui --port 8080

Then, navigate to `http://localhost:8080 <http://localhost:8080>`_ in your browser to view the results.

Customize Autologging Behavior
==============================

You can also control the behavior of autologging by passing arguments to :py:func:`mlflow.autolog` function.
For example, you can disable logging of model checkpoints and assosiate tags with your run as follows:

.. code-block:: python

    import mlflow

    mlflow.autolog(
        log_model_signatures=False,
        extra_tags={"YOUR_TAG": "VALUE"},
    )

See :py:func:`mlflow.autolog` for the full set of arguments you can use.

Enable / Disable Autologging for Specific Libraries
---------------------------------------------------
One common use case is to enable/disable autologging for a specific library. For example, if you train your model on PyTorch but use scikit-learn 
for data preprocessing, you may want to disable autologging for scikit-learn while keeping it enabled for PyTorch. You can achieve this by either 
(1) enable autologging only for PyTorch using PyTorch flavor (2) disable autologging for scikit-learn using its flavor with ``disable=True``.

.. code-block:: python

    import mlflow

    # Option 1: Enable autologging only for PyTorch
    mlflow.pytorch.autolog()

    # Option 2: Disable autologging for scikit-learn, but enable it for other libraries
    mlflow.sklearn.autolog(disable=True)
    mlflow.autolog()

Supported Libraries
===================

.. note::

    The generic autolog function :py:func:`mlflow.autolog` enables autologging for each supported library you have installed as soon as you import it.
    Alternatively, you can use library-specific autolog calls such as :py:func:`mlflow.pytorch.autolog` to explicitly enable (or disable) autologging for a particular library.

The following libraries support autologging:

.. contents::
  :local:
  :depth: 1

For flavors that automatically save models as an artifact, `additional files <https://mlflow.org/docs/latest/models.html#storage-format>`_ for dependency management are logged.

.. _autolog-fastai:

Fastai
------

Call the generic autolog function :py:func:`mlflow.fastai.autolog` before your training code to enable automatic logging of metrics and parameters.
See an example usage with `Fastai <https://github.com/mlflow/mlflow/tree/master/examples/fastai>`_.

Autologging captures the following information:

.. _EarlyStoppingCallback: https://docs.fast.ai/callbacks.html#EarlyStoppingCallback
.. _OneCycleScheduler: https://docs.fast.ai/callbacks.html#OneCycleScheduler

+-----------+------------------------+----------------------------------------------------------+---------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Framework | Metrics                | Parameters                                               | Tags          | Artifacts                                                                                                                                                             |
+-----------+------------------------+----------------------------------------------------------+---------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| fastai    | user-specified metrics | Logs optimizer data as parameters. For example,          |  --           | Model checkpoints are logged to a ‘models’ directory; `MLflow Model`_ (fastai Learner model) on training end; Model summary text is logged                            |
|           |                        | ``epochs``, ``lr``, ``opt_func``, etc;                   |               |                                                                                                                                                                       |
|           |                        | Logs the parameters of the `EarlyStoppingCallback`_ and  |               |                                                                                                                                                                       |
|           |                        | `OneCycleScheduler`_ callbacks                           |               |                                                                                                                                                                       |
+-----------+------------------------+----------------------------------------------------------+---------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+

.. _autolog-gluon:

Gluon
-----
Call the generic autolog function :py:func:`mlflow.gluon.autolog` before your training code to enable automatic logging of metrics and parameters.
See example usages with `Gluon <https://github.com/mlflow/mlflow/tree/master/examples/gluon>`_ .

Autologging captures the following information:

+------------------+--------------------------------------------------------+----------------------------------------------------------+---------------+-------------------------------------------------------------------------------------------------------------------------------+
| Framework        | Metrics                                                | Parameters                                               | Tags          | Artifacts                                                                                                                     |
+------------------+--------------------------------------------------------+----------------------------------------------------------+---------------+-------------------------------------------------------------------------------------------------------------------------------+
| Gluon            | Training loss; validation loss; user-specified metrics | Number of layers; optimizer name; learning rate; epsilon | --            | `MLflow Model <https://mlflow.org/docs/latest/models.html>`_ (Gluon model); on training end                                   |
+------------------+--------------------------------------------------------+----------------------------------------------------------+---------------+-------------------------------------------------------------------------------------------------------------------------------+

.. _autolog-keras:

Keras/TensorFlow
----------------
Call the generic autolog function or :py:func:`mlflow.tensorflow.autolog` before your training code to enable automatic logging of metrics and parameters. As an example, try running the `Keras/Tensorflow example <https://github.com/mlflow/mlflow/blob/master/examples/keras/train.py>`_.

Note that only versions of ``tensorflow>=2.3`` are supported.
The respective metrics associated with ``tf.estimator`` and ``EarlyStopping`` are automatically logged.
As an example, try running the `Keras/TensorFlow example <https://github.com/mlflow/mlflow/blob/master/examples/keras/train.py>`_.

Autologging captures the following information:

+------------------------------------------+------------------------------------------------------------+-------------------------------------------------------------------------------------+---------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| Framework/module                         | Metrics                                                    | Parameters                                                                          | Tags          | Artifacts                                                                                                                                     |
+------------------------------------------+------------------------------------------------------------+-------------------------------------------------------------------------------------+---------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| ``tf.keras``                             | Training loss; validation loss; user-specified metrics     | ``fit()`` parameters; optimizer name; learning rate; epsilon                        | --            | Model summary on training start; `MLflow Model <https://mlflow.org/docs/latest/models.html>`_ (Keras model); TensorBoard logs on training end |
+------------------------------------------+------------------------------------------------------------+-------------------------------------------------------------------------------------+---------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| ``tf.keras.callbacks.EarlyStopping``     | Metrics from the ``EarlyStopping`` callbacks. For example, | ``fit()`` parameters from ``EarlyStopping``.                                        | --            | --                                                                                                                                            |
|                                          | ``stopped_epoch``, ``restored_epoch``,                     | For example, ``min_delta``, ``patience``, ``baseline``,                             |               |                                                                                                                                               |
|                                          | ``restore_best_weight``, etc                               | ``restore_best_weights``, etc                                                       |               |                                                                                                                                               |
+------------------------------------------+------------------------------------------------------------+-------------------------------------------------------------------------------------+---------------+-----------------------------------------------------------------------------------------------------------------------------------------------+

If no active run exists when ``autolog()`` captures data, MLflow will automatically create a run to log information to.
Also, MLflow will then automatically end the run once training ends via calls to ``tf.keras.fit()``.

If a run already exists when ``autolog()`` captures data, MLflow will log to that run but not automatically end that run after training. You will have to manually stop the run if you wish to start a new run context for logging to a new run.

.. _autolog-langchain:

LangChain
---------

Call the generic autolog function :py:func:`mlflow.langchain.autolog` before your training code to enable automatic logging of traces. See `LangChain Autologging <../llms/langchain/autologging.html>`_ for more details.

Autologging captures the following information:

+-----------+-------------+----------------+---------------+---------------------------------------------------------------------------------+
| Framework | Metrics     | Parameters     | Tags          | Artifacts                                                                       |
+-----------+-------------+----------------+---------------+---------------------------------------------------------------------------------+
| LangChain | --          | --             |  --           | - Traces                                                                        |
|           |             |                |               | - `MLflow Model`_ (LangChain model) with model signature on training end        |
|           |             |                |               | - Input example                                                                 |
+-----------+-------------+----------------+---------------+---------------------------------------------------------------------------------+

.. _autolog-llamaindex:

LlamaIndex
----------

Call the generic autolog function :py:func:`mlflow.llama_index.autolog` before your training code to enable automatic logging of traces. 

Autologging captures the following information:

+-----------+-------------+----------------+---------------+---------------------------------------------------------------------------------+
| Framework | Metrics     | Parameters     | Tags          | Artifacts                                                                       |
+-----------+-------------+----------------+---------------+---------------------------------------------------------------------------------+
| LlamaIndex| --          | --             |  --           | - Traces                                                                        |
+-----------+-------------+----------------+---------------+---------------------------------------------------------------------------------+

.. _autolog-lightgbm:

LightGBM
--------
Call the generic autolog function :py:func:`mlflow.lightgbm.autolog` before your training code to enable automatic logging of metrics and parameters.

Autologging captures the following information:

+-----------+------------------------+------------------------------+---------------+-----------------------------------------------------------------------------------------------------------+
| Framework | Metrics                | Parameters                   | Tags          | Artifacts                                                                                                 |
+-----------+------------------------+------------------------------+---------------+-----------------------------------------------------------------------------------------------------------+
| LightGBM  | user-specified metrics | `lightgbm.train`_ parameters | --            | `MLflow Model`_ (LightGBM model) with model signature on training end; feature importance; input example  |
+-----------+------------------------+------------------------------+---------------+-----------------------------------------------------------------------------------------------------------+

If early stopping is activated, metrics at the best iteration will be logged as an extra step/iteration.

.. _lightgbm.train: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html#lightgbm-train

.. _autolog-openai:

OpenAI
------

Call the generic autolog function :py:func:`mlflow.openai.autolog` before your training code to enable automatic logging of artifacts.
See an example usage with `OpenAI <https://github.com/mlflow/mlflow/tree/master/examples/openai>`_.

Autologging captures the following information:

+-----------+-------------+----------------+---------------+------------------------------------------------------------------------------+
| Framework | Metrics     | Parameters     | Tags          | Artifacts                                                                    |
+-----------+-------------+----------------+---------------+------------------------------------------------------------------------------+
| OpenAI    | --          | --             |  --           | - `MLflow Model`_ (OpenAI model) with model signature on training end        |
|           |             |                |               | - Input example                                                              |
+-----------+-------------+----------------+---------------+------------------------------------------------------------------------------+

.. _autolog-paddle:

Paddle
------
Call the generic autolog function :py:func:`mlflow.paddle.autolog` before your training code to enable automatic logging of metrics and parameters.

Autologging captures the following information:

+-----------+------------------------+--------------------------------+---------------+---------------------------------------------------------------------------------------------------------+
| Framework | Metrics                | Parameters                     | Tags          | Artifacts                                                                                               |
+-----------+------------------------+--------------------------------+---------------+---------------------------------------------------------------------------------------------------------+
| Paddle    | user-specified metrics | `paddle.Model.fit`_ parameters | --            | `MLflow Model`_ (Paddle model) with model signature on training end                                     |
+-----------+------------------------+--------------------------------+---------------+---------------------------------------------------------------------------------------------------------+

.. _paddle.Model.fit: https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/Model_en.html

.. _autolog-pyspark:

PySpark
-------

Call :py:func:`mlflow.pyspark.ml.autolog` before your training code to enable automatic logging of metrics, params, and models.
See example usage with `PySpark <https://github.com/mlflow/mlflow/tree/master/examples/pyspark_ml_autologging>`_.

Autologging for pyspark ml estimators captures the following information:

+---------------------------------------+--------------------------+------------------------------+---------------------------------------------------------+
| Metrics                               | Parameters               | Tags                         | Artifacts                                               |
+---------------------------------------+--------------------------+------------------------------+---------------------------------------------------------+
| Post training metrics obtained by     | Parameters obtained by   | - Class name                 | - `MLflow Model`_ containing a fitted estimator         |
| ``Evaluator.evaluate``                | ``Estimator.fit``        | - Fully qualified class name | - ``metric_info.json`` for post training metrics        |
+---------------------------------------+--------------------------+------------------------------+---------------------------------------------------------+

.. _autolog-pytorch:

PyTorch
-------

Call the generic autolog function :py:func:`mlflow.pytorch.autolog` before your PyTorch Lightning training code to enable automatic logging of metrics, parameters, and models. See example usages `here <https://github.com/chauhang/mlflow/tree/master/examples/pytorch/MNIST>`__. Note
that currently, PyTorch autologging supports only models trained using PyTorch Lightning.

Autologging is triggered on calls to ``pytorch_lightning.trainer.Trainer.fit`` and captures the following information:

+------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------------------------------+---------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| Framework/module                               | Metrics                                                     | Parameters                                                                           | Tags          | Artifacts                                                                                                                                     |
+------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------------------------------+---------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| ``pytorch_lightning.trainer.Trainer``          | Training loss; validation loss; average_test_accuracy;      | ``fit()`` parameters; optimizer name; learning rate; epsilon.                        | --            | Model summary on training start, `MLflow Model <https://mlflow.org/docs/latest/models.html>`_ (PyTorch model) on training end;                |
|                                                | user-defined-metrics.                                       |                                                                                      |               |                                                                                                                                               |
|                                                |                                                             |                                                                                      |               |                                                                                                                                               |
|                                                |                                                             |                                                                                      |               |                                                                                                                                               |
|                                                |                                                             |                                                                                      |               |                                                                                                                                               |
+------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------------------------------+---------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| ``pytorch_lightning.callbacks.earlystopping``  | Training loss; validation loss; average_test_accuracy;      | ``fit()`` parameters; optimizer name; learning rate; epsilon                         | --            | Model summary on training start; `MLflow Model <https://mlflow.org/docs/latest/models.html>`_ (PyTorch model) on training end;                |
|                                                | user-defined-metrics.                                       | Parameters from the ``EarlyStopping`` callbacks.                                     |               | Best PyTorch model checkpoint, if training stops due to early stopping callback.                                                              |
|                                                | Metrics from the ``EarlyStopping`` callbacks.               | For example, ``min_delta``, ``patience``, ``baseline``,``restore_best_weights``, etc |               |                                                                                                                                               |
|                                                | For example, ``stopped_epoch``, ``restored_epoch``,         |                                                                                      |               |                                                                                                                                               |
|                                                | ``restore_best_weight``, etc.                               |                                                                                      |               |                                                                                                                                               |
|                                                |                                                             |                                                                                      |               |                                                                                                                                               |
|                                                |                                                             |                                                                                      |               |                                                                                                                                               |
|                                                |                                                             |                                                                                      |               |                                                                                                                                               |
+------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------------------------------+---------------+-----------------------------------------------------------------------------------------------------------------------------------------------+

If no active run exists when ``autolog()`` captures data, MLflow will automatically create a run to log information, ending the run once
the call to ``pytorch_lightning.trainer.Trainer.fit()`` completes.

If a run already exists when ``autolog()`` captures data, MLflow will log to that run but not automatically end that run after training.

.. note::
  - Parameters not explicitly passed by users (parameters that use default values) while using ``pytorch_lightning.trainer.Trainer.fit()`` are not currently automatically logged
  - In case of a multi-optimizer scenario (such as usage of autoencoder), only the parameters for the first optimizer are logged

.. _autolog-sklearn:

Scikit-learn
------------

Call :py:func:`mlflow.sklearn.autolog` before your training code to enable automatic logging of sklearn metrics, params, and models.
See example usage `here <https://github.com/mlflow/mlflow/tree/master/examples/sklearn_autolog>`_.

Autologging for estimators (e.g. `LinearRegression`_) and meta estimators (e.g. `Pipeline`_) creates a single run and logs:

+-------------------------+--------------------------+------------------------------+------------------+
| Metrics                 | Parameters               | Tags                         | Artifacts        |
+-------------------------+--------------------------+------------------------------+------------------+
| Training score obtained | Parameters obtained by   | - Class name                 | Fitted estimator |
| by ``estimator.score``  | ``estimator.get_params`` | - Fully qualified class name |                  |
+-------------------------+--------------------------+------------------------------+------------------+


.. _LinearRegression:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

.. _Pipeline:
    https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html


Autologging for parameter search estimators (e.g. `GridSearchCV`_) creates a single parent run and nested child runs

.. code-block::

  - Parent run
    - Child run 1
    - Child run 2
    - ...

containing the following data:

+------------------+----------------------------+-------------------------------------------+------------------------------+-------------------------------------+
| Run type         | Metrics                    | Parameters                                | Tags                         | Artifacts                           |
+------------------+----------------------------+-------------------------------------------+------------------------------+-------------------------------------+
| Parent           | Training score             | - Parameter search estimator's parameters | - Class name                 | - Fitted parameter search estimator |
|                  |                            | - Best parameter combination              | - Fully qualified class name | - Fitted best estimator             |
|                  |                            |                                           |                              | - Search results csv file           |
+------------------+----------------------------+-------------------------------------------+------------------------------+-------------------------------------+
| Child            | CV test score for          | Each parameter combination                | - Class name                 | --                                  |
|                  | each parameter combination |                                           | - Fully qualified class name |                                     |
+------------------+----------------------------+-------------------------------------------+------------------------------+-------------------------------------+

.. _GridSearchCV:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


.. _autolog-spark:

Spark
-----

Initialize a SparkSession with the mlflow-spark JAR attached (e.g.
``SparkSession.builder.config("spark.jars.packages", "org.mlflow.mlflow-spark")``) and then
call the generic autolog function :py:func:`mlflow.spark.autolog` to enable automatic logging of Spark datasource
information at read-time, without the need for explicit
log statements. Note that autologging of Spark ML (MLlib) models is not yet supported.

Autologging captures the following information:

+------------------+---------+------------+----------------------------------------------------------------------------------------------+-----------+
| Framework        | Metrics | Parameters |  Tags                                                                                        | Artifacts |
+------------------+---------+------------+----------------------------------------------------------------------------------------------+-----------+
| Spark            | --      | --         | Single tag containing source path, version, format. The tag contains one line per datasource | --        |
+------------------+---------+------------+----------------------------------------------------------------------------------------------+-----------+

.. note::
  - Moreover, Spark datasource autologging occurs asynchronously - as such, it's possible (though unlikely) to see race conditions when launching short-lived MLflow runs that result in datasource information not being logged.

.. important::
    With Pyspark 3.2.0 or above, Spark datasource autologging requires ``PYSPARK_PIN_THREAD`` environment variable to be set to ``false``.

.. _autolog-statsmodels:

Statsmodels
-----------
Call the generic autolog function :py:func:`mlflow.statsmodels.autolog` before your training code to enable automatic logging of metrics and parameters.

Autologging captures the following information:

+--------------+------------------------+------------------------------------------------+---------------+-----------------------------------------------------------------------------+
| Framework    | Metrics                | Parameters                                     | Tags          | Artifacts                                                                   |
+--------------+------------------------+------------------------------------------------+---------------+-----------------------------------------------------------------------------+
| Statsmodels  | user-specified metrics | `statsmodels.base.model.Model.fit`_ parameters | --            | `MLflow Model`_ (`statsmodels.base.wrapper.ResultsWrapper`) on training end |
+--------------+------------------------+------------------------------------------------+---------------+-----------------------------------------------------------------------------+

.. note::
  - Each model subclass that overrides `fit` expects and logs its own parameters.

.. _statsmodels.base.model.Model.fit: https://www.statsmodels.org/dev/dev/generated/statsmodels.base.model.Model.html

.. _autolog-xgboost:

XGBoost
-------
Call the generic autolog function :py:func:`mlflow.xgboost.autolog` before your training code to enable automatic logging of metrics and parameters.

Autologging captures the following information:

+-----------+------------------------+-----------------------------+---------------+---------------------------------------------------------------------------------------------------------+
| Framework | Metrics                | Parameters                  | Tags          | Artifacts                                                                                               |
+-----------+------------------------+-----------------------------+---------------+---------------------------------------------------------------------------------------------------------+
| XGBoost   | user-specified metrics | `xgboost.train`_ parameters | --            | `MLflow Model`_ (XGBoost model) with model signature on training end; feature importance; input example |
+-----------+------------------------+-----------------------------+---------------+---------------------------------------------------------------------------------------------------------+

If early stopping is activated, metrics at the best iteration will be logged as an extra step/iteration.

.. _xgboost.train: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train
.. _MLflow Model: https://mlflow.org/docs/latest/models.html
