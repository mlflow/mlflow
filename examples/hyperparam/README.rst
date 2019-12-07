Hyperparameter Tuning Example
------------------------------

This example demonstrates how to perform hyperparameter tuning with MLflow and some popular optimization libraries.

The example tries to optimize the RMSE metric of a Keras deep learning model on a wine quality
dataset. The Keras model is fitted by the ``train`` entry point and has two hyperparameters that we
try to optimize: ``learning-rate`` and ``momentum``. The input dataset is split into three parts: training,
validation, and test. The training dataset is used to fit the model and the validation dataset is used to
select the best hyperparameter values, and the test set is used to evaluate expected performance and
to verify that we did not overfit on the particular training and validation combination. All three
metrics are logged with MLflow, so you can use the MLflow UI to inspect how they vary between different
hyperparameter values.

examples/hyperparam/MLproject has 4 targets:
  * ``train``:
    train simple deep learning model on the wine-quality dataset from our tutorial.
    It has 2 tunable hyperparameters: ``learning-rate`` and ``momentum``.
    Contains examples of how Keras callbacks can be used for MLflow integration.
  * ``random``:
    perform simple random search over the parameter space.
  * ``gpyopt``:
    use `GPyOpt <https://github.com/SheffieldML/GPyOpt>`_ to optimize hyperparameters of train.
    GPyOpt can run multiple mlflow runs in parallel if run with ``batch-size`` > 1 and ``max_p`` > 1.
  * ``hyperopt``:
    use `Hyperopt <https://github.com/hyperopt/hyperopt>`_ to optimize hyperparameters.

What You'll Need
^^^^^^^^^^^^^^^^

To run this example, you'll need to:

   - Install MLflow, Keras, Gpyopt, and Hyperopt.

       .. code-block:: bash

           pip install mlflow
           pip install keras
           pip install gpyopt
           pip install hyperopt

   - Install `conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
   - Clone (download) the MLflow repository via ``git clone https://github.com/mlflow/mlflow``
   - ``cd`` into the ``examples/hyperparam`` directory within your clone of MLflow - we'll use this working
     directory for running the tutorial. We avoid running directly from our clone of MLflow as doing
     so would cause the tutorial to use MLflow from source, rather than your PyPI installation of
     MLflow.

Running this Example
^^^^^^^^^^^^^^^^^^^^

We'll first start by creating experiments to differentiate between individual training runs and runs
optimized by ``hyperopt``/``gpyopt``.

This command creates an experiment for individual runs and returns its experiment ID. This is ``<individual_runs_experiment_id>``.

.. code-block:: bash

    mlflow experiments create -n individual_runs

This command creates an experiment for hyperparam runs and returns its experiment ID. This is ``<hyperparam_experiment_id>``.

.. code-block:: bash

    mlflow experiments create -n hyper_param_runs

The following command runs the Keras deep learning training with default parameters and logs them in ``individual_runs``,
the first experiment we created.

.. code-block:: bash

    mlflow run -e train --experiment-id <individual_runs_experiment_id> examples/hyperparam

The next three commands use MLflow Projects to perform hyperparameter tuning with either random search, GpyOpt, or
Hyperopt, respectively. They each log their results under ``hyperparam_experiment_id``, and you can compare these results
by using the command ``mlflow ui`` and navigating to ``localhost:3000`` in your browser.

.. code-block:: bash

    mlflow run -e random --experiment-id <hyperparam_experiment_id>  -P examples/hyperparam

.. code-block:: bash

    mlflow run -e gpyopt --experiment-id <hyperparam_experiment_id>  -P examples/hyperparam

.. code-block:: bash

    mlflow run -e hyperopt --experiment-id <hyperparam_experiment_id> -P examples/hyperparam
