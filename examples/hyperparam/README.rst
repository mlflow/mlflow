Hyperparameter Tuning Example
------------------------------

Example of how to do hyperparameter tuning with MLflow and some popular optimization libraries.

This example tries to optimize the RMSE metric of a Keras deep learning model on a wine quality
dataset. The Keras model is fitted by the ``train`` entry point and has two hyperparameters that we
try to optimize: ``learning-rate`` and ``momentum``. The input dataset is split into three parts: training,
validation, and test. The training dataset is used to fit the model and the validation dataset is used to
select the best hyperparameter values, and the test set is used to evaluate expected performance and
to verify that we did not overfit on the particular training and validation combination. All three
metrics are logged with MLflow and you can use the MLflow UI to inspect how they vary between different
hyperparameter values.

examples/hyperparam/MLproject has 4 targets:
  * train:
    train a simple deep learning model on the wine-quality dataset from our tutorial.
    It has 2 tunable hyperparameters: ``learning-rate`` and ``momentum``.
    Contains examples of how Keras callbacks can be used for MLflow integration.
  * random:
    perform simple random search over the parameter space.
  * hyperopt:
    use `Hyperopt <https://github.com/hyperopt/hyperopt>`_ to optimize hyperparameters.


Running this Example
^^^^^^^^^^^^^^^^^^^^

You can run any of the targets as a standard MLflow run.

.. code-block:: bash

    mlflow experiments create -n individual_runs

Creates experiment for individual runs and return its experiment ID.

.. code-block:: bash

    mlflow experiments create -n hyper_param_runs

Creates an experiment for hyperparam runs and return its experiment ID.

.. code-block:: bash

    mlflow run -e train --experiment-id <individual_runs_experiment_id> examples/hyperparam

Runs the Keras deep learning training with default parameters and log it in experiment 1.

.. code-block:: bash

    mlflow run -e random --experiment-id <hyperparam_experiment_id> examples/hyperparam

.. code-block:: bash

    mlflow run -e hyperopt --experiment-id <hyperparam_experiment_id> examples/hyperparam

Runs the hyperparameter tuning with either random search or Hyperopt and log the
results under ``hyperparam_experiment_id``.

You can compare these results by using ``mlflow ui``.
