Hyperparameter Tuning Example
------------------------------

Example of how you can do hyperparameter tuning with MLflow and some popular optimization libraries.

In this example, we try to optimize rmse metric of a Keras deep learning model on a wine quality
dataset. The Keras model is fitted by the ``train`` entry point and has two hyperparameters that we
try to optimize - learning rate and momentum. The input dataset is split into three parts - training
validation and test. The training dataset is used to fit the model, validation dataset is used to
select the best hyperparameter values and the test set is used to evaluate expected performance and
to verify that we did not overfit on the particular training and validation combination. All three
metrics are logged with MLflow and you can use MLflow ui to inspect how they vary between different
hyperparameter values.

examples/hyperparam/MLproject has 4 targets:
  * train
    train simple deep learning model on the wine-quality dataset from our tutorial.
    It has 3 tunable hyperparameters - learning_rate, beta1, beta2.
    Contains examples of how Keras callbacks can be used for mlflow integration.
  * random
    perform simple random search over the parameter space.
  * gpyopt
    use `GPyOpt <https://github.com/SheffieldML/GPyOpt>`_ to optimize hyperparameters of train.
    GPyOpt can run multiple mlflow runs in parallel if run with batch-size > 1 and max_p > 1.
  * hyperopt
    use `Hyperopt <https://github.com/hyperopt/hyperopt>`_ to optimize hyperparameters.

All the hyperparameter targets take optional experiment id for training runs. If provided,
training runs will be logged under this experiment id. This is a short term solution to organizing
the runs so that it is easy to view individual training runs and the hyperparameter runs separately.
In the future this will be achieved by MLflow tags.


Running this Example
^^^^^^^^^^^^^^^^^^^^

You can run any of the targets as a standard mlflow run.

.. code:: bash

    mlflow experiments create individual_runs

This will create experiment for individual runs and return its experiment id.

.. code:: bash

    mlflow experiments create hyper_param_runs

This will create experiment for hyperparam runs and return its experiment id.

.. code:: bash

    mlflow run -e train --experiment-id <individual_runs_experiment_id> example/hyperparam

This will run the Keras deep learning training with default parameters and log it in experiment 1.

.. code:: bash

    mlflow run -e random --experiment-id <hyperparam_experiment_id>  -P \
        training_experiment_id=<individual_runs_experiment_id> example/hyperparam

.. code:: bash

    mlflow run -e gpyopt --experiment-id <hyperparam_experiment_id>  -P \
        training_experiment_id=<individual_runs_experiment_id> example/hyperparam

.. code:: bash

    mlflow run -e hyperopt --experiment-id <hyperparam_experiment_id> -P \
        training_experiment_id=<individual_runs_experiment_id> example/hyperparam

This will run the hyperparameter tuning with either random search or GpyOpt or Hyperopt and log the
results under ``hyperparam_experiment_id``.

You can compare these results by using ``mlflow ui``!
