Hyper Parameter Tuning Example
------------------------------
Example of how you can use hyper param tuning with mlflow using external hyper param tuning
libraries. Both examples are implemented as an MLflow run entry point which evaluates the model by
calling another MLflow run entry point. This way both the parent hyper param tuning run and the
spawned training runs get logged. Both targets take optional experiment id for training runs. If
provided, training runs will be logged under this experiment id. This is a short term solution to
organizing the runs so that it is easy to view individual training runs and the hyper param runs
separately. In the future this will be achieved by MLflow tags.

examples/hyperparam/MLproject has 3 targets:
  * dl_train
    trains simple deep learning model on the wine-quality dataset from our tutorial.
    It has 3 tunable hyper parameters - learning_rate, beta1, beta2.
    Contains examples of how keras callbacks can be used for mlflow integration.
  * GPyOpt
    uses `GPyOpt <https://github.com/SheffieldML/GPyOpt>`_ to optimize hyper parameters of dl_train
    GPyOpt can run multiple mlflow runs in parallel if run with batch-size > 1 and max_p > 1.
  * HyperOpt
    uses `Hyperopt <https://github.com/hyperopt/hyperopt>`_ to optimize hyper parameters.


Running this Example
^^^^^^^^^^^^^^^^^^^^
You can run any of the targets as a standard mlflow run.

.. code::

mlflow experiments create individual_runs


This will create experiment for individual runs and return its experiment it.


.. code::

mlflow experiments create hyper_param_runs


This will create experiment for hyper param runs and return its experiment it.


.. code::

mlflow run  -e dl_train --experiment-id <individual_runs_experiment_id> example/hyperparam

This will run the keras deep learning training with default parameters and log it in experiment 1.

.. code::

mlflow run  -e GPyOpt --experiment-id <hyperparam_experiment_id>  -P training_experiment_id=<individual_runs_experiment_id> example/hyperparam

.. code::

mlflow run  -e HyperOpt --experiment-id <hyperparam_experiment_id> -P training_experiment_id=<individual_runs_experiment_id> example/hyperparam

This will run the hyper parameter tuning with either GpyOpt or Hyperopt and log the results under
<hyperparam_experiment_id>.

You can compare these results by using ``mlflow ui``!
