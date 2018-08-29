Hyperparameter Tuning Example
------------------------------

Example of how you can use hyperparameter tuning with mlflow using external hyperparameter tuning
libraries. All three examples are implemented as an MLflow run entry point which evaluates the model
by in another MLflow run entry point. This way both the parent hyperparameter tuning run and the
spawned training runs get logged. All targets take optional experiment id for training runs. If
provided, training runs will be logged under this experiment id. This is a short term solution to
organizing the runs so that it is easy to view individual training runs and the hyperparameter runs
separately. In the future this will be achieved by MLflow tags.

examples/hyperparam/MLproject has 4 targets:
  * main
    trains simple deep learning model on the wine-quality dataset from our tutorial.
    It has 3 tunable hyperparameters - learning_rate, beta1, beta2.
    Contains examples of how Keras callbacks can be used for mlflow integration.
  * random
    performs simple random search over the parameter space.
  * gpyopt
    uses `GPyOpt <https://github.com/SheffieldML/GPyOpt>`_ to optimize hyperparameters of train.
    GPyOpt can run multiple mlflow runs in parallel if run with batch-size > 1 and max_p > 1.
  * hyperOpt
    uses `Hyperopt <https://github.com/hyperopt/hyperopt>`_ to optimize hyperparameters.


Running this Example
^^^^^^^^^^^^^^^^^^^^

You can run any of the targets as a standard mlflow run.

.. code:: bash

    mlflow experiments create individual_runs

This will create experiment for individual runs and return its experiment it.

.. code:: bash

    mlflow experiments create hyper_param_runs

This will create experiment for hyperparam runs and return its experiment it.

.. code:: bash

    mlflow run  --experiment-id <individual_runs_experiment_id> example/hyperparam

This will run the Keras deep learning training with default parameters and log it in experiment 1.

.. code:: bash

    mlflow run  -e random --experiment-id <hyperparam_experiment_id>  -P \
        training_experiment_id=<individual_runs_experiment_id> example/hyperparam

.. code:: bash

    mlflow run  -e gpyopt --experiment-id <hyperparam_experiment_id>  -P \
        training_experiment_id=<individual_runs_experiment_id> example/hyperparam

.. code:: bash

    mlflow run  -e hyperopt --experiment-id <hyperparam_experiment_id> -P \
        training_experiment_id=<individual_runs_experiment_id> example/hyperparam

This will run the hyperparameter tuning with either random search or GpyOpt or Hyperopt and log the
results under ``hyperparam_experiment_id``.

You can compare these results by using ``mlflow ui``!
