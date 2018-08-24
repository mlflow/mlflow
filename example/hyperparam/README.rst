=====================================
MLflow Hyper Parameter Tuning Example
=====================================
Example of how you can use hyper param tuning with mlflow using external hyper param tuning
libraries. 

examples/hyperparam/MLProject has 3 targets:
  * dl_train
    trains simple deep learning model on the wine-quality dataset from our tutorial.
    It has 3 tunable parameters - learning_rate, beta1, beta2.
    Contains examples of how keras callbacks can be used for mlflow integration.
  * GPyOpt
    uses `GPyOpt <https://github.com/SheffieldML/GPyOpt>`_. to optimize hyper parameters of dl_train
    GPyOpt can run multiple mlflow runs in parallel if run with batch-size > 1 and max_p > 1.
  * HyperOpt
    uses `Hyperopt <https://github.com/hyperopt/hyperopt>`_. to optimize hyper parameters.

Both targets take optional experiment id for training runs. If provided, training runs will be
logged under this experiment id (to avoid confusion in the ui).

Example usage:
~~~~~~~~~~~~~

mlflow run  -e HyperOpt --experiment-id 5 -P max_runs=16 -P max_epochs=32  -P training_experiment_id=6 example/hyperparam