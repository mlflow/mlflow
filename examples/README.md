## MLflow examples

### Quick Start example
* `quickstart/mlflow_tracking.py` is a basic example to introduce MLflow concepts.

## Tutorials
Various examples that depict MLflow tracking, project, and serving use cases.

* `h2o` depicts how MLflow can be use to track various random forest architectures to train models 
for predicting wine quality.
* `hyperparam`  shows how to do hyperparameter tuning with MLflow and some popular optimization libraries.
* `multistep_workflow` is an end-to-end of a data ETL and ML training pipeline built as an MLflow 
project. The example shows how parts of the workflow can leverage from previously run steps.
* `pytorch` uses CNN on MNIST dataset for character recognition. The example logs TensorBoard events
and stores (logs) them as MLflow artifacts.
* `remote_store` has a usage example of REST based backed store for tracking.
* `r_wine` demonstrates how to log parameters, metrics, and models from R.
* `sklearn_elasticnet_diabetes` uses the sklearn diabetes dataset to predict diabetes progression
   using ElasticNet.
* `sklearn_elasticnet_wine_quality` is an example for MLflow projects. This uses the Wine
   Quality dataset and Elastic Net to predict quality. The example uses `MLproject` to set up a 
   Conda environment, define parameter types and defaults, entry point for training, etc.
* `sklearn_logisic_regression` is a simple MLflow example with hooks to log training data to MLflow
tracking server.
* `tensorflow` is an end-to-end one run example from train to predict.
