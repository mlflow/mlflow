## MLflow examples

### Quick Start example
* `quick_start/mlflow_tracking.py` is a basic example to introduce MLflow concepts.

## Tutorials
Various example that depict MLflow tracking, projects, and serving use cases.

* `backend_store`: has a usage example of REST based backed store for tracking.
* `h2o` depicts how MLflow can be use to track various random forest architectures to train models 
for predicting wine quality.
* `multistep_workflow` is an end-to-end of a data ETL and ML training pipeline built as an MLflow 
project. The example shows how parts of the workflow can leverage from previously run steps.
* `pytorch` uses CNN on MNIST dataset for character recognition. The example logs Tensorboard Events
and stores (logs) them as MLflow artifacts.
* `sklearn_elasticnet` on 2 MLflow examples:
   * `train_diabetes.py` : Uses the sklearn Diabetes dataset to predict diabetes progression 
   using ElasticNet.
   * `wine_quality` is an example for MLflow projects. This uses the previously mentioned Wine 
   Quality dataset and uses Elastic Net to predict quality. The example uses `MLproject` to setup a 
   conda environment, define parameter types and defaults, entry point for training ... etc.
* `sklearn_logisic_regression` is a simple ML flow with hooks to training data to MLflow 
tracking server.
* `tensorflow` is an end-to-end one run example from train to predict.
