## MLflow examples

### Quick Start example

- `quickstart/mlflow_tracking.py` is a basic example to introduce MLflow concepts.

## Tutorials

Various examples that depict MLflow tracking, project, and serving use cases.

- `pytorch` uses CNN on MNIST dataset for character recognition. The example logs TensorBoard events
  and stores (logs) them as MLflow artifacts.
- `remote_store` has a usage example of REST based backed store for tracking.
- `r_wine` demonstrates how to log parameters, metrics, and models from R.
- `sklearn_elasticnet_diabetes` uses the sklearn diabetes dataset to predict diabetes progression
  using ElasticNet.
- `sklearn_elasticnet_wine_quality` is an example for MLflow projects. This uses the Wine
  Quality dataset and Elastic Net to predict quality. The example uses `MLproject` to set up a
  Conda environment, define parameter types and defaults, entry point for training, etc.
- `sklearn_logistic_regression` is a simple MLflow example with hooks to log training data to MLflow
  tracking server.
- `supply_chain_security` shows how to strengthen the security of ML projects against supply-chain attacks by enforcing hash checks on Python packages.
- `docker` demonstrates how to create and run an MLflow project using docker (rather than conda)
  to manage project dependencies

## Demos

- `demos` folder contains notebooks used during MLflow presentations.
