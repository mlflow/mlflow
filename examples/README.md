## MLflow examples

### Quick Start example

- `quickstart/mlflow_tracking.py` is a basic example to introduce MLflow concepts.

## Tutorials

Various examples that depict MLflow tracking, project, and serving use cases.

- `h2o` depicts how MLflow can be use to track various random forest architectures to train models
  for predicting wine quality.
- `hyperparam` shows how to do hyperparameter tuning with MLflow and some popular optimization libraries.
- `keras` modifies
  [a Keras classification example](https://github.com/keras-team/keras/blob/ed07472bc5fc985982db355135d37059a1f887a9/examples/reuters_mlp.py)
  and uses MLflow's `mlflow.tensorflow.autolog()` API to automatically log metrics and parameters
  to MLflow during training.
- `multistep_workflow` is an end-to-end of a data ETL and ML training pipeline built as an MLflow
  project. The example shows how parts of the workflow can leverage from previously run steps.
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
- `tensorflow` contains end-to-end one run examples from train to predict for TensorFlow 2.8+ It includes usage of MLflow's
  `mlflow.tensorflow.autolog()` API, which captures TensorBoard data and logs to MLflow with no code change.
- `docker` demonstrates how to create and run an MLflow project using docker (rather than conda)
  to manage project dependencies
- `johnsnowlabs` gives you access to [20.000+ state-of-the-art enterprise NLP models in 200+ languages](https://nlp.johnsnowlabs.com/models) for medical, finance, legal and many more domains.

## Demos

- `demos` folder contains notebooks used during MLflow presentations.
