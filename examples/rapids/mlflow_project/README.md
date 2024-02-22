### Train and Publish Locally With RAPIDS and MLflow

**[RAPIDS](https://rapids.ai/)** is a suite of open source libraries for GPU-accelerated analytics.

**[RAPIDS cuML](https://github.com/rapidsai/cuml)** matches the scikit-learn API, so it can build on MLflow's existing support for scikit-learn-like models to support
persistence and deployment."

The example workflows below train RAPIDs regression models to predict airline flight delays, using
MLflow to log models and deploy them as local REST API endpoints for real-time inference. You can run them:

- On a GPU-enabled instance for free in Colab. If following this approach, we recommend using the "Jupyter notebook workflow" below
  and following the setup steps in [this Colab notebook](https://colab.research.google.com/drive/1rY7Ln6rEE1pOlfSHCYOVaqt8OvDO35J0#forceEdit=true&offline=true&sandboxMode=true) to configure your
  environment.

- On your own machine with an NVIDIA GPU and CUDA installed. See the [RAPIDS getting-started guide](https://rapids.ai/start.html)
  for more details on necessary prerequisites for running the examples on your own machine.

#### Jupyter Notebook Workflow

[Jupyter Notebook](notebooks/rapids_mlflow.ipynb)

#### CLI Based Workflow

1. Create data
   1. `cd examples/rapids/mlflow_project`
      ```shell script
      # Create iris.csv
      python -c "from sklearn.datasets import load_iris; d = load_iris(as_frame=True); d.frame.to_csv('iris.csv', index=False)"
      ```
1. Set MLflow tracking uri
   1. ```shell script
       export MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow-db.sqlite
      ```
1. Train the model using a single run.
   1. ```shell script
      # Launch the job
      mlflow run . -e simple\
               --experiment-name RAPIDS-CLI \
               -P max_depth=10 -P max_features=0.75 -P n_estimators=500 \
               -P conda-env=$PWD/envs/conda.yaml \
               -P fpath=iris.csv
      ```
1. Train the model with Hyperopt

   1. ```shell script
      # Launch the job
      mlflow run . -e hyperopt \
               --experiment-name RAPIDS-CLI \
               -P conda-env=$PWD/envs/conda.yaml \
               -P fpath=iris.csv
      ```
   1. In the output, note: "Created version '[VERSION]' of model 'rapids_mlflow'"

1. Deploy your model

   1. Deploy your model
      1. `$ mlflow models serve --env-manager=local -m models:/rapids_mlflow_cli/[VERSION] -p 55755`

1. Query the deployed model with test data `src/sample_server_query.sh` example script.
   1. `bash src/sample_server_query.sh`
