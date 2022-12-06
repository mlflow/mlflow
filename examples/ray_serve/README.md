# MLflow-Ray-Serve deployment plugin

In this example, we will first train a model to classify the Iris dataset using `sklearn`. Next, we will deploy our model on Ray Serve and then scale it up, all using the MLflow Ray Serve plugin.

The plugin supports both a command line interface and a Python API. Below we will use the command line interface. For the full API documentation, see https://www.mlflow.org/docs/latest/cli.html#mlflow-deployments and https://www.mlflow.org/docs/latest/python_api/mlflow.deployments.html.

## Plugin Installation

Please follow the installation instructions for the Ray Serve deployment plugin: https://github.com/ray-project/mlflow-ray-serve

## Instructions

First, navigate to the directory for this example, `mlflow/examples/ray_serve/`.

Second, run `python train_model.py`. This trains and saves our classifier to the MLflow Model Registry and sets up automatic logging to MLflow. It also prints the mean squared error and the target names, which are species of iris:

```
MSE: 1.04
Target names:  ['setosa' 'versicolor' 'virginica']
```

Next, set the MLflow Tracking URI environment variable to the location where the Model Registry resides:

`export MLFLOW_TRACKING_URI=sqlite:///mlruns.db`

Now start a Ray cluster with the following command:

`ray start --head`

Next, start a long-running Ray Serve instance on your Ray cluster:

`serve start`

Ray Serve is now running and ready to deploy MLflow models. The MLflow Ray Serve plugin features both a Python API as well as a command-line interface. For this example, we'll use the command line interface.

Finally, we can deploy our model by creating an instance using the following command:

`mlflow deployments create -t ray-serve -m models:/RayMLflowIntegration/1 --name iris:v1`

The `-t` parameter here is the deployment target, which in our case is Ray Serve. The `-m` parameter is the Model URI, which consists of the registered model name and version in the Model Registry.

We can now run a prediction on our deployed model as follows. The file `input.json` contains a sample input containing the sepal length, sepal width, petal length, petal width of a sample flower. Now we can get the prediction using the following command:

`mlflow deployments predict -t ray-serve --name iris:v1 --input-path input.json`

This will output `[0]`, `[1]`, or `[2]`, corresponding to the species listed above in the target names.

We can scale our deployed model up to use several replicas, improving throughput:

`mlflow deployments update -t ray-serve --name iris:v1 --config num_replicas=2`

Here we only used 2 replicas, but you can use as many as you like, depending on how many CPU cores are available in your Ray cluster.

The deployed model instance can be deleted as follows:

`mlflow deployments delete -t ray-serve --name iris:v1`

To tear down the Ray cluster, run the following command:

`ray stop`
