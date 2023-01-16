## MNIST example with MLflow

This example demonstrates training of MNIST handwritten recognition model and logging it as torch scripted model.
`mlflow.pytorch.log_model()` is used to log the scripted model to MLflow and `mlflow.pytorch.load_model()` to load it from MLflow

### Code related to MLflow:

This will log the TorchScripted model into MLflow and load the logged model.

## Setting Tracking URI

MLflow tracking URI can be set using the environment variable `MLFLOW_TRACKING_URI`

Example: `export MLFLOW_TRACKING_URI=http://localhost:5000/`

For more details - https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded

### Running the code

To run the example via MLflow, navigate to the `mlflow/examples/pytorch/torchscript/MNIST` directory and run the command

```
mlflow run .
```

This will run `mnist_torchscript.py` with the default set of parameters such as `--max_epochs=5`. You can see the default value in the `MLproject` file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P epochs=X
```

where `X` is your desired value for `epochs`.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument `--env-manager=local`.

```
mlflow run . --env-manager=local
```

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more information on MLflow tracking, click [here](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking) to view documentation.
