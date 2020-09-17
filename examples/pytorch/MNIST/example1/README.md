## MNIST example with MLFlow

In this example, we train a model to predict handwritten digits.The autolog code uses Pytorch Lightning's MLFlowLogger to log metrics. 
The code is trained using pytorch lightning loop and the autolog function call in the main - `autolog()`
is responsible for logging the params, metrics, model summary and the model.
Early stopping condition and Model Checkpoint are added in this example.

### Code related to MLflow:
* [`mlflow.pytorch.pytorch_autolog`]
This is an experimental api that logs ML model artifacts and metrics.
The metrics are logged during training of the model.

### Running the code
To run the example via MLflow, navigate to the `mlflow/examples/pytorch/MNIST/example1` directory and run the command

```
mlflow run .
```

This will run `mnist_autolog_example1.py` with the default set of parameters such as  `--max_epochs=5`. You can see the default value in the `MLproject` file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P epochs=X
```

where `X` is your desired value for `epochs`.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument `--no-conda`.

```
mlflow run . --no-conda
```

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more information on MLflow tracking, click [here](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking) to view documentation.
