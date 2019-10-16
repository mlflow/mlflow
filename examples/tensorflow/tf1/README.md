## MLflow with TensorFlow 1.X

In this example, we train a `tf.estimator.DNNRegressor` on the TensorFlow [Boston Housing dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html).
With MLflow, we use `mlflow.tensorflow.autolog()` along with `mlflow.log_param()` and `mlflow.log_metric()` in order to capture data about how we trained the target model.
We also use MLflow in order to generically save the model as a `mlflow.pyfunc`, then load it back to predict on pandas DataFrames.

### Code related to MLflow:
* [`mlflow.tensorflow.autolog()`](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging-from-tensorflow-and-keras-experimental):
This is an experimental api that logs ML model artifacts and TensorBoard metrics created by the `tf.estimator` we are using.
The TensorBoard metrics are logged during training of the model. By default, MLflow autologs every 100 steps.
The ML model artifact creation is handled during the call to `tf.estimator.export_saved_model()`.

* [`mlflow.log_param()`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_param):
This manually logs a parameter that autolog doesn't cover. In this example, we log the `hidden_units` parameter.

* [`mlflow.log_metric()`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric):
This manually logs a metric that autolog doesn't cover. In this example, we log the mean square error of the model.

* [`mlflow.pyfunc.load_model()`](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model):
This function loads back the model as a generic python function. You can predict on this with a pandas DataFrame input.

### Running the code
To run the example via MLflow, navigate to the `mlflow/examples/tensorflow/tf1` directory and run the command

```
mlflow run .
```

This will run `train_predict_.py` with the default parameter `--steps=1000`. You can see the default value in the `MLproject` file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P steps=X
```

where `X` is your desired value for `steps`.

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


