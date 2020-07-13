## MLflow with TensorFlow 1.X

In this example, we train a [TensorFlow estimator](https://www.tensorflow.org/guide/estimator) to predict house prices using the TensorFlow [Boston Housing dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html).
The code is mostly pure TensorFlow - we add a call to `mlflow.tensorflow.autolog()` before training to record params & metrics from model training (e.g. model loss) as part of an MLflow run. After training, `mlflow.tensorflow.autolog()` links the model with the same MLflow run when `export_saved_model()` is called, allowing us to associate the model with its training metrics & params. We then demonstrate how to load the saved model back as a generic `mlflow.pyfunc`, allowing us to make predictions on pandas DataFrames.

### Code related to MLflow:
* [`mlflow.tensorflow.autolog()`](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging-from-tensorflow-and-keras-experimental):
This is an experimental api that logs ML model artifacts and TensorBoard metrics created by the `tf.estimator` we are using.
The TensorBoard metrics are logged during training of the model. By default, MLflow autologs every 100 steps.
The ML model artifact creation is handled during the call to `tf.estimator.export_saved_model()`.

* [`mlflow.pyfunc.load_model()`](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model):
This function loads back the model as a generic python function. You can predict on this with a pandas DataFrame input.

### Running the code

Note: A minimum MLflow version of 1.6 is required to run this example.

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


