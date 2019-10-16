## MLflow with TensorFlow 2.0.0

In this example, we use TensorFlow's premade estimator iris data example and add MLflow tracking to it.
In order to better understand the TensorFlow code, please see TensorFlow's [tutorial](https://www.tensorflow.org/tutorials/estimator/premade).

Lines related to MLflow:
* `mlflow.tensorflow.autolog()`
This is an [experimental api](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging-from-tensorflow-and-keras-experimental) that takes care of logging ML model artifacts and TensorBoard metrics created by the `tf.estimator` we are using.
The TensorBoard metrics are logged during training of the model.
The ML model artifact creation is handled during the call to `tf.estimator.export_saved_model()`.
* `mlflow.log_param()`
This is for when you want to manually log a parameter that autolog doesn't cover.
* `mlflow.log_metric()`
This is for when you want to manually log a parameter that autolog doesn't cover.
* `mlflow.pyfunc.load_model()`
This function loads back the model as a generic python function. You can predict on this with a pandas DataFrame input.


To run the example via MLflow, run the command

```
mlflow run .
```

This will run `train_predict_2.py` with the default parameters `--batch_size=100` and `--train_steps=1000`.

In order to run the file with custom parameters, run the command

```
mlflow run . -P batch_size=X -P train_steps=Y
```

where `X` and `Y` are your desired values for the parameters.



* If you have the required modules and would like to skip the creation of a conda environment, add the argument `--no-conda`.