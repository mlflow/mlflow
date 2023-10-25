# Sktime Example

This example trains a `Sktime` NaiveForecaster model using the Longley dataset for
forecasting with exogenous variables. It shows a custom model type implementation
that logs the training hyper-parameters, evaluation metrics and the trained model
as an artifact.

## Running the code

Run the `train.py` module to create a new MLflow experiment and to
compute interval forecasts loading the trained model in native `sktime`
flavor and `pyfunc` flavor:

```
python train.py
```

To view the newly created experiment and logged artifacts open the MLflow UI:

```
mlflow ui
```

## Model serving

This section illustrates an example of serving the `pyfunc` flavor to a local REST
API endpoint and subsequently requesting a prediction from the served model. To serve the model run the command below where you substitute the run id printed during execution of the `train.py` module:

```
mlflow models serve -m runs:/<run_id>/model --env-manager local --host 127.0.0.1

```

Open a new terminal and run the `score_model.py` module to request a prediction from the served model (for more details read the [MLflow deployment API reference](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models)):

```
python score_model.py
```

## Running the code as a project

You can also run the code as a project as follows:

```
mlflow run .

```

## Running unit tests

The `test_sktime_model_export.py` module includes a number of tests that can be
executed as follows:

```
pytest test_sktime_model_export.py

```

While these tests will depend on the specifics of each individual flavor and in particular the design of the model wrapper interface (e.g. `_SktimeModelWrapper`), the above module can provide some orientation
for the type of tests that can be useful when creating a new custom model flavor.
