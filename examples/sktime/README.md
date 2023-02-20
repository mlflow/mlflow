# Sktime Example

This example trains a Sktime NaiveForecaster model using the Longley dataset for
forecasting with exogenous variables and logs hyperparameters, metrics, and
trained model.

## Running the code

```
python train.py
```

Then you can open the MLflow UI to track the experiments and compare your runs via:

```
mlflow ui
```

## Running the code as a project

```
mlflow run .

```

## Running unit tests

```
pytest test_sktime_model_export.py

```
