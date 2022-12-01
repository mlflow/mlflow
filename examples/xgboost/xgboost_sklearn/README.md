# XGBoost Scikit-learn Model Example

This example trains an [`XGBoost.XGBRegressor`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor) with the diabetes dataset and logs hyperparameters, metrics, and trained model.

Like the other XGBoost example, we enable autologging for XGBoost scikit-learn models via `mlflow.xgboost.autolog()`. Saving / loading models also supports XGBoost scikit-learn models.

You can run this example using the following command:

```
python train.py
```
