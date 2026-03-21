# Examples for XGBoost Autologging

Two examples are provided to demonstrate XGBoost autologging functionalities. The `xgboost_native` folder contains an example that logs a Booster model trained by `xgboost.train()`. The `xgboost_sklearn` includes another example showing how autologging works for XGBoost scikit-learn models. In fact, there is no difference in turning on autologging for all XGBoost models. That is, `mlflow.xgboost.autolog()` works for all XGBoost models.
