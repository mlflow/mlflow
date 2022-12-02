# Examples for LightGBM Autologging

LightGBM autologging functionalities are demonstrated through two examples. The first example in the `lightgbm_native` folder logs a Booster model trained by `lightgbm.train()`. The second example in the `lightgbm_sklearn` folder shows how autologging works for LightGBM scikit-learn models. The autologging for all LightGBM models is enabled via `mlflow.lightgbm.autolog()`.
