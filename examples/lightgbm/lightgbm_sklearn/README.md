# XGBoost Scikit-learn Model Example

This example trains an [`LightGBM.LGBMClassifier`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) with the diabetes dataset and logs hyperparameters, metrics, and trained model.

Like the other LightGBM example, we enable autologging for LightGBM scikit-learn models via `mlflow.lightgbm.autolog()`. Saving / loading models also supports LightGBM scikit-learn models.

You can run this example using the following command:

```shell
python train.py
```
