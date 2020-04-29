# XGBoost Example

This example trains an XGBoost classifier with the iris dataset and logs hyperparameters, metrics, and trained model.

## Running the code

```
python train.py --eta 0.2 --colsample-bytree 0.8 --subsample 0.9
```
After that try experiments with different parameters like:
```
python train.py --eta 0.4 --colsample-bytree 0.7 --subsample 0.8
```

Then you can open the mlflow ui to compare your logs and track the experiments by:
```
mflow ui
```


## Running the code as a project

```
mlflow run . -P colsample-bytree=0.8 -P subsample=0.9
```
