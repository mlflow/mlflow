# LightGBM Example

This example trains a LightGBM classifier with the iris dataset and logs hyperparameters, metrics, and trained model.

## Running the code

```
python train.py --colsample-bytree 0.8 --subsample 0.9
```

## Running the code as a project

```
mlflow run . -P colsample-bytree=0.8 -P subsample=0.9
```
