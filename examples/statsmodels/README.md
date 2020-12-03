# Statsmodels Example

This example trains a Statsmodels OLS (Ordinary Least Squares) model with synthetically generated data
and logs hyperparameters, metric (MSE), and trained model.

## Running the code

```
python train.py --inverse-method qr
```
The inverse method is the method used to compute the inverse matrix, and can be either qr or pinv (default).
'pinv' uses the Moore-Penrose pseudoinverse to solve the least squares problem. 'qr' uses the QR factorization.
You can try experimenting with both, as well as omitting the --inverse-method argument.

Then you can open the MLflow UI to track the experiments and compare your runs via:
```
mlflow ui
```

## Running the code as a project

```
mlflow run . -P inverse_method=qr

```
