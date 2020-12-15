import numpy as np
import argparse
import mlflow
import statsmodels.api as sm
import mlflow.statsmodels
from sklearn.metrics import mean_squared_error


def parse_args():
    parser = argparse.ArgumentParser(description="Statsmodels example")
    parser.add_argument(
        "--inverse-method",
        type=str,
        default="pinv",
        help="Can be 'pinv', or 'qr'. 'pinv' uses the Moore-Penrose pseudoinverse "
        "to solve the least squares problem. 'qr' uses the QR factorization. "
        "(default: 'pinv')",
    )
    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()

    # prepare train and test data
    # Ordinary Least Squares (OLS)
    np.random.seed(9876789)
    nsamples = 100
    x = np.linspace(0, 10, 100)
    X = np.column_stack((x, x ** 2))
    beta = np.array([1, 0.1, 10])
    e = np.random.normal(size=nsamples)
    X = sm.add_constant(X)
    y = np.dot(X, beta) + e

    # enable auto logging
    mlflow.statsmodels.autolog()

    with mlflow.start_run():

        ols = sm.OLS(y, X)
        model = ols.fit(method=args.inverse_method)

        # evaluate model
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)

        # log metrics
        mlflow.log_metrics({"mse": mse})


if __name__ == "__main__":
    main()
