import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

import mlflow
import mlflow.xgboost


def parse_args():
    parser = argparse.ArgumentParser(description='XGBoost example')
    parser.add_argument('--colsample-bytree', type=float, default=1.0,
                        help='subsample ratio of columns when constructing each tree (default: 1.0)')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample ratio of the training instances (default: 1.0)')
    parser.add_argument('--conda-env', type=str, default=None,
                        help='the path to a conda environment yaml file (default: None)')
    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()

    # prepare train and test data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # train model
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'colsample_bytree': args.colsample_bytree,
        'subsample': args.subsample,
        'seed': 42,
    }
    model = xgb.train(params, dtrain)

    # evaluate model
    y_proba = model.predict(dtest)
    y_pred = y_proba.argmax(axis=1)
    loss = log_loss(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    # log parameters, metrics, and model
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics({'log_loss': loss, 'accuracy': acc})
        mlflow.xgboost.log_model(model, 'model', args.conda_env)
        print('Model logged in run {}'.format(run.info.run_uuid))


if __name__ == '__main__':
    main()
