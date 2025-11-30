import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

import mlflow
import mlflow.h2o

h2o.init()

wine = h2o.import_file(path="wine-quality.csv")
r = wine["quality"].runif()
train = wine[r < 0.7]
test = wine[0.3 <= r]


def train_random_forest(ntrees):
    with mlflow.start_run():
        rf = H2ORandomForestEstimator(ntrees=ntrees)
        train_cols = [n for n in wine.col_names if n != "quality"]
        rf.train(train_cols, "quality", training_frame=train, validation_frame=test)

        mlflow.log_param("ntrees", ntrees)

        mlflow.log_metric("rmse", rf.rmse())
        mlflow.log_metric("r2", rf.r2())
        mlflow.log_metric("mae", rf.mae())

        mlflow.h2o.log_model(rf, name="model")


if __name__ == "__main__":
    for ntrees in [10, 20, 50, 100, 200]:
        train_random_forest(ntrees)
