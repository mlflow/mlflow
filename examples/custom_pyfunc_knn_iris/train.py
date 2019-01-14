"""
train_diabetes.py

  Trains and saves an MLflow model with a custom inference function.

  First, this example fits an ElasticNet regression model on the Wine Quality dataset from Cortez 
  et al. (http://archive.ics.uci.edu/ml/datasets/Wine+Quality). 
  
  Then, it saves a custom MLflow model with the "python_function" flavor that transforms the 
  regression score into a binary classification. 
  
  Finally, it evaluates the MLflow model against several sample inputs.

 Usage:
   python train.py 0.01 0.01
   python train.py 0.01 0.75
   python train.py 0.01 1.0
"""


import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.pyfunc
import mlflow.sklearn


class WineClassifier(mlflow.pyfunc.PythonModel):

    def __init__(self, quality_threshold):
        self.quality_threshold = quality_threshold

    def load_context(self, context):
        self.model = mlflow.sklearn.load_model(path=context.artifacts["sk_model"])

    def predict(self, context, model_input):
        wine_scores = self.model.predict(model_input)
        return wine_scores


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Train an ElasticNet regression model on the Iris dataset
    # in Scikit-learn
    lr_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr_model.fit(train, train_y)

    # Log the ElasticNet regressor as an MLflow model in a new run
    sklearn_artifact_path = "sk_model_artifact"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=lr_model, artifact_path=sklearn_artifact_path)
        sk_model_artifact_uri = mlflow.get_artifact_uri(sklearn_artifact_path)

    pyfunc_artifact_path = "pyfunc_model_artifact"
    with mlflow.start_run():
        my_model = WineClassifier(quality_threshold=5)
        mlflow.pyfunc.log_model(artifact_path=pyfunc_artifact_path,
                                artifacts={
                                    "sk_model": sk_model_artifact_uri
                                },
                                python_model=my_model)
        pyfunc_run_id = mlflow.active_run().info.run_uuid

    pyfunc_model = mlflow.pyfunc.load_pyfunc(pyfunc_artifact_path, pyfunc_run_id)
    print("PREDS", pyfunc_model.predict(test_x), type(pyfunc_model.predict(test_x)))
