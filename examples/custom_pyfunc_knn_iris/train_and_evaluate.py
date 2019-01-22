"""
train_diabetes.py

  Trains and saves an MLflow model with a custom inference function.

  First, this example fits an ElasticNet regression model on the Wine Quality dataset from Cortez 
  et al. (http://archive.ics.uci.edu/ml/datasets/Wine+Quality). 
  
  Then, it saves a custom MLflow model with the "python_function" flavor that transforms the 
  wine rating (ElasticNet regression score) into a binary classification ('good' or 'bad').
  
  Finally, it evaluates the MLflow model against several sample inputs.

 Usage:
   python train.py -a 0.01 -l 0.01
   python train.py -a 0.01 -l 0.75 -q 5.7
   python train.py -a 0.06 -q 8.0
"""

import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.pyfunc
import mlflow.sklearn


class WineClassifier(mlflow.pyfunc.PythonModel):

    def __init__(self, quality_threshold):
        self.classification_mapper = np.vectorize(
            lambda wine_quality : "good" if wine_quality >= quality_threshold else "bad")
        
    def load_context(self, context):
        self.sk_elasticnet = mlflow.sklearn.load_model(
            path=context.artifacts["sk_elasticnet"])

    def predict(self, context, model_input):
        wine_scores = self.sk_elasticnet.predict(model_input)
        return np.vstack([wine_scores, self.classification_mapper(wine_scores)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train and evaluate a custom MLflow pyfunc model for classifying wines')
    parser.add_argument('--quality-threshold', '-q', type=float, default=6.0, help=(
        "The minimum quality rating required for the model to rate a wine as 'good'"))
    parser.add_argument('--alpha', '-a', type=float, default=0.5, help=(
        "The value of the 'alpha' parameter to use when training the ElasticNet regression model"
        " for rating wines."))
    parser.add_argument('--l1-ratio', '-l', type=float, default=0.5, help=(
        "The value of the 'l1 ratio' parameter to use when training the ElasticNet regression model"
        " for rating wines."))
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1).sample(frac=1)
    train_y = train[["quality"]]

    print("Training ElasticNet regression model on the wine dataset with alpha={alpha},"
          " l1_ratio={l1_ratio}, and quality_threshold={quality_threshold}\n".format(
              alpha=args.alpha, l1_ratio=args.l1_ratio, quality_threshold=args.quality_threshold))

    # Train an ElasticNet regression model on the wine dataset
    # in Scikit-learn
    sk_elasticnet = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
    sk_elasticnet.fit(train_x, train_y)

    # Log the ElasticNet regressor as an MLflow model in a new run
    sk_elasticnet_artifact_path = "sk_elasticnet"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sk_elasticnet, artifact_path=sk_elasticnet_artifact_path)
        sk_elasticnet_artifact_uri = mlflow.get_artifact_uri(sk_elasticnet_artifact_path)

    # Construct a WineClassifier object, which extends `mlflow.pyfunc.PythonModel`, that will
    # interpret the wine rating output by the ElasticNet model to classify the wine as either
    # 'good' or 'bad'
    my_model = WineClassifier(quality_threshold=6)

    print("Logging custom pyfunc model for classifying wines...\n")

    # Save the WineClassifier and the ElasticNet model that it depends on as a new MLflow
    # model with the pyfunc flavor
    pyfunc_artifact_path = "pyfunc_model_artifact"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(artifact_path=pyfunc_artifact_path,
                                artifacts={
                                    "sk_elasticnet": sk_elasticnet_artifact_uri, 
                                },
                                python_model=my_model)
        pyfunc_run_id = mlflow.active_run().info.run_uuid

    print("Evaluating custom pyfunc wine classifier...\n")

    # Load the new MLflow pyfunc model and it evaluate it on some sample inputs
    pyfunc_model = mlflow.pyfunc.load_pyfunc(pyfunc_artifact_path, pyfunc_run_id)
    for i in range(10):
        model_input = test_x.iloc[i].reshape(1,-1)
        print("Evaluating wine with attributes: '{attrib}' ...".format(
            attrib=model_input.flatten()))
        score, classification = pyfunc_model.predict(model_input)
        print("Score: {score}, Classification: \"{classification}\"\n".format(
            score=score[0], classification=classification[0]))
