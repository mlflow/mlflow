# Sample Decision Tree Classifier 

from __future__ import print_function
import sys
from sklearn import datasets, metrics
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn
from mlflow import version

def train(min_samples_leaf, max_depth, dataset):
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("max_depth", max_depth)

    clf = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    print("Classifier:",clf)
    clf.fit(dataset.data, dataset.target)
    expected = dataset.target
    predicted = clf.predict(dataset.data)

    mlflow.sklearn.log_model(clf, "model") 

    write_artifact('confusion_matrix.txt',str(metrics.confusion_matrix(expected, predicted)))
    write_artifact('classification_report.txt',metrics.classification_report(expected, predicted))

    auc = metrics.auc(expected, predicted)
    accuracy_score = metrics.accuracy_score(expected, predicted)
    zero_one_loss = metrics.zero_one_loss(expected, predicted)

    mlflow.log_metric("auc", auc)
    mlflow.log_metric("accuracy_score", accuracy_score)
    mlflow.log_metric("zero_one_loss", zero_one_loss)

    print("Params:  min_samples_leaf={} max_depth={}".format(min_samples_leaf,max_depth))
    print("Metrics: auc={} accuracy_score={} zero_one_loss={}".format(auc,accuracy_score,zero_one_loss))

def write_artifact(file, data):
    with open(file, 'w') as f:
        f.write(data)
    mlflow.log_artifact(file)

if __name__ == "__main__":
    print("MLflow Version:", version.VERSION)
    print("MLflow Tracking URI:", mlflow.get_tracking_uri())
    print("MLflow Artifact URI:", mlflow.get_artifact_uri())
    min_samples_leaf = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    train(min_samples_leaf, max_depth, datasets.load_iris())
