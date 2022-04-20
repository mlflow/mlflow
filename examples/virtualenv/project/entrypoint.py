import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mlflow

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
clf.fit(X, y)

with mlflow.start_run():
    mlflow.sklearn.log_model(clf, artifact_path="model")
