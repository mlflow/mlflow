import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import mlflow

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
clf.fit(X, y)

with mlflow.start_run():
    mlflow.sklearn.log_model(clf, artifact_path="model")
