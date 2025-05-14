import os

import numpy as np
import shap
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

# prepare training data
X, y = load_iris(return_X_y=True, as_frame=True)


# train a model
model = RandomForestClassifier()
model.fit(X, y)

# log an explanation
with mlflow.start_run() as run:
    mlflow.shap.log_explanation(model.predict_proba, X)

# list artifacts
client = MlflowClient()
artifact_path = "model_explanations_shap"
artifacts = [x.path for x in client.list_artifacts(run.info.run_id, artifact_path)]
print("# artifacts:")
print(artifacts)

# load back the logged explanation
dst_path = download_artifacts(run_id=run.info.run_id, artifact_path=artifact_path)
base_values = np.load(os.path.join(dst_path, "base_values.npy"))
shap_values = np.load(os.path.join(dst_path, "shap_values.npy"))

# show a force plot
shap.force_plot(base_values[0], shap_values[0, :, 0], X.iloc[0, :], matplotlib=True)
