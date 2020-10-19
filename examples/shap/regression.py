import os

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

import mlflow
from utils import to_pandas_Xy, show_image


# prepare training data
X, y = to_pandas_Xy(load_boston())
X = X.iloc[:50, :8]
y = y.iloc[:50]

# train a model
model = LinearRegression()
model.fit(X, y)

# log an explanation
with mlflow.start_run() as run:
    mlflow.shap.log_explanation(model.predict, X)

# list artifacts
client = mlflow.tracking.MlflowClient()
artifact_path = "model_explanations_shap"
artifacts = [x.path for x in client.list_artifacts(run.info.run_id, artifact_path)]
print("# artifacts:")
print(artifacts)

# load back the logged explanation
dst_path = client.download_artifacts(run.info.run_id, artifact_path)
base_values = np.load(os.path.join(dst_path, "base_values.npy"))
shap_values = np.load(os.path.join(dst_path, "shap_values.npy"))

print("\n# base_values")
print(base_values)
print("\n# shap_values")
print(shap_values[:3])

# show the summary bar plot
show_image(os.path.join(dst_path, "summary_bar_plot.png"))
