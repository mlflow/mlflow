from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import mlflow
from mlflow.models import infer_signature

X, y = load_iris(return_X_y=True, as_frame=True)
model = LogisticRegression().fit(X, y)
signature = infer_signature(X, model.predict(X))

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, name="model", signature=signature)
    runs_uri = f"runs:/{run.info.run_id}/model"
    model = mlflow.sklearn.load_model(runs_uri)
    print(model.predict(X)[:10])
