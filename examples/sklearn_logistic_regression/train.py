import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

if __name__ == "__main__":
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    predictions = lr.predict(X)
    signature = infer_signature(X, predictions)
    mlflow.sklearn.log_model(lr, "model", signature=signature)
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
