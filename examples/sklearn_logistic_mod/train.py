import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlflow.models.signature import infer_signature

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    X = pd.DataFrame({"x": [-2, -1, 0, 1, 2, 1]}, dtype="int32")
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)

    class Model(mlflow.pyfunc.PythonModel):
        def __init__(self, m):
            self.m = m

        def predict(self, context, x):
            return self.m.predict(x)

    model = Model(lr)
    signature = infer_signature(X, model.predict(None, X))
    print("Signature")
    print(signature)
    mlflow.pyfunc.log_model("model", python_model=model, signature=signature)
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
