import pandas as pd
import requests

from sktime.datasets import load_longley
from sktime.forecasting.model_selection import temporal_train_test_split

y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

# Define local host and endpoint url
host = "127.0.0.1"
url = f"http://{host}:5000/invocations"

# Model scoring via REST API requires transforming the configuration DataFrame
# into JSON format. As numpy ndarray type is not JSON serializable we need to
# convert the exogenous regressor into a list. The wrapper instance will convert
# the list back to ndarray type as required by sktime predict methods. For more
# details read the MLflow deployment API reference.
# (https://mlflow.org/docs/latest/models.html#deploy-mlflow-models)
X_test_list = X_test.to_numpy().tolist()
predict_conf = pd.DataFrame(
    [
        {
            "fh": [1, 2, 3, 4],
            "predict_method": "predict_interval",
            "coverage": [0.9, 0.95],
            "X": X_test_list,
        }
    ]
)

# Create dictionary with pandas DataFrame in the split orientation
json_data = {"dataframe_split": predict_conf.to_dict(orient="split")}

# Score model
response = requests.post(url, json=json_data)
print(f"\nPyfunc 'predict_interval':\n${response.json()}")
