import pandas as pd
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.entities import (
    DatasetInput,
    LoggedModelInput,
    LoggedModelOutput,
    LoggedModelStatus,
    Run,
)

client = mlflow.MlflowClient()

# Read the wine-quality csv file from the URL
csv_url = (
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
)
data = pd.read_csv(csv_url, sep=";")

# Split the data into training and test sets. (0.75, 0.25) split.
X = data.drop(["quality"], axis=1)
y = data[["quality"]]
train_X, test_X, train_y, test_y = train_test_split(X, y)

train_dataset = mlflow.data.from_pandas(train_X.assign(quality=train_y), name="train_dataset")
test_dataset = mlflow.data.from_pandas(test_X.assign(quality=test_y), name="test_dataset")


with mlflow.start_run() as training_run:
    logged_model = client.create_logged_model(training_run.info.experiment_id, name="model")
    client.finalize_logged_model(logged_model.model_id, LoggedModelStatus.READY)

    mlflow.log_input(dataset=test_dataset, model=LoggedModelInput(logged_model.model_id))
    mlflow.log_outputs(models=[LoggedModelOutput(model_id=logged_model.model_id, step=0)])

    # Check that inputs and outputs were logged correctly
    active_run = client.get_run(training_run.info.run_id)
    assert active_run.inputs.dataset_inputs == [DatasetInput(test_dataset._to_mlflow_entity())]
    assert active_run.inputs.model_inputs == [LoggedModelInput(model_id=logged_model.model_id)]
    assert active_run.outputs.model_outputs == [
        LoggedModelOutput(model_id=logged_model.model_id, step=0)
    ]

    # Check that to/from proto conversion works as expected
    assert Run.from_proto(active_run.to_proto()).to_proto() == active_run.to_proto()
