import pytest

from mlflow.exceptions import MlflowException
from mlflow.tracing.destination import Databricks


def test_set_destination_databricks():
    destination = Databricks(experiment_id="123")
    assert destination.type == "databricks"
    assert destination.experiment_id == "123"

    destination = Databricks(experiment_id=123)
    assert destination.experiment_id == "123"

    destination = Databricks(experiment_name="Default")
    assert destination.experiment_name == "Default"
    assert destination.experiment_id == "0"

    destination = Databricks(experiment_name="Default", experiment_id="0")
    assert destination.experiment_id == "0"

    with pytest.raises(MlflowException, match=r"experiment_id and experiment_name must"):
        Databricks(experiment_name="Default", experiment_id="123")
