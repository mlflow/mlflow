from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.tracing.destination import Databricks, MlflowExperiment


def test_set_destination_mlflow_experiment():
    # With experiment_id
    destination = MlflowExperiment(experiment_id="123")
    assert destination.type == "experiment"
    assert destination.experiment_id == "123"
    assert destination.experiment_name is None
    assert destination.tracking_uri is None

    # With numeric experiment_id (should be converted to string)
    destination = MlflowExperiment(experiment_id=123)
    assert destination.experiment_id == "123"

    # With experiment_name
    destination = MlflowExperiment(experiment_name="Default")
    assert destination.experiment_name == "Default"
    assert destination.experiment_id == "0"

    # With both experiment_name and matching experiment_id
    destination = MlflowExperiment(experiment_name="Default", experiment_id="0")
    assert destination.experiment_id == "0"
    assert destination.experiment_name == "Default"

    # With tracking_uri
    destination = MlflowExperiment(experiment_id="123", tracking_uri="http://localhost:5000")
    assert destination.tracking_uri == "http://localhost:5000"

    # Should use active experiment when neither experiment_id nor name is provided
    destination = MlflowExperiment()
    assert destination.experiment_id is not None

    # With mismatched experiment_id and experiment_name
    with pytest.raises(
        MlflowException,
        match=r"experiment_id and experiment_name must refer to the same experiment",
    ):
        MlflowExperiment(experiment_name="Default", experiment_id="123")

    # Test when no experiment_id/name is provided and no active experiment
    with mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value=None):
        with pytest.raises(
            MlflowException,
            match=r"No experiment_id or experiment_name provided and no active experiment found",
        ):
            MlflowExperiment()


def test_set_destination_databricks():
    # With experiment_id
    destination = Databricks(experiment_id="123")
    assert destination.type == "databricks"
    assert destination.experiment_id == "123"

    # With numeric experiment_id (should be converted to string)
    destination = Databricks(experiment_id=123)
    assert destination.experiment_id == "123"

    # With experiment_name
    destination = Databricks(experiment_name="Default")
    assert destination.experiment_name == "Default"
    assert destination.experiment_id == "0"

    # With both experiment_name and matching experiment_id
    destination = Databricks(experiment_name="Default", experiment_id="0")
    assert destination.experiment_id == "0"
    assert destination.experiment_name == "Default"

    # With tracking_uri (inherited from MlflowExperiment)
    destination = Databricks(experiment_id="123", tracking_uri="http://localhost:5000")
    assert destination.tracking_uri == "http://localhost:5000"

    # Should use active experiment when neither experiment_id nor name is provided
    destination = Databricks()
    assert destination.experiment_id is not None

    # With mismatched experiment_id and experiment_name
    with pytest.raises(
        MlflowException,
        match=r"experiment_id and experiment_name must refer to the same experiment",
    ):
        Databricks(experiment_name="Default", experiment_id="123")

    # Test when no experiment_id/name is provided and no active experiment
    with mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value=None):
        with pytest.raises(
            MlflowException,
            match=r"No experiment_id or experiment_name provided and no active experiment found",
        ):
            Databricks()
