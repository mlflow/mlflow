import os
import pathlib
import yaml
from unittest import mock

import pytest

import mlflow
from mlflow.pipelines.utils import get_pipeline_config
from mlflow.pipelines.utils.tracking import get_pipeline_tracking_config, log_code_snapshot
from mlflow.utils.file_utils import path_to_local_file_uri, path_to_local_sqlite_uri

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    enter_pipeline_example_directory,
    enter_test_pipeline_directory,
    list_all_artifacts,
)  # pylint: enable=unused-import


@pytest.mark.usefixtures("enter_test_pipeline_directory")
@pytest.mark.parametrize(
    "tracking_uri,artifact_location,experiment_name,experiment_id",
    [
        ("mysql://myhost:8000/test_uri", "test/artifact/location", "myexpname", "myexpid"),
        ("mysql://myhost:8000/test_uri", "test/artifact/location", None, "myexpid"),
        ("mysql://myhost:8000/test_uri", "test/artifact/location", "myexpname", None),
        (None, "test/artifact/location", "myexpname", "myexpid"),
        ("mysql://myhost:8000/test_uri", None, "myexpname", "myexpid"),
        (None, None, None, "myexpid"),
    ],
)
def test_get_pipeline_tracking_config_returns_expected_config(
    tracking_uri, artifact_location, experiment_name, experiment_id
):
    default_tracking_uri = path_to_local_sqlite_uri(
        path=str((pathlib.Path.cwd() / "metadata" / "mlflow" / "mlruns.db").resolve())
    )
    default_artifact_location = path_to_local_file_uri(
        path=str((pathlib.Path.cwd() / "metadata" / "mlflow" / "mlartifacts").resolve())
    )
    default_experiment_name = "sklearn_regression"  # equivalent to pipeline name

    profile_contents = {
        "experiment": {},
        "INGEST_DATA_LOCATION": None,
        "INGEST_SCORING_DATA_LOCATION": None,
        "SCORED_OUTPUT_DATA_LOCATION": None,
    }
    if tracking_uri is not None:
        profile_contents["experiment"]["tracking_uri"] = tracking_uri
    if artifact_location is not None:
        profile_contents["experiment"]["artifact_location"] = artifact_location
    if experiment_name is not None:
        profile_contents["experiment"]["name"] = experiment_name
    if experiment_id is not None:
        profile_contents["experiment"]["id"] = experiment_id

    profile_path = pathlib.Path.cwd() / "profiles" / "testprofile.yaml"
    with open(profile_path, "w") as f:
        yaml.safe_dump(profile_contents, f)

    pipeline_config = get_pipeline_config(pipeline_root_path=os.getcwd(), profile="testprofile")
    pipeline_tracking_config = get_pipeline_tracking_config(
        pipeline_root_path=os.getcwd(), pipeline_config=pipeline_config
    )
    assert pipeline_tracking_config.tracking_uri == (tracking_uri or default_tracking_uri)
    assert pipeline_tracking_config.artifact_location == (
        artifact_location or default_artifact_location
    )
    if experiment_name is not None:
        assert pipeline_tracking_config.experiment_name == experiment_name
    elif experiment_id is not None:
        assert pipeline_tracking_config.experiment_id == experiment_id
    elif experiment_id is None and experiment_name is None:
        assert pipeline_tracking_config.experiment_name == default_experiment_name


@pytest.mark.usefixtures("enter_test_pipeline_directory")
@pytest.mark.parametrize(
    "tracking_uri,artifact_location,experiment_name,experiment_id",
    [
        ("mysql://myhost:8000/test_uri", "test/artifact/location", "myexpname", "myexpid"),
        ("mysql://myhost:8000/test_uri", "test/artifact/location", None, "myexpid"),
        ("mysql://myhost:8000/test_uri", "test/artifact/location", "myexpname", None),
        (None, "test/artifact/location", "myexpname", "myexpid"),
        ("mysql://myhost:8000/test_uri", None, "myexpname", "myexpid"),
        (None, None, None, "myexpid"),
    ],
)
def test_get_pipeline_tracking_config_returns_expected_config_on_databricks(
    tracking_uri, artifact_location, experiment_name, experiment_id
):
    with mock.patch("mlflow.pipelines.utils.tracking.is_in_databricks_runtime", return_value=True):
        default_tracking_uri = "databricks"
        default_experiment_name = "sklearn_regression"  # equivalent to pipeline name

        profile_contents = {
            "experiment": {},
            "INGEST_DATA_LOCATION": None,
            "INGEST_SCORING_DATA_LOCATION": None,
            "SCORED_OUTPUT_DATA_LOCATION": None,
        }
        if tracking_uri is not None:
            profile_contents["experiment"]["tracking_uri"] = tracking_uri
        if artifact_location is not None:
            profile_contents["experiment"]["artifact_location"] = artifact_location
        if experiment_name is not None:
            profile_contents["experiment"]["name"] = experiment_name
        if experiment_id is not None:
            profile_contents["experiment"]["id"] = experiment_id

        profile_path = pathlib.Path.cwd() / "profiles" / "testprofile.yaml"
        with open(profile_path, "w") as f:
            yaml.safe_dump(profile_contents, f)

        pipeline_config = get_pipeline_config(pipeline_root_path=os.getcwd(), profile="testprofile")
        pipeline_tracking_config = get_pipeline_tracking_config(
            pipeline_root_path=os.getcwd(), pipeline_config=pipeline_config
        )
        assert pipeline_tracking_config.tracking_uri == (tracking_uri or default_tracking_uri)
        assert pipeline_tracking_config.artifact_location == artifact_location
        if experiment_name is not None:
            assert pipeline_tracking_config.experiment_name == experiment_name
        elif experiment_id is not None:
            assert pipeline_tracking_config.experiment_id == experiment_id
        elif experiment_id is None and experiment_name is None:
            assert pipeline_tracking_config.experiment_name == default_experiment_name


def test_log_code_snapshot(tmp_path: pathlib.Path):
    files = [
        "pipeline.yaml",
        "requirements.txt",
        "profiles/local.yaml",
        "steps/train.py",
        "runtime/pipeline.yaml",
    ]
    for f in files:
        path = tmp_path.joinpath(f)
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text("")

    mlflow.set_experiment(experiment_id="0")
    pipeline_config = {"name": "fake_config", "dict": {"key": 123}}
    with mlflow.start_run() as run:
        log_code_snapshot(tmp_path, run.info.run_id, pipeline_config=pipeline_config)
        tracking_uri = mlflow.get_tracking_uri()

    artifacts = set(list_all_artifacts(tracking_uri, run.info.run_id))
    assert artifacts.issuperset(f"pipeline_snapshot/{f}" for f in files)
