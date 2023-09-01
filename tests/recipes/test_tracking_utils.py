import os
import pathlib
from unittest import mock

import pytest
import yaml

import mlflow
from mlflow.recipes.utils import get_recipe_config
from mlflow.recipes.utils.tracking import get_recipe_tracking_config, log_code_snapshot
from mlflow.utils.file_utils import path_to_local_file_uri, path_to_local_sqlite_uri

from tests.recipes.helper_functions import list_all_artifacts


@pytest.mark.usefixtures("enter_test_recipe_directory")
@pytest.mark.parametrize(
    ("tracking_uri", "artifact_location", "experiment_name", "experiment_id", "run_name_prefix"),
    [
        (
            "mysql://myhost:8000/test_uri",
            "test/artifact/location",
            "myexpname",
            "myexpid",
            "testrun",
        ),
        ("mysql://myhost:8000/test_uri", "test/artifact/location", None, "myexpid", "testrun"),
        ("mysql://myhost:8000/test_uri", "test/artifact/location", "myexpname", None, "testrun"),
        (None, "test/artifact/location", "myexpname", "myexpid", "testrun"),
        ("mysql://myhost:8000/test_uri", None, "myexpname", "myexpid", "testrun"),
        ("mysql://myhost:8000/test_uri", "test/artifact/location", "myexpname", "myexpid", None),
        (None, None, None, "myexpid", None),
    ],
)
def test_get_recipe_tracking_config_returns_expected_config(
    tracking_uri, artifact_location, experiment_name, experiment_id, run_name_prefix
):
    default_tracking_uri = path_to_local_sqlite_uri(
        path=str((pathlib.Path.cwd() / "metadata" / "mlflow" / "mlruns.db").resolve())
    )
    default_artifact_location = path_to_local_file_uri(
        path=str((pathlib.Path.cwd() / "metadata" / "mlflow" / "mlartifacts").resolve())
    )
    default_experiment_name = "sklearn_regression"  # equivalent to recipe name

    profile_contents = {
        "experiment": {},
        "INGEST_CONFIG": None,
        "INGEST_SCORING_CONFIG": None,
        "PREDICT_OUTPUT_CONFIG": None,
    }
    if tracking_uri is not None:
        profile_contents["experiment"]["tracking_uri"] = tracking_uri
    if artifact_location is not None:
        profile_contents["experiment"]["artifact_location"] = artifact_location
    if experiment_name is not None:
        profile_contents["experiment"]["name"] = experiment_name
    if experiment_id is not None:
        profile_contents["experiment"]["id"] = experiment_id
    if run_name_prefix is not None:
        profile_contents["experiment"]["run_name_prefix"] = run_name_prefix

    profile_path = pathlib.Path.cwd() / "profiles" / "testprofile.yaml"
    with open(profile_path, "w") as f:
        yaml.safe_dump(profile_contents, f)

    recipe_config = get_recipe_config(recipe_root_path=os.getcwd(), profile="testprofile")
    recipe_tracking_config = get_recipe_tracking_config(
        recipe_root_path=os.getcwd(), recipe_config=recipe_config
    )
    assert recipe_tracking_config.tracking_uri == (tracking_uri or default_tracking_uri)
    assert recipe_tracking_config.artifact_location == (
        artifact_location or default_artifact_location
    )
    if run_name_prefix is None:
        assert recipe_tracking_config.run_name is None
    else:
        assert run_name_prefix in recipe_tracking_config.run_name
    if experiment_name is not None:
        assert recipe_tracking_config.experiment_name == experiment_name
    elif experiment_id is not None:
        assert recipe_tracking_config.experiment_id == experiment_id
    elif experiment_id is None and experiment_name is None:
        assert recipe_tracking_config.experiment_name == default_experiment_name


@pytest.mark.usefixtures("enter_test_recipe_directory")
@pytest.mark.parametrize(
    ("tracking_uri", "artifact_location", "experiment_name", "experiment_id", "run_name_prefix"),
    [
        (
            "mysql://myhost:8000/test_uri",
            "test/artifact/location",
            "myexpname",
            "myexpid",
            "testrun",
        ),
        ("mysql://myhost:8000/test_uri", "test/artifact/location", None, "myexpid", "testrun"),
        ("mysql://myhost:8000/test_uri", "test/artifact/location", "myexpname", None, "testrun"),
        (None, "test/artifact/location", "myexpname", "myexpid", "testrun"),
        ("mysql://myhost:8000/test_uri", None, "myexpname", "myexpid", "testrun"),
        ("mysql://myhost:8000/test_uri", "test/artifact/location", "myexpname", "myexpid", None),
        (None, None, None, "myexpid", None),
    ],
)
def test_get_recipe_tracking_config_returns_expected_config_on_databricks(
    tracking_uri, artifact_location, experiment_name, experiment_id, run_name_prefix
):
    with mock.patch("mlflow.recipes.utils.tracking.is_in_databricks_runtime", return_value=True):
        default_tracking_uri = "databricks"
        default_experiment_name = "sklearn_regression"  # equivalent to recipe name

        profile_contents = {
            "experiment": {},
            "INGEST_CONFIG": None,
            "INGEST_SCORING_CONFIG": None,
            "PREDICT_OUTPUT_CONFIG": None,
        }
        if tracking_uri is not None:
            profile_contents["experiment"]["tracking_uri"] = tracking_uri
        if artifact_location is not None:
            profile_contents["experiment"]["artifact_location"] = artifact_location
        if experiment_name is not None:
            profile_contents["experiment"]["name"] = experiment_name
        if experiment_id is not None:
            profile_contents["experiment"]["id"] = experiment_id
        if run_name_prefix is not None:
            profile_contents["experiment"]["run_name_prefix"] = run_name_prefix

        profile_path = pathlib.Path.cwd() / "profiles" / "testprofile.yaml"
        with open(profile_path, "w") as f:
            yaml.safe_dump(profile_contents, f)

        recipe_config = get_recipe_config(recipe_root_path=os.getcwd(), profile="testprofile")
        recipe_tracking_config = get_recipe_tracking_config(
            recipe_root_path=os.getcwd(), recipe_config=recipe_config
        )
        assert recipe_tracking_config.tracking_uri == (tracking_uri or default_tracking_uri)
        assert recipe_tracking_config.artifact_location == artifact_location
        if run_name_prefix is None:
            assert recipe_tracking_config.run_name is None
        else:
            assert run_name_prefix in recipe_tracking_config.run_name
        if experiment_name is not None:
            assert recipe_tracking_config.experiment_name == experiment_name
        elif experiment_id is not None:
            assert recipe_tracking_config.experiment_id == experiment_id
        elif experiment_id is None and experiment_name is None:
            assert recipe_tracking_config.experiment_name == default_experiment_name


def test_log_code_snapshot(tmp_path: pathlib.Path):
    files = [
        "recipe.yaml",
        "requirements.txt",
        "profiles/local.yaml",
        "steps/train.py",
        "runtime/recipe.yaml",
    ]
    for f in files:
        path = tmp_path.joinpath(f)
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text("")

    mlflow.set_experiment(experiment_id="0")
    recipe_config = {"name": "fake_config", "dict": {"key": 123}}
    with mlflow.start_run() as run:
        log_code_snapshot(tmp_path, run.info.run_id, recipe_config=recipe_config)
        tracking_uri = mlflow.get_tracking_uri()

    artifacts = set(list_all_artifacts(tracking_uri, run.info.run_id))
    assert artifacts.issuperset(f"recipe_snapshot/{f}" for f in files)
