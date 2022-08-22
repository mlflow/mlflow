import os
import pathlib
from distutils.dir_util import copy_tree

import pandas as pd
import pytest
import yaml
from unittest import mock

import mlflow
from mlflow.entities import Run, SourceType
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.pipelines.pipeline import Pipeline
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path, _get_execution_directory_basename
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.mlflow_tags import (
    MLFLOW_SOURCE_NAME,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
    LEGACY_MLFLOW_GIT_REPO_URL,
    MLFLOW_SOURCE_TYPE,
)

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    enter_pipeline_example_directory,
    enter_test_pipeline_directory,
    list_all_artifacts,
    chdir,
)  # pylint: enable=unused-import

_STEP_NAMES = ["ingest", "split", "train", "transform", "evaluate"]


@pytest.mark.usefixtures("enter_pipeline_example_directory")
def test_create_pipeline_fails_with_invalid_profile():
    with pytest.raises(
        MlflowException,
        match=r"(Failed to find|Did not find the YAML configuration)",
    ):
        Pipeline(profile="local123")


@pytest.mark.usefixtures("enter_pipeline_example_directory")
def test_create_pipeline_and_clean_works():
    p = Pipeline(profile="local")
    p.clean()


@pytest.mark.usefixtures("enter_pipeline_example_directory")
@pytest.mark.parametrize("empty_profile", [None, ""])
def test_create_pipeline_fails_with_empty_profile_name(empty_profile):
    with pytest.raises(MlflowException, match="A profile name must be provided"):
        _ = Pipeline(profile=empty_profile)


@pytest.mark.usefixtures("enter_pipeline_example_directory")
def test_create_pipeline_fails_with_path_containing_space(tmp_path):
    space_parent = tmp_path / "space parent"
    space_path = space_parent / "child"
    os.makedirs(space_parent, exist_ok=True)
    os.makedirs(space_path, exist_ok=True)
    copy_tree(os.getcwd(), str(space_path))

    with chdir(space_path), pytest.raises(
        MlflowException, match="Pipeline directory path cannot contain spaces"
    ):
        Pipeline(profile="local")


@pytest.mark.usefixtures("enter_pipeline_example_directory")
@pytest.mark.parametrize("custom_execution_directory", [None, "custom"])
def test_pipelines_execution_directory_is_managed_as_expected(
    custom_execution_directory, enter_pipeline_example_directory, tmp_path
):
    if custom_execution_directory is not None:
        custom_execution_directory = tmp_path / custom_execution_directory

    if custom_execution_directory is not None:
        os.environ["MLFLOW_PIPELINES_EXECUTION_DIRECTORY"] = str(custom_execution_directory)

    expected_execution_directory_location = (
        pathlib.Path(custom_execution_directory)
        if custom_execution_directory
        else pathlib.Path.home()
        / ".mlflow"
        / "pipelines"
        / _get_execution_directory_basename(enter_pipeline_example_directory)
    )

    # Run the full pipeline and verify that outputs for each step were written to the expected
    # execution directory locations
    p = Pipeline(profile="local")
    p.run()
    assert (expected_execution_directory_location / "Makefile").exists()
    assert (expected_execution_directory_location / "steps").exists()
    for step_name in _STEP_NAMES:
        step_outputs_path = expected_execution_directory_location / "steps" / step_name / "outputs"
        assert step_outputs_path.exists()
        first_output = next(step_outputs_path.iterdir(), None)
        # TODO: Assert that the ingest step has outputs once ingest execution has been implemented
        assert first_output is not None or step_name == "ingest"

    # Clean the pipeline and verify that all step outputs have been removed
    p.clean()
    for step_name in _STEP_NAMES:
        step_outputs_path = expected_execution_directory_location / "steps" / step_name / "outputs"
        assert not list(step_outputs_path.iterdir())


@pytest.mark.usefixtures("enter_test_pipeline_directory")
def test_pipelines_log_to_expected_mlflow_backend_with_expected_run_tags_once_on_reruns(
    tmp_path,
):
    experiment_name = "my_test_exp"
    tracking_uri = "sqlite:///" + str((tmp_path / "tracking_dst.db").resolve())
    artifact_location = str((tmp_path / "mlartifacts").resolve())

    profile_path = pathlib.Path.cwd() / "profiles" / "local.yaml"
    with open(profile_path, "r") as f:
        profile_contents = yaml.safe_load(f)

    profile_contents["experiment"]["name"] = experiment_name
    profile_contents["experiment"]["tracking_uri"] = tracking_uri
    profile_contents["experiment"]["artifact_location"] = path_to_local_file_uri(artifact_location)

    with open(profile_path, "w") as f:
        yaml.safe_dump(profile_contents, f)

    mlflow.set_tracking_uri(tracking_uri)
    pipeline = Pipeline(profile="local")
    pipeline.clean()
    pipeline.run()

    logged_runs = mlflow.search_runs(experiment_names=[experiment_name], output_format="list")
    assert len(logged_runs) == 1
    logged_run = logged_runs[0]
    assert logged_run.info.artifact_uri == path_to_local_file_uri(
        str((pathlib.Path(artifact_location) / logged_run.info.run_id / "artifacts").resolve())
    )
    assert "r2_score_on_data_test" in logged_run.data.metrics
    artifacts = MlflowClient(tracking_uri).list_artifacts(
        run_id=logged_run.info.run_id, path="train"
    )
    assert {artifact.path for artifact in artifacts} == {
        "train/card.html",
        "train/estimator",
        "train/model",
    }
    run_tags = MlflowClient(tracking_uri).get_run(run_id=logged_run.info.run_id).data.tags
    pipelineSourceTag = {MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.PIPELINE)}
    assert resolve_tags(pipelineSourceTag).items() <= run_tags.items()

    pipeline.run()
    logged_runs = mlflow.search_runs(experiment_names=[experiment_name], output_format="list")
    assert len(logged_runs) == 1


@pytest.mark.usefixtures("enter_pipeline_example_directory")
def test_pipelines_run_sets_mlflow_git_tags():
    pipeline = Pipeline(profile="local")
    pipeline.clean()
    pipeline.run(step="train")

    profile_path = pathlib.Path.cwd() / "profiles" / "local.yaml"
    with open(profile_path, "r") as f:
        profile_contents = yaml.safe_load(f)

    tracking_uri = profile_contents["experiment"]["tracking_uri"]
    experiment_name = profile_contents["experiment"]["name"]

    mlflow.set_tracking_uri(tracking_uri)
    logged_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        output_format="list",
        order_by=["attributes.start_time DESC"],
    )
    logged_run = logged_runs[0]

    run_tags = MlflowClient(tracking_uri).get_run(run_id=logged_run.info.run_id).data.tags
    assert {
        MLFLOW_SOURCE_NAME,
        MLFLOW_GIT_COMMIT,
        MLFLOW_GIT_REPO_URL,
        LEGACY_MLFLOW_GIT_REPO_URL,
    } < run_tags.keys()
    assert run_tags[MLFLOW_SOURCE_NAME] == run_tags[MLFLOW_GIT_REPO_URL]


@pytest.mark.usefixtures("enter_test_pipeline_directory")
def test_pipelines_run_throws_exception_and_produces_failure_card_when_step_fails():
    profile_path = pathlib.Path.cwd() / "profiles" / "local.yaml"
    with open(profile_path, "r") as f:
        profile_contents = yaml.safe_load(f)

    profile_contents["INGEST_DATA_LOCATION"] = "a bad location"

    with open(profile_path, "w") as f:
        yaml.safe_dump(profile_contents, f)

    pipeline = Pipeline(profile="local")
    pipeline.clean()
    with pytest.raises(MlflowException, match="Failed to run.*test_pipeline.*ingest"):
        pipeline.run()
    with pytest.raises(MlflowException, match="Failed to run.*split.*test_pipeline.*ingest"):
        pipeline.run(step="split")

    step_card_path = get_step_output_path(
        pipeline_root_path=pipeline._pipeline_root_path,
        step_name="ingest",
        relative_path="card.html",
    )
    with open(step_card_path, "r") as f:
        card_content = f.read()

    assert "Ingest" in card_content
    assert "Failed" in card_content
    assert "Stacktrace" in card_content


@pytest.mark.usefixtures("enter_pipeline_example_directory")
def test_test_step_logs_step_cards_as_artifacts():
    pipeline = Pipeline(profile="local")
    pipeline.clean()
    pipeline.run()

    tracking_uri = pipeline._get_step("train").tracking_config.tracking_uri
    local_run_id_path = get_step_output_path(
        pipeline_root_path=pipeline._pipeline_root_path,
        step_name="train",
        relative_path="run_id",
    )
    run_id = pathlib.Path(local_run_id_path).read_text()
    artifacts = set(list_all_artifacts(tracking_uri, run_id))
    assert artifacts.issuperset(
        {
            "ingest/card.html",
            "split/card.html",
            "transform/card.html",
            "train/card.html",
            "evaluate/card.html",
            "register/card.html",
        }
    )


@pytest.mark.usefixtures("enter_pipeline_example_directory")
def test_pipeline_get_artifacts():
    pipeline = Pipeline(profile="local")
    pipeline.clean()

    pipeline.run("ingest")
    pipeline.run("split")
    pipeline.run("transform")
    pipeline.run("train")
    pipeline.run("register")

    assert isinstance(pipeline.get_artifact("ingested_data"), pd.DataFrame)
    assert isinstance(pipeline.get_artifact("training_data"), pd.DataFrame)
    assert isinstance(pipeline.get_artifact("validation_data"), pd.DataFrame)
    assert isinstance(pipeline.get_artifact("test_data"), pd.DataFrame)
    assert isinstance(pipeline.get_artifact("transformed_training_data"), pd.DataFrame)
    assert isinstance(pipeline.get_artifact("transformed_validation_data"), pd.DataFrame)
    assert hasattr(pipeline.get_artifact("transformer"), "transform")
    assert isinstance(pipeline.get_artifact("model"), mlflow.pyfunc.PyFuncModel)
    assert isinstance(pipeline.get_artifact("run"), Run)
    assert isinstance(pipeline.get_artifact("registered_model_version"), ModelVersion)

    with pytest.raises(MlflowException, match="The artifact with name 'abcde' is not supported."):
        pipeline.get_artifact("abcde")

    pipeline.clean()
    with mock.patch("mlflow.pipelines.regression.v1.pipeline._logger.warning") as mock_warning:
        pipeline.get_artifact("ingested_data")
        mock_warning.assert_called_once_with(
            "The artifact with name 'ingested_data' was not found."
            " Re-run the 'ingest' step to generate it."
        )


def test_generate_worst_examples_dataframe():
    test_df = pd.DataFrame(
        {
            "a": [3, 2, 5],
            "b": [6, 9, 1],
        }
    )
    target_col = "b"
    predictions = [5, 3, 4]

    result_df = BaseStep._generate_worst_examples_dataframe(
        test_df, predictions, target_col, worst_k=2
    )

    def assert_result_correct(df):
        assert df.columns.tolist() == ["absolute_error", "prediction", "b", "a"]
        assert df.absolute_error.tolist() == [6, 3]
        assert df.prediction.tolist() == [3, 4]
        assert df.b.tolist() == [9, 1]
        assert df.a.tolist() == [2, 5]
        assert df.index.tolist() == [0, 1]

    assert_result_correct(result_df)

    test_df2 = test_df.set_axis([2, 1, 0], axis="index")
    result_df2 = BaseStep._generate_worst_examples_dataframe(
        test_df2, predictions, target_col, worst_k=2
    )
    assert_result_correct(result_df2)
