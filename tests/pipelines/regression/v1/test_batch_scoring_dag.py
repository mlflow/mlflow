import pandas as pd
import pathlib
import pytest
import shutil

import mlflow
from mlflow.pipelines.utils.execution import (
    get_or_create_base_execution_directory,
)
from mlflow.pipelines.regression.v1.pipeline import RegressionPipeline

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    enter_pipeline_example_directory,
    PIPELINE_EXAMPLE_PATH_FROM_MLFLOW_ROOT,
    chdir,
)  # pylint: enable=unused-import

_STEP_NAMES = ["ingest_scoring", "predict"]


# Because the Batch Scoring DAG takes a lot of time to run (around 5 minutes), we run the entire
# DAG in this set up function and then assert various expected results in the tests.
@pytest.fixture(scope="module", autouse=True)
def run_batch_scoring():
    mlflow_repo_root_directory = pathlib.Path(mlflow.__file__).parent.parent
    pipeline_example_path = mlflow_repo_root_directory / PIPELINE_EXAMPLE_PATH_FROM_MLFLOW_ROOT
    with chdir(pipeline_example_path):
        p = RegressionPipeline(pipeline_root_path=pipeline_example_path, profile="local")
        p.run("register")
        p.run("ingest_scoring")
        p.run("predict")
        yield p
        p.clean()
        shutil.rmtree("./data/sample_output.parquet", ignore_errors=True)


def test_pipeline_batch_dag_get_artifacts(run_batch_scoring):
    p = run_batch_scoring
    assert isinstance(p.get_artifact("ingested_scoring_data"), pd.DataFrame)
    assert isinstance(p.get_artifact("scored_data"), pd.DataFrame)


def test_pipeline_batch_dag_execution_directories(enter_pipeline_example_directory):
    expected_execution_directory_location = get_or_create_base_execution_directory(
        enter_pipeline_example_directory
    )
    for step_name in _STEP_NAMES:
        step_outputs_path = expected_execution_directory_location / "steps" / step_name / "outputs"
        assert step_outputs_path.exists()
        first_output = next(step_outputs_path.iterdir(), None)
        # TODO: Assert that the ingest step has outputs once ingest execution has been implemented
        assert first_output is not None or step_name == "ingest_scoring"


@pytest.mark.parametrize("step", _STEP_NAMES)
def test_pipeline_batch_dag_clean_step_works(
    step, run_batch_scoring, enter_pipeline_example_directory
):
    p = run_batch_scoring
    p.clean(step)
    expected_execution_directory_location = get_or_create_base_execution_directory(
        enter_pipeline_example_directory
    )
    step_outputs_path = expected_execution_directory_location / "steps" / step / "outputs"
    assert not list(step_outputs_path.iterdir())
