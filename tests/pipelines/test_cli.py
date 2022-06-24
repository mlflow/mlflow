import pytest

import mlflow.pipelines.cli as pipelines_cli

from click.testing import CliRunner
from mlflow.exceptions import MlflowException
from mlflow.pipelines.utils import _PIPELINE_PROFILE_ENV_VAR

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    enter_pipeline_example_directory,
)  # pylint: enable=unused-import

_STEP_NAMES = ["ingest", "split", "train", "transform", "evaluate", "register"]


@pytest.fixture
def clean_up_pipeline():
    try:
        yield
    finally:
        CliRunner().invoke(pipelines_cli.clean)


@pytest.mark.usefixtures("enter_pipeline_example_directory", "clean_up_pipeline")
@pytest.mark.parametrize("step", _STEP_NAMES)
def test_pipelines_cli_step_works(step):
    CliRunner().invoke(cli=pipelines_cli.run, args=f"--step {step}")
    CliRunner().invoke(cli=pipelines_cli.inspect, args=f"--step {step}")
    CliRunner().invoke(cli=pipelines_cli.clean, args=f"--step {step}")


@pytest.mark.usefixtures("enter_pipeline_example_directory", "clean_up_pipeline")
def test_pipelines_cli_flow_completes_successfully():
    CliRunner().invoke(pipelines_cli.clean)
    CliRunner().invoke(pipelines_cli.run)
    CliRunner().invoke(pipelines_cli.inspect)


@pytest.mark.usefixtures("enter_pipeline_example_directory", "clean_up_pipeline")
def test_pipelines_cli_fails_with_illegal_profile():
    result = CliRunner().invoke(
        pipelines_cli.clean, env={_PIPELINE_PROFILE_ENV_VAR: "illegal_profile"}
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, MlflowException)


@pytest.mark.usefixtures("enter_pipeline_example_directory", "clean_up_pipeline")
def test_pipelines_cli_works_with_non_default_profile():
    result = CliRunner().invoke(pipelines_cli.clean, env={_PIPELINE_PROFILE_ENV_VAR: "databricks"})
    assert "with profile: 'databricks'" in result.output
    assert result.exit_code == 0
