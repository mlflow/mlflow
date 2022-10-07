import pytest

from mlflow.pipelines.regression.v1.pipeline import RegressionPipeline

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    enter_pipeline_example_directory,
)  # pylint: enable=unused-import


@pytest.fixture
def create_pipeline(enter_pipeline_example_directory):
    pipeline_root_path = enter_pipeline_example_directory
    profile = "local"
    return RegressionPipeline(pipeline_root_path=pipeline_root_path, profile=profile)
