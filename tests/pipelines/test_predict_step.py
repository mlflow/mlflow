import os
import pytest

from mlflow.exceptions import MlflowException
from mlflow.pipelines.steps.predict import PredictStep

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    enter_test_pipeline_directory,
    enter_pipeline_example_directory,
)  # pylint: enable=unused-import


@pytest.mark.usefixtures("enter_test_pipeline_directory")
def test_predict_throws_when_improperly_configured():
    from os import listdir
    from os.path import isfile, join

    onlyfiles = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]
    print(onlyfiles)

    with pytest.raises(MlflowException, match="Config for predict step is not found"):
        PredictStep.from_pipeline_config(
            pipeline_config={},
            pipeline_root=os.getcwd(),
        )

    with pytest.raises(
        MlflowException, match="The `output_format` configuration key must be specified"
    ):
        PredictStep.from_pipeline_config(
            pipeline_config={
                "steps": {
                    "predict": {
                        "model_uri": "my model",
                        # Missing output_format
                    }
                }
            },
            pipeline_root=os.getcwd(),
        )

    with pytest.raises(
        MlflowException, match="The `model_uri` configuration key must be specified"
    ):
        PredictStep.from_pipeline_config(
            pipeline_config={
                "steps": {
                    "predict": {
                        # Missing model_uri
                        "output_format": "parquet",
                    }
                }
            },
            pipeline_root=os.getcwd(),
        )

    with pytest.raises(
        MlflowException, match="Invalid `output_format` in predict step configuration"
    ):
        PredictStep.from_pipeline_config(
            pipeline_config={
                "steps": {
                    "predict": {
                        "model_uri": "my model",
                        "output_format": "fancy_format",
                    }
                }
            },
            pipeline_root=os.getcwd(),
        )
