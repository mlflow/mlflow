import os
import logging

import pandas as pd

import mlflow.pipelines.regression.v1.dag_help_strings as dag_help_strings
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient
from mlflow.pipelines.pipeline import _BasePipeline
from mlflow.pipelines.steps.ingest import IngestStep
from mlflow.pipelines.steps.split import (
    SplitStep,
    _OUTPUT_TRAIN_FILE_NAME,
    _OUTPUT_VALIDATION_FILE_NAME,
    _OUTPUT_TEST_FILE_NAME,
)
from mlflow.pipelines.steps.transform import TransformStep
from mlflow.pipelines.steps.train import TrainStep
from mlflow.pipelines.steps.evaluate import EvaluateStep
from mlflow.pipelines.steps.register import RegisterStep, RegisteredModelVersionInfo
from mlflow.pipelines.step import BaseStep
from typing import List
from mlflow.pipelines.utils import get_pipeline_root_path
from mlflow.pipelines.utils.execution import get_or_create_base_execution_directory
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@experimental
class RegressionPipeline(_BasePipeline):
    """
    Regression Pipeline
    """

    _PIPELINE_STEPS = (IngestStep, SplitStep, TransformStep, TrainStep, EvaluateStep, RegisterStep)

    def _get_step_classes(self) -> List[BaseStep]:
        return self._PIPELINE_STEPS

    def _get_pipeline_dag_file(self) -> str:
        import jinja2

        j2_env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(__file__)))
        pipeline_dag_template = j2_env.get_template("resources/pipeline_dag_template.html").render(
            {
                "pipeline_yaml_help": {
                    "help_string_type": "yaml",
                    "help_string": dag_help_strings.PIPELINE_YAML,
                },
                "ingest_step_help": {
                    "help_string": dag_help_strings.INGEST_STEP,
                    "help_string_type": "text",
                },
                "ingest_user_code_help": {
                    "help_string": dag_help_strings.INGEST_USER_CODE,
                    "help_string_type": "python",
                },
                "ingested_data_help": {
                    "help_string": dag_help_strings.INGESTED_DATA,
                    "help_string_type": "text",
                },
                "split_step_help": {
                    "help_string": dag_help_strings.SPLIT_STEP,
                    "help_string_type": "text",
                },
                "split_user_code_help": {
                    "help_string": dag_help_strings.SPLIT_USER_CODE,
                    "help_string_type": "python",
                },
                "training_data_help": {
                    "help_string": dag_help_strings.TRAINING_DATA,
                    "help_string_type": "text",
                },
                "validation_data_help": {
                    "help_string": dag_help_strings.VALIDATION_DATA,
                    "help_string_type": "text",
                },
                "test_data_help": {
                    "help_string": dag_help_strings.TEST_DATA,
                    "help_string_type": "text",
                },
                "transform_step_help": {
                    "help_string": dag_help_strings.TRANSFORM_STEP,
                    "help_string_type": "text",
                },
                "transform_user_code_help": {
                    "help_string": dag_help_strings.TRANSFORM_USER_CODE,
                    "help_string_type": "python",
                },
                "fitted_transformer_help": {
                    "help_string": dag_help_strings.FITTED_TRANSFORMER,
                    "help_string_type": "text",
                },
                "transformed_training_and_validation_data_help": {
                    "help_string": dag_help_strings.TRANSFORMED_TRAINING_AND_VALIDATION_DATA,
                    "help_string_type": "text",
                },
                "train_step_help": {
                    "help_string": dag_help_strings.TRAIN_STEP,
                    "help_string_type": "text",
                },
                "train_user_code_help": {
                    "help_string": dag_help_strings.TRAIN_USER_CODE,
                    "help_string_type": "python",
                },
                "fitted_model_help": {
                    "help_string": dag_help_strings.FITTED_MODEL,
                    "help_string_type": "text",
                },
                "mlflow_run_help": {
                    "help_string": dag_help_strings.MLFLOW_RUN,
                    "help_string_type": "text",
                },
                "custom_metrics_user_code_help": {
                    "help_string": dag_help_strings.CUSTOM_METRICS_USER_CODE,
                    "help_string_type": "python",
                },
                "evaluate_step_help": {
                    "help_string": dag_help_strings.EVALUATE_STEP,
                    "help_string_type": "text",
                },
                "model_validation_status_help": {
                    "help_string": dag_help_strings.MODEL_VALIDATION_STATUS,
                    "help_string_type": "text",
                },
                "register_step_help": {
                    "help_string": dag_help_strings.REGISTER_STEP,
                    "help_string_type": "text",
                },
                "registered_model_version_help": {
                    "help_string": dag_help_strings.REGISTERED_MODEL_VERSION,
                    "help_string_type": "text",
                },
            }
        )

        pipeline_dag_file = os.path.join(
            get_or_create_base_execution_directory(self._pipeline_root_path), "pipeline_dag.html"
        )
        with open(pipeline_dag_file, "w") as f:
            f.write(pipeline_dag_template)

        return pipeline_dag_file

    @experimental
    def get_artifact(self, artifact: str):
        """
        Read an artifact from pipeline output. artifact names can be obtained from
        `Pipeline.inspect()` or `Pipeline.run()` output.

        Returns None if the specified artifact is not found.
        Raise an error if the artifact is not supported.

        :param artifact: A string representing the artifact, supported artifact values are:

         - `ingested_data`: returns the ingested data created in the ingest step as a pandas
           DataFrame.

         - `training_data`: returns the training data created in the split step as a
           pandas DataFrame.

         - `validation_data`: returns the validation data created in the split step as
           a pandas DataFrame.

         - `test_data`: returns the test data created in the split step as a pandas
           DataFrame.

         - `transformed_training_data`: returns the transformed training data created in the
           transform step as a pandas DataFrame.

         - `transformed_validation_data`: returns the transformed validation data created in
           the transform step as a pandas DataFrame.

         - `model`: returns the Pyfunc model from the train step output.

         - `transformer`: returns the sklearn transformer from transform step output.

         - `run`: returns an MLflow run object.

         - `registered_model_version`: returns the MLflow Model Registry ModelVersion created by
                                       register step.
        """
        ingest_step, split_step, transform_step, train_step, _, register_step = self._steps

        ingest_output_dir = get_step_output_path(self._pipeline_root_path, ingest_step.name, "")
        split_output_dir = get_step_output_path(self._pipeline_root_path, split_step.name, "")
        transform_output_dir = get_step_output_path(
            self._pipeline_root_path, transform_step.name, ""
        )
        train_output_dir = get_step_output_path(self._pipeline_root_path, train_step.name, "")

        def log_artifact_not_found_warning(artifact_name, step_name):
            _logger.warning(
                f"{artifact_name} is not found. Re-run the {step_name} step to generate."
            )

        def read_run_id():
            run_id_file_path = os.path.join(train_output_dir, "run_id")
            if os.path.exists(run_id_file_path):
                with open(run_id_file_path, "r") as f:
                    return f.read().strip()
            else:
                return None

        train_step_tracking_uri = train_step.tracking_config.tracking_uri
        pipeline_root_path = get_pipeline_root_path()

        def read_dataframe(artifact_name, output_dir, file_name, step_name):
            data_path = os.path.join(output_dir, file_name)
            if os.path.exists(data_path):
                return pd.read_parquet(data_path)
            else:
                log_artifact_not_found_warning(artifact_name, step_name)
                return None

        if artifact == "ingested_data":
            return read_dataframe(
                "ingested_data",
                ingest_output_dir,
                IngestStep._DATASET_OUTPUT_NAME,
                ingest_step.name,
            )

        elif artifact == "training_data":
            return read_dataframe(
                "training_data", split_output_dir, _OUTPUT_TRAIN_FILE_NAME, split_step.name
            )

        elif artifact == "validation_data":
            return read_dataframe(
                "validation_data", split_output_dir, _OUTPUT_VALIDATION_FILE_NAME, split_step.name
            )

        elif artifact == "test_data":
            return read_dataframe(
                "test_data", split_output_dir, _OUTPUT_TEST_FILE_NAME, split_step.name
            )

        elif artifact == "transformed_training_data":
            return read_dataframe(
                "transformed_training_data",
                transform_output_dir,
                "transformed_training_data.parquet",
                transform_step.name,
            )

        elif artifact == "transformed_validation_data":
            return read_dataframe(
                "transformed_validation_data",
                transform_output_dir,
                "transformed_validation_data.parquet",
                transform_step.name,
            )

        elif artifact == "model":
            run_id = read_run_id()
            if run_id:
                with _use_tracking_uri(train_step_tracking_uri, pipeline_root_path):
                    return mlflow.pyfunc.load_model(f"runs:/{run_id}/{train_step.name}/model")
            else:
                log_artifact_not_found_warning("model", train_step.name)
                return None

        elif artifact == "transformer":
            run_id = read_run_id()
            if run_id:
                with _use_tracking_uri(train_step_tracking_uri, pipeline_root_path):
                    return mlflow.sklearn.load_model(
                        f"runs:/{run_id}/{transform_step.name}/transformer"
                    )
            else:
                log_artifact_not_found_warning("transformer", train_step.name)
                return None

        elif artifact == "run":
            run_id = read_run_id()
            if run_id:
                with _use_tracking_uri(train_step_tracking_uri, pipeline_root_path):
                    return MlflowClient().get_run(run_id)
            else:
                log_artifact_not_found_warning("mlflow run", train_step.name)
                return None

        elif artifact == "registered_model_version":
            register_output_dir = get_step_output_path(
                self._pipeline_root_path, register_step.name, ""
            )
            registered_model_info_path = os.path.join(
                register_output_dir, "registered_model_version.json"
            )
            if os.path.exists(registered_model_info_path):
                registered_model_info = RegisteredModelVersionInfo.from_json(
                    path=registered_model_info_path
                )
                with _use_tracking_uri(train_step_tracking_uri, pipeline_root_path):
                    return MlflowClient().get_model_version(
                        name=registered_model_info.name, version=registered_model_info.version
                    )
            else:
                log_artifact_not_found_warning("registered_model_version", register_step.name)
                return None

        else:
            raise MlflowException(
                f"The artifact {artifact} is not supported.", error_code=INVALID_PARAMETER_VALUE
            )
