import logging
import os
import time
from typing import Dict, Any

import mlflow
from mlflow.exceptions import MlflowException, BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.pipelines.artifacts import DataframeArtifact, RegisteredModelVersionInfo
from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.step import StepClass
from mlflow.pipelines.steps.register import _REGISTERED_MV_INFO_FILE
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.step import get_pandas_data_profiles
from mlflow.pipelines.utils.tracking import (
    get_pipeline_tracking_config,
    apply_pipeline_tracking_config,
    TrackingConfig,
)
from mlflow.projects.utils import get_databricks_env_vars
from mlflow.utils.file_utils import write_spark_dataframe_to_parquet_on_local_disk
from mlflow.utils._spark_utils import (
    _get_active_spark_session,
    _create_local_spark_session_for_pipelines,
)

_logger = logging.getLogger(__name__)


# This should maybe imported from the ingest scoring step for consistency
_INPUT_FILE_NAME = "scoring-dataset.parquet"
_SCORED_OUTPUT_FILE_NAME = "scored.parquet"
_PREDICTION_COLUMN_NAME = "prediction"

# Max dataframe size for profiling after scoring
_MAX_PROFILE_SIZE = 10000
# Environment manager for Spark UDF model restoration
_ENV_MANAGER = "virtualenv"


class PredictStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str) -> None:
        super().__init__(step_config, pipeline_root)
        self.tracking_config = TrackingConfig.from_dict(self.step_config)

    def _validate_and_apply_step_config(self):
        required_configuration_keys = ["output_format", "output_location"]
        for key in required_configuration_keys:
            if key not in self.step_config:
                raise MlflowException(
                    f"The `{key}` configuration key must be specified for the predict step.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        if self.step_config["output_format"] not in {"parquet", "delta", "table"}:
            raise MlflowException(
                "Invalid `output_format` in predict step configuration.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if "model_uri" not in self.step_config:
            try:
                register_config = self.step_config["register"]
                model_name = register_config["model_name"]
            except KeyError:
                raise MlflowException(
                    "No model specified for batch scoring: predict step does not have `model_uri` "
                    "configuration key and register step does not have `model_name` configuration "
                    " key.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            else:
                self.step_config["model_uri"] = f"models:/{model_name}/latest"
        self.registry_uri = self.step_config.get("registry_uri", None)
        self.skip_data_profiling = self.step_config.get("skip_data_profiling", False)
        self.save_mode = self.step_config.get("save_mode", "default")
        self.run_end_time = None
        self.execution_duration = None

    def _build_profiles_and_card(self, scored_sdf) -> BaseCard:
        # Build profiles for scored dataset
        card = BaseCard(self.pipeline_name, self.name)

        scored_size = scored_sdf.count()

        if not self.skip_data_profiling:
            _logger.info("Profiling scored dataset")
            if scored_size > _MAX_PROFILE_SIZE:
                _logger.info("Sampling scored dataset for profiling because dataset size is large.")
                sample_percentage = _MAX_PROFILE_SIZE / scored_size
                scored_sdf = scored_sdf.sample(sample_percentage)
            scored_df = scored_sdf.toPandas()
            scored_dataset_profile = get_pandas_data_profiles(
                [["Profile of Scored Dataset", scored_df]]
            )

            # Optional tab : data profile for scored data:
            card.add_tab("Scored Data Profile", "{{PROFILE}}").add_pandas_profile(
                "PROFILE", scored_dataset_profile
            )

        # Tab #1/2: run summary.
        (
            card.add_tab(
                "Run Summary",
                """
                {{ SCORED_DATA_NUM_ROWS }}
                {{ EXE_DURATION }}
                {{ LAST_UPDATE_TIME }}
                """,
            ).add_markdown(
                "SCORED_DATA_NUM_ROWS",
                f"**Number of scored dataset rows:** `{scored_size}`",
            )
        )

        return card

    def _run(self, output_directory):
        import pandas as pd
        from pyspark.sql.functions import struct

        run_start_time = time.time()

        apply_pipeline_tracking_config(self.tracking_config)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)

        # Get or create spark session
        try:
            spark = _get_active_spark_session()
            if spark:
                _logger.info("Found active spark session")
            else:
                _logger.info("Creating new spark session")
                spark = _create_local_spark_session_for_pipelines()
        except Exception as e:
            raise MlflowException(
                message=(
                    "Encountered an error while getting or creating an active Spark session to"
                    " score dataset with spark UDF."
                ),
                error_code=BAD_REQUEST,
            ) from e

        # check if output location is already populated for non-delta output formats
        output_format = self.step_config["output_format"]
        output_location = self.step_config["output_location"]
        output_populated = False
        if self.save_mode in ["default", "error", "errorifexists"]:
            if output_format == "parquet" or output_format == "delta":
                output_populated = os.path.exists(output_location)
            else:
                try:
                    output_populated = spark._jsparkSession.catalog().tableExists(output_location)
                except Exception:
                    # swallow spark failures
                    pass
        if output_populated:
            raise MlflowException(
                message=(
                    f"Output location `{output_location}` of format `{output_format}` is already "
                    "populated. To overwrite, please change the spark `save_mode` in the predict "
                    "step configuration."
                ),
                error_code=BAD_REQUEST,
            )

        # read cleaned dataset
        ingested_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="ingest_scoring",
            relative_path=_INPUT_FILE_NAME,
        )
        # Because the cached parquet file is not on DBFS, we have to first load it as a pandas df
        input_pdf = pd.read_parquet(ingested_data_path)
        input_sdf = spark.createDataFrame(input_pdf)
        if _PREDICTION_COLUMN_NAME in input_sdf.columns:
            _logger.warning(
                f"Input scoring dataframe already contains a column '{_PREDICTION_COLUMN_NAME}'. "
                f"This column will be dropped in favor of the predict output column name."
            )

        # get model uri
        model_uri = self.step_config["model_uri"]
        registered_model_file_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="register",
            relative_path=_REGISTERED_MV_INFO_FILE,
        )
        if os.path.exists(registered_model_file_path):
            rmi = RegisteredModelVersionInfo.from_json(path=registered_model_file_path)
            model_uri = f"models:/{rmi.name}/{rmi.version}"

        # scored dataset
        result_type = self.step_config.get("result_type", "double")
        predict = mlflow.pyfunc.spark_udf(
            spark, model_uri, result_type=result_type, env_manager=_ENV_MANAGER
        )
        scored_sdf = input_sdf.withColumn(
            _PREDICTION_COLUMN_NAME, predict(struct(*input_sdf.columns))
        )

        # save predictions
        if output_format in ["parquet", "delta"]:
            scored_sdf.coalesce(1).write.format(output_format).mode(self.save_mode).save(
                output_location
            )
        else:
            scored_sdf.write.format("delta").mode(self.save_mode).saveAsTable(output_location)

        # predict step artifacts
        write_spark_dataframe_to_parquet_on_local_disk(
            scored_sdf, os.path.join(output_directory, _SCORED_OUTPUT_FILE_NAME)
        )

        self.run_end_time = time.time()
        self.execution_duration = self.run_end_time - run_start_time
        return self._build_profiles_and_card(scored_sdf)

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = {}
        if pipeline_config.get("steps", {}).get("predict", {}) is not None:
            step_config.update(pipeline_config.get("steps", {}).get("predict", {}))
        step_config["register"] = pipeline_config.get("steps", {}).get("register", {})
        step_config["registry_uri"] = pipeline_config.get("model_registry", {}).get("uri", None)
        step_config.update(
            get_pipeline_tracking_config(
                pipeline_root_path=pipeline_root,
                pipeline_config=pipeline_config,
            ).to_dict()
        )
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "predict"

    @property
    def environment(self):
        return get_databricks_env_vars(tracking_uri=self.tracking_config.tracking_uri)

    def get_artifacts(self):
        return [
            DataframeArtifact(
                "scored_data", self.pipeline_root, self.name, _SCORED_OUTPUT_FILE_NAME
            )
        ]

    def step_class(self):
        return StepClass.PREDICTION
