import logging
import os
import time
from typing import Dict, Any

import mlflow
from mlflow.exceptions import MlflowException, BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.step import get_pandas_data_profile
from mlflow.utils._spark_utils import _get_active_spark_session

_logger = logging.getLogger(__name__)


# This should maybe imported from the ingest scoring step for consistency
_INPUT_FILE_NAME = "scoring-dataset.parquet"
_SCORED_OUTPUT_FILE_NAME = "scored.parquet"
_PREDICTION_COLUMN_NAME = "prediction"


class PredictStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str):
        super().__init__(step_config, pipeline_root)
        self.skip_data_profiling = step_config.get("skip_data_profiling", False)

        self.run_end_time = None
        self.execution_duration = None

    def _build_profiles_and_card(self, scored_df) -> BaseCard:
        # Build profiles for scored dataset
        card = BaseCard(self.pipeline_name, self.name)

        if not self.skip_data_profiling:
            _logger.info("Profiling ingested dataset")
            scored_dataset_profile = get_pandas_data_profile(scored_df, "Profile of Scored Dataset")

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
                f"**Number of scored dataset rows:** `{len(scored_df)}`",
            )
        )

        return card

    def _run(self, output_directory):
        from pyspark.sql.functions import struct

        run_start_time = time.time()

        # Get or create spark session
        try:
            spark = _get_active_spark_session()
        except Exception as e:
            raise MlflowException(
                message=(
                    "Encountered an error while searching for an active Spark session to"
                    " score dataset with spark UDF. Please create a Spark session and try again."
                ),
                error_code=BAD_REQUEST,
            ) from e
        if not spark:
            raise MlflowException(
                message=(
                    "No active SparkSession detected to score dataset with spark UDF. "
                    "Please create a Spark session and try again."
                ),
                error_code=BAD_REQUEST,
            )

        # read cleaned dataset
        ingested_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="ingest_scoring",
            relative_path=_INPUT_FILE_NAME,
        )
        input_sdf = spark.read.parquet(ingested_data_path)
        if _PREDICTION_COLUMN_NAME in input_sdf.columns:
            _logger.warning(
                f"Input scoring dataframe already contains a column '{_PREDICTION_COLUMN_NAME}'. "
                f"This column will be dropped in favor of the predict output column name."
            )

        # score dataset
        model_uri = self.step_config["model_uri"]
        env_manager = "local" if "_disable_env_restoration" in self.step_config else "conda"
        predict = mlflow.pyfunc.spark_udf(spark, model_uri, env_manager=env_manager)
        scored_sdf = input_sdf.withColumn(
            _PREDICTION_COLUMN_NAME, predict(struct(*input_sdf.columns))
        )

        # save predictions
        # note: the current output writing logic allows no overwrites
        output_format = self.step_config["output_format"]
        if output_format == "parquet" or output_format == "delta":
            scored_sdf.coalesce(1).write.format(output_format).save(
                self.step_config["output_location"]
            )
        else:
            scored_sdf.write.format("delta").saveAsTable(self.step_config["output_location"])

        # predict step artifacts
        scored_pdf = scored_sdf.toPandas()
        scored_pdf.to_parquet(
            os.path.join(output_directory, _SCORED_OUTPUT_FILE_NAME), engine="pyarrow"
        )

        self.run_end_time = time.time()
        self.execution_duration = self.run_end_time - run_start_time
        return self._build_profiles_and_card(scored_pdf)

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        try:
            step_config = pipeline_config["steps"]["predict"]
        except KeyError:
            raise MlflowException(
                "Config for predict step is not found.", error_code=INVALID_PARAMETER_VALUE
            )
        required_configuration_keys = ["output_format", "output_location"]
        for key in required_configuration_keys:
            if key not in step_config:
                raise MlflowException(
                    f"The `{key}` configuration key must be specified for the predict step.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        if step_config["output_format"] not in {"parquet", "delta", "table"}:
            raise MlflowException(
                "Invalid `output_format` in predict step configuration.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if "model_uri" not in step_config:
            try:
                register_config = pipeline_config["steps"]["register"]
                model_name = register_config["model_name"]
            except KeyError:
                raise MlflowException(
                    "No model specified for batch scoring: predict step does not have `model_uri` "
                    "configuration key and register step does not have `model_name` configuration "
                    " key.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            else:
                step_config["model_uri"] = f"models:/{model_name}/latest"
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "predict"
