import logging
import os
import time
from typing import Dict, Any

import mlflow
from mlflow.exceptions import MlflowException, BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.utils._spark_utils import _get_active_spark_session

_logger = logging.getLogger(__name__)


# This should maybe imported from the preprocessing step for consistency
_INPUT_FILE_NAME = "dataset_preprocessed.parquet"
_SCORED_OUTPUT_FILE_NAME = "scored.parquet"


class PredictStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str):
        super().__init__(step_config, pipeline_root)

        self.run_end_time = None
        self.execution_duration = None

    def _build_profiles_and_card(self, scored_df) -> BaseCard:
        # do something with this to make the linter happy
        len(scored_df)
        # Build card
        card = BaseCard(self.pipeline_name, self.name)
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

        # read cleaned dataset
        ingested_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="preprocessing",
            relative_path=_INPUT_FILE_NAME,
        )
        input_sdf = spark.read.parquet(ingested_data_path)

        # score dataset
        model_uri = self.step_config["model_uri"]
        predict = mlflow.pyfunc.spark_udf(spark, model_uri)
        scored_sdf = input_sdf.withColumn("prediction", predict(struct(*input_sdf.columns)))

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
        scored_pdf.to_parquet(os.path.join(output_directory, "scored.parquet"), engine="pyarrow")

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
        required_configuration_keys = ["model_uri", "output_format", "output_location"]
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
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "predict"
