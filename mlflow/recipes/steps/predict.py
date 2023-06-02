import logging
import os
import time
from typing import Dict, Any

import mlflow
from mlflow.exceptions import MlflowException, BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.recipes.artifacts import DataframeArtifact, RegisteredModelVersionInfo
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep
from mlflow.recipes.step import StepClass
from mlflow.recipes.steps.register import _REGISTERED_MV_INFO_FILE
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.step import get_pandas_data_profiles
from mlflow.recipes.utils.tracking import (
    get_recipe_tracking_config,
    apply_recipe_tracking_config,
    TrackingConfig,
)
from mlflow.utils.databricks_utils import get_databricks_env_vars
from mlflow.utils.file_utils import write_spark_dataframe_to_parquet_on_local_disk
from mlflow.utils._spark_utils import (
    _get_active_spark_session,
    _create_local_spark_session_for_recipes,
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
    def __init__(self, step_config: Dict[str, Any], recipe_root: str) -> None:
        super().__init__(step_config, recipe_root)
        self.tracking_config = TrackingConfig.from_dict(self.step_config)

    def _validate_and_apply_step_config(self):
        required_configuration_keys = ["using", "location"]
        for key in required_configuration_keys:
            if key not in self.step_config:
                raise MlflowException(
                    f"The `{key}` configuration key must be specified for the predict step.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        if self.step_config["using"] not in {"parquet", "delta", "table"}:
            raise MlflowException(
                "Invalid `using` in predict step configuration.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if "model_uri" not in self.step_config:
            try:
                register_config = self.step_config["model_registry"]
                model_name = register_config["model_name"]
            except KeyError:
                raise MlflowException(
                    "No model specified for batch scoring: model_registry does not have "
                    "`model_uri` and does not have `model_name` configuration key.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            else:
                self.step_config["model_uri"] = f"models:/{model_name}/latest"
        self.registry_uri = self.step_config.get("registry_uri", None)
        self.skip_data_profiling = self.step_config.get("skip_data_profiling", False)
        self.save_mode = self.step_config.get("save_mode", "overwrite")
        self.run_end_time = None
        self.execution_duration = None

    def _build_profiles_and_card(self, scored_sdf) -> BaseCard:
        # Build profiles for scored dataset
        card = BaseCard(self.recipe_name, self.name)

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

        apply_recipe_tracking_config(self.tracking_config)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)

        # Get or create spark session
        try:
            spark = _get_active_spark_session()
            if spark:
                _logger.info("Found active spark session")
            else:
                _logger.info("Creating new spark session")
                spark = _create_local_spark_session_for_recipes()
        except Exception as e:
            raise MlflowException(
                message=(
                    "Encountered an error while getting or creating an active Spark session to"
                    " score dataset with spark UDF."
                ),
                error_code=BAD_REQUEST,
            ) from e

        # read cleaned dataset
        ingested_data_path = get_step_output_path(
            recipe_root_path=self.recipe_root,
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
            recipe_root_path=self.recipe_root,
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

        # check if output location is already populated for non-delta output formats
        output_format = self.step_config["using"]
        output_location = self.step_config["location"]
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
                    f"Output location `{output_location}` using format `{output_format}` is "
                    "already populated. To overwrite, please change the spark `save_mode` in "
                    "the predict step configuration."
                ),
                error_code=BAD_REQUEST,
            )

        if output_format == "table":
            try:
                from delta.tables import DeltaTable

                output_populated = DeltaTable.forName(spark, output_location)
            except Exception:
                # swallow spark failures
                pass

            if output_populated:
                _logger.info(f"Table already exists at {output_location}")
                # If the table already exists, we are just setting up the table properties to
                # ensure that the table can be written with column names with spaces.
                spark.sql(
                    f"ALTER TABLE {output_location} SET TBLPROPERTIES "
                    "('delta.columnMapping.mode'='name','delta.minReaderVersion'='2',"
                    "'delta.minWriterVersion'='5')"
                )
            else:
                _logger.info(f"Creating a new table at {output_location}")
                from delta.tables import DeltaTable

                # If the table location specified doesn't exist, we are creating a new table
                # with properties required to ensure that column names can have spaces.
                DeltaTable.create().addColumns(scored_sdf.schema).property(
                    "delta.minReaderVersion", "2"
                ).property("delta.minWriterVersion", "5").property(
                    "delta.columnMapping.mode", "name"
                ).tableName(
                    output_location
                ).execute()
                # We are overriding the save_mode to append for the create case, since the table
                # is already created above, so adding any record to the empty table can be
                # appended to the table
                self.save_mode = "append"

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
    def from_recipe_config(cls, recipe_config, recipe_root):
        step_config = {}
        if recipe_config.get("steps", {}).get("predict", {}) is not None:
            step_config.update(recipe_config.get("steps", {}).get("predict", {}))
        if recipe_config.get("steps", {}).get("predict", {}).get("output", {}) is not None:
            step_config.update(recipe_config.get("steps", {}).get("predict", {}).get("output", {}))
        step_config["register"] = recipe_config.get("steps", {}).get("register", {})
        step_config["model_registry"] = recipe_config.get("model_registry", {})
        step_config["recipe"] = recipe_config.get("recipe", "regression/v1")
        if recipe_config.get("model_registry", {}).get("registry_uri") is not None:
            step_config["registry_uri"] = recipe_config.get("model_registry", {}).get(
                "registry_uri"
            )

        step_config.update(
            get_recipe_tracking_config(
                recipe_root_path=recipe_root,
                recipe_config=recipe_config,
            ).to_dict()
        )
        return cls(step_config, recipe_root)

    @property
    def name(self):
        return "predict"

    @property
    def environment(self):
        return get_databricks_env_vars(tracking_uri=self.tracking_config.tracking_uri)

    def get_artifacts(self):
        return [
            DataframeArtifact("scored_data", self.recipe_root, self.name, _SCORED_OUTPUT_FILE_NAME)
        ]

    def step_class(self):
        return StepClass.PREDICTION
