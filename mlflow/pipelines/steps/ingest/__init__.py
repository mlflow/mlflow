import abc
import logging
import os

from pathlib import Path
from mlflow.exceptions import MlflowException
from mlflow.pipelines.artifacts import DataframeArtifact
from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.step import StepClass
from mlflow.pipelines.utils.step import get_pandas_data_profiles
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.file_utils import read_parquet_as_pandas_df
from mlflow.pipelines.steps.ingest.datasets import (
    ParquetDataset,
    DeltaTableDataset,
    SparkSqlDataset,
    CustomDataset,
)
from typing import Dict, Any
import pandas as pd

_logger = logging.getLogger(__name__)


class BaseIngestStep(BaseStep, metaclass=abc.ABCMeta):
    _DATASET_FORMAT_SPARK_TABLE = "spark_table"
    _DATASET_FORMAT_DELTA = "delta"
    _DATASET_FORMAT_PARQUET = "parquet"
    _DATASET_PROFILE_OUTPUT_NAME = "dataset_profile.html"
    _STEP_CARD_OUTPUT_NAME = "card.pkl"
    _SUPPORTED_DATASETS = [
        ParquetDataset,
        DeltaTableDataset,
        SparkSqlDataset,
        # NB: The custom dataset is deliberately listed last as a catch-all for any
        # format not matched by the datasets above. When mapping a format to a dataset,
        # datasets are explored in the listed order
        CustomDataset,
    ]

    def _validate_and_apply_step_config(self):
        if len(self.step_config) == 0:
            raise MlflowException(
                message="The `data` section of pipeline.yaml must be specified",
                error_code=INVALID_PARAMETER_VALUE,
            )
        dataset_format = self.step_config.get("format")
        if not dataset_format:
            raise MlflowException(
                message=(
                    "Dataset format must be specified via the `format` key within the `data`"
                    " section of pipeline.yaml"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        for dataset_class in BaseIngestStep._SUPPORTED_DATASETS:
            if dataset_class.handles_format(dataset_format):
                self.dataset = dataset_class.from_config(
                    dataset_config=self.step_config,
                    pipeline_root=self.pipeline_root,
                )
                break
        else:
            raise MlflowException(
                message=f"Unrecognized dataset format: {dataset_format}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.skip_data_profiling = self.step_config.get("skip_data_profiling", False)

    def _run(self, output_directory: str) -> BaseCard:

        dataset_dst_path = os.path.abspath(os.path.join(output_directory, self.dataset_output_name))
        self.dataset.resolve_to_parquet(
            dst_path=dataset_dst_path,
        )
        _logger.debug("Successfully stored data in parquet format at '%s'", dataset_dst_path)

        ingested_df = read_parquet_as_pandas_df(data_parquet_path=dataset_dst_path)
        ingested_dataset_profile = None
        if not self.skip_data_profiling:
            _logger.debug("Profiling ingested dataset")
            ingested_dataset_profile = get_pandas_data_profiles(
                [["Profile of Ingested Dataset", ingested_df]]
            )
            dataset_profile_path = Path(
                str(os.path.join(output_directory, BaseIngestStep._DATASET_PROFILE_OUTPUT_NAME))
            )
            dataset_profile_path.write_text(ingested_dataset_profile, encoding="utf-8")
            _logger.debug(f"Wrote dataset profile to '{dataset_profile_path}'")

        schema = pd.io.json.build_table_schema(ingested_df, index=False)

        step_card = self._build_step_card(
            ingested_dataset_profile=ingested_dataset_profile,
            ingested_rows=len(ingested_df),
            schema=schema,
            data_preview=ingested_df.head(),
            dataset_src_location=getattr(self.dataset, "location", None),
            dataset_sql=getattr(self.dataset, "sql", None),
        )
        return step_card

    def _build_step_card(
        self,
        ingested_dataset_profile: str,
        ingested_rows: int,
        schema: Dict,
        data_preview: pd.DataFrame = None,
        dataset_src_location: str = None,
        dataset_sql: str = None,
    ) -> BaseCard:
        """
        Constructs a step card instance corresponding to the current ingest step state.

        :param ingested_dataset_path: The local filesystem path to the ingested parquet dataset
                                      file.
        :param dataset_src_location: The source location of the dataset
                                     (e.g. '/tmp/myfile.parquet', 's3://mybucket/mypath', ...),
                                     if the dataset is a location-based dataset. Either
                                     ``dataset_src_location`` or ``dataset_sql`` must be specified.
        :param dataset_sql: The Spark SQL query string that defines the dataset
                            (e.g. 'SELECT * FROM my_spark_table'), if the dataset is a Spark SQL
                            dataset. Either ``dataset_src_location`` or ``dataset_sql`` must be
                            specified.
        :return: An BaseCard instance corresponding to the current ingest step state.
        """
        if dataset_src_location is None and dataset_sql is None:
            raise MlflowException(
                message=(
                    "Failed to build step card because neither a dataset location nor a"
                    " dataset Spark SQL query were specified"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

        card = BaseCard(self.pipeline_name, self.name)
        if not self.skip_data_profiling:
            (  # Tab #1 -- Ingested dataset profile.
                card.add_tab("Data Profile", "{{PROFILE}}").add_pandas_profile(
                    "PROFILE", ingested_dataset_profile
                )
            )
        # Tab #2 -- Ingested dataset schema.
        schema_html = BaseCard.render_table(schema["fields"])
        card.add_tab("Data Schema", "{{SCHEMA}}").add_html("SCHEMA", schema_html)

        if data_preview is not None:
            # Tab #3 -- Ingested dataset preview.
            card.add_tab("Data Preview", "{{DATA_PREVIEW}}").add_html(
                "DATA_PREVIEW", BaseCard.render_table(data_preview)
            )

        (  # Tab #4 -- Step run summary.
            card.add_tab(
                "Run Summary",
                "{{ INGESTED_ROWS }}"
                + "{{ DATA_SOURCE }}"
                + "{{ EXE_DURATION }}"
                + "{{ LAST_UPDATE_TIME }}",
            )
            .add_markdown(
                name="INGESTED_ROWS",
                markdown=f"**Number of rows ingested:** `{ingested_rows}`",
            )
            .add_markdown(
                name="DATA_SOURCE",
                markdown=(
                    f"**Dataset source location:** `{dataset_src_location}`"
                    if dataset_src_location is not None
                    else f"**Dataset SQL:** `{dataset_sql}`"
                ),
            )
        )
        return card


class IngestStep(BaseIngestStep):
    _DATASET_OUTPUT_NAME = "dataset.parquet"

    def __init__(self, step_config: Dict[str, Any], pipeline_root: str):
        super().__init__(step_config, pipeline_root)
        self.dataset_output_name = IngestStep._DATASET_OUTPUT_NAME

    @classmethod
    def from_pipeline_config(cls, pipeline_config: Dict[str, Any], pipeline_root: str):
        data_config = pipeline_config.get("data", {})
        ingest_config = pipeline_config.get("steps", {}).get("ingest", {})
        return cls(
            step_config={**data_config, **ingest_config},
            pipeline_root=pipeline_root,
        )

    @property
    def name(self) -> str:
        return "ingest"

    def get_artifacts(self):
        return [
            DataframeArtifact(
                "ingested_data", self.pipeline_root, self.name, IngestStep._DATASET_OUTPUT_NAME
            )
        ]

    def step_class(self):
        return StepClass.TRAINING


class IngestScoringStep(BaseIngestStep):
    _DATASET_OUTPUT_NAME = "scoring-dataset.parquet"

    def __init__(self, step_config: Dict[str, Any], pipeline_root: str):
        super().__init__(step_config, pipeline_root)
        self.dataset_output_name = IngestScoringStep._DATASET_OUTPUT_NAME

    @classmethod
    def from_pipeline_config(cls, pipeline_config: Dict[str, Any], pipeline_root: str):
        data_config = pipeline_config.get("data_scoring", {})
        ingest_config = pipeline_config.get("steps", {}).get("ingest", {})
        return cls(
            step_config={**data_config, **ingest_config},
            pipeline_root=pipeline_root,
        )

    @property
    def name(self) -> str:
        return "ingest_scoring"

    def get_artifacts(self):
        return [
            DataframeArtifact(
                "ingested_scoring_data",
                self.pipeline_root,
                self.name,
                IngestScoringStep._DATASET_OUTPUT_NAME,
            )
        ]

    def step_class(self):
        return StepClass.PREDICTION
