import logging
import os
import time
import importlib
import sys
from typing import Dict, Any

from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.step import get_pandas_data_profile
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE


_logger = logging.getLogger(__name__)


_INPUT_FILE_NAME = "dataset.parquet"
_SCORED_OUTPUT_FILE_NAME = "dataset_scored.parquet"


class PredictStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str):
        super().__init__(step_config, pipeline_root)

        self.run_end_time = None
        self.execution_duration = None

    def _build_profiles_and_card(self, scored_df) -> BaseCard:
        # Build profiles for input dataset, and train / validation / test splits
        scored_data_profile = get_pandas_data_profile(
            scored_df.reset_index(drop=True),
            "Profile of Scored Dataset",
        )

        # Build card
        card = BaseCard(self.pipeline_name, self.name)
        # Tab #1: data profile for cleaned data:
        card.add_tab("Scored Data Profile", "{{PROFILE}}").add_pandas_profile(
            "PROFILE", scored_data_profile
        )
        # Tab #2: run summary.
        (
            card.add_tab(
                "Run Summary",
                """
                {{ SCHEMA_LOCATION }}
                {{ SCORED_DATA_NUM_ROWS }}
                {{ EXE_DURATION}}
                {{ LAST_UPDATE_TIME }}
                """,
            )
            .add_markdown(
                "SCORED_DATA_NUM_ROWS", f"**Number of scored dataset rows:** `{len(scored_df)}`"
            )
        )

        return card

    def _run(self, output_directory):
        import pandas as pd

        run_start_time = time.time()

        # read cleaned dataset
        ingested_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="data_clean",
            relative_path=_INPUT_FILE_NAME,
        )
        input_df = pd.read_parquet(ingested_data_path)

        scored_df = input_df
        scored_df["score"] = 1

        # Output train / validation / test splits
        scored_df.to_parquet(os.path.join(output_directory, _SCORED_OUTPUT_FILE_NAME))

        self.run_end_time = time.time()
        self.execution_duration = self.run_end_time - run_start_time
        return self._build_profiles_and_card(scored_df)

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = pipeline_config.get("steps", {}).get("predict", {})
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "predict"
