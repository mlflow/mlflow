import logging
import os
import time
from typing import Dict, Any

from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.step import get_pandas_data_profile


_logger = logging.getLogger(__name__)


# This should maybe imported from the preprocessing step for consistency
_INPUT_FILE_NAME = "dataset_preprocessed.parquet"
_SCORED_OUTPUT_FILE_NAME = "dataset_scored.parquet"


class PredictStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str):
        super().__init__(step_config, pipeline_root)

        self.run_end_time = None
        self.execution_duration = None

    def _build_profiles_and_card(self, scored_df) -> BaseCard:

        # Build card
        card = BaseCard(self.pipeline_name, self.name)
        return card

    def _run(self, output_directory):
        import pandas as pd

        run_start_time = time.time()

        # read cleaned dataset
        ingested_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="preprocessing",
            relative_path=_INPUT_FILE_NAME,
        )
        input_df = pd.read_parquet(ingested_data_path)

        # TODO: Read the model specified from YAML using the
        #  tracking/model_registry URI also specified in the YAML

        # TODO: Score the input_df against the specified model
        scored_df = input_df
        scored_df["score"] = 1

        # TODO: Account for the 3 different types of output formats
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
