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


_SPLIT_HASH_BUCKET_NUM = 1000
_INPUT_FILE_NAME = "dataset.parquet"
_OUTPUT_TRAIN_FILE_NAME = "train.parquet"
_OUTPUT_VALIDATION_FILE_NAME = "validation.parquet"
_OUTPUT_TEST_FILE_NAME = "test.parquet"


def _make_elem_hashable(elem):
    import numpy as np

    if isinstance(elem, list):
        return tuple(_make_elem_hashable(e) for e in elem)
    elif isinstance(elem, dict):
        return tuple((_make_elem_hashable(k), _make_elem_hashable(v)) for k, v in elem.items())
    elif isinstance(elem, np.ndarray):
        return elem.shape, tuple(elem.flatten(order="C"))
    else:
        return elem


def _get_split_df(input_df, hash_buckets, split_ratios):
    # split dataset into train / validation / test splits
    train_ratio, validation_ratio, test_ratio = split_ratios
    ratio_sum = train_ratio + validation_ratio + test_ratio
    train_bucket_end = train_ratio / ratio_sum
    validation_bucket_end = (train_ratio + validation_ratio) / ratio_sum
    train_df = input_df[hash_buckets.map(lambda x: x < train_bucket_end)]
    validation_df = input_df[
        hash_buckets.map(lambda x: train_bucket_end <= x < validation_bucket_end)
    ]
    test_df = input_df[hash_buckets.map(lambda x: x >= validation_bucket_end)]

    empty_splits = [
        split_name
        for split_name, split_df in [
            ("train split", train_df),
            ("validation split", validation_df),
            ("test split", test_df),
        ]
        if len(split_df) == 0
    ]
    if len(empty_splits) > 0:
        _logger.warning(f"The following input dataset splits are empty: {','.join(empty_splits)}.")
    return train_df, validation_df, test_df


def _hash_pandas_dataframe(input_df):
    from pandas.util import hash_pandas_object

    return hash_pandas_object(input_df.applymap(_make_elem_hashable))


def _create_hash_buckets(input_df):
    # Create hash bucket used for splitting dataset
    # Note: use `hash_pandas_object` instead of python builtin hash because it is stable
    # across different process runs / different python versions
    start_time = time.time()
    hash_buckets = _hash_pandas_dataframe(input_df).map(
        lambda x: (x % _SPLIT_HASH_BUCKET_NUM) / _SPLIT_HASH_BUCKET_NUM
    )
    execution_duration = time.time() - start_time
    _logger.info(
        f"Creating hash buckets on input dataset containing {len(input_df)} "
        f"rows consumes {execution_duration} seconds."
    )
    return hash_buckets


class SplitStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str):
        super().__init__(step_config, pipeline_root)

        self.run_end_time = None
        self.execution_duration = None
        self.num_dropped_rows = None

        self.target_col = self.step_config.get("target_col")
        if self.target_col is None:
            raise MlflowException(
                "Missing target_col config in pipeline config.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        split_ratios = self.step_config.get("split_ratios", [0.75, 0.125, 0.125])
        if not (
            isinstance(split_ratios, list)
            and len(split_ratios) == 3
            and all(isinstance(x, (int, float)) and x > 0 for x in split_ratios)
        ):
            raise MlflowException(
                "Config split_ratios must be a list containing 3 positive numbers."
            )

        self.split_ratios = split_ratios

    def _build_profiles_and_card(self, train_df, validation_df, test_df) -> BaseCard:
        # Build profiles for input dataset, and train / validation / test splits
        train_profile = get_pandas_data_profile(
            train_df.reset_index(drop=True),
            "Profile of Train Dataset",
        )
        validation_profile = get_pandas_data_profile(
            validation_df.reset_index(drop=True),
            "Profile of Validation Dataset",
        )
        test_profile = get_pandas_data_profile(
            test_df.reset_index(drop=True),
            "Profile of Test Dataset",
        )

        # Build card
        card = BaseCard(self.pipeline_name, self.name)
        # Tab #1 - #3: data profiles for train/validation and test.
        card.add_tab("Data Profile (Train)", "{{PROFILE}}").add_pandas_profile(
            "PROFILE", train_profile
        )
        card.add_tab("Data Profile (Validation)", "{{PROFILE}}").add_pandas_profile(
            "PROFILE", validation_profile
        )
        card.add_tab("Data Profile (Test)", "{{PROFILE}}").add_pandas_profile(
            "PROFILE", test_profile
        )
        # Tab #4: run summary.
        (
            card.add_tab(
                "Run Summary",
                """
                {{ SCHEMA_LOCATION }}
                {{ TRAIN_SPLIT_NUM_ROWS }}
                {{ VALIDATION_SPLIT_NUM_ROWS }}
                {{ TEST_SPLIT_NUM_ROWS }}
                {{ NUM_DROPPED_ROWS }}
                {{ EXE_DURATION}}
                {{ LAST_UPDATE_TIME }}
                """,
            )
            .add_markdown(
                "NUM_DROPPED_ROWS", f"**Number of dropped rows:** `{self.num_dropped_rows}`"
            )
            .add_markdown(
                "TRAIN_SPLIT_NUM_ROWS", f"**Number of train dataset rows:** `{len(train_df)}`"
            )
            .add_markdown(
                "VALIDATION_SPLIT_NUM_ROWS",
                f"**Number of validation dataset rows:** `{len(validation_df)}`",
            )
            .add_markdown(
                "TEST_SPLIT_NUM_ROWS", f"**Number of test dataset rows:** `{len(test_df)}`"
            )
        )

        return card

    def _run(self, output_directory):
        import pandas as pd

        run_start_time = time.time()

        # read ingested dataset
        ingested_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="ingest",
            relative_path=_INPUT_FILE_NAME,
        )
        input_df = pd.read_parquet(ingested_data_path)

        # drop rows which target value is missing
        raw_input_num_rows = len(input_df)
        input_df = input_df.dropna(how="any", subset=[self.target_col])
        self.num_dropped_rows = raw_input_num_rows - len(input_df)

        # split dataset
        hash_buckets = _create_hash_buckets(input_df)
        train_df, validation_df, test_df = _get_split_df(input_df, hash_buckets, self.split_ratios)
        # Import from user function module to process dataframes
        post_split_config = self.step_config.get("post_split_method", None)
        if post_split_config is not None:
            (post_split_module_name, post_split_fn_name) = post_split_config.rsplit(".", 1)
            sys.path.append(self.pipeline_root)
            post_split = getattr(
                importlib.import_module(post_split_module_name), post_split_fn_name
            )
            (train_df, validation_df, test_df) = post_split(train_df, validation_df, test_df)

        # Output train / validation / test splits
        train_df.to_parquet(os.path.join(output_directory, _OUTPUT_TRAIN_FILE_NAME))
        validation_df.to_parquet(os.path.join(output_directory, _OUTPUT_VALIDATION_FILE_NAME))
        test_df.to_parquet(os.path.join(output_directory, _OUTPUT_TEST_FILE_NAME))

        self.run_end_time = time.time()
        self.execution_duration = self.run_end_time - run_start_time
        return self._build_profiles_and_card(train_df, validation_df, test_df)

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = pipeline_config.get("steps", {}).get("split", {})
        step_config["target_col"] = pipeline_config.get("target_col")
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "split"
