import importlib
import logging
import os
import sys
import time
from enum import Enum
from functools import partial
from multiprocessing.pool import Pool, ThreadPool

import numpy as np
import pandas as pd
from packaging.version import Version

from mlflow.exceptions import BAD_REQUEST, INVALID_PARAMETER_VALUE, MlflowException
from mlflow.recipes.artifacts import DataframeArtifact
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.step import get_pandas_data_profiles, validate_classification_config
from mlflow.store.artifact.artifact_repo import _NUM_DEFAULT_CPUS
from mlflow.utils.time import Timer

_logger = logging.getLogger(__name__)


_SPLIT_HASH_BUCKET_NUM = 1000
_INPUT_FILE_NAME = "dataset.parquet"
_OUTPUT_TRAIN_FILE_NAME = "train.parquet"
_OUTPUT_VALIDATION_FILE_NAME = "validation.parquet"
_OUTPUT_TEST_FILE_NAME = "test.parquet"
_USER_DEFINED_SPLIT_STEP_MODULE = "steps.split"
_MAX_CLASSES_TO_PROFILE = 5


class SplitValues(Enum):
    """
    Represents the custom split return values.
    """

    # Indicates that the row is part of test split
    TEST = "TEST"
    # Indicates that the row is part of train split
    TRAINING = "TRAINING"
    # Indicates that the row is part of validation split
    VALIDATION = "VALIDATION"


def _make_elem_hashable(elem):
    if isinstance(elem, list):
        return tuple(_make_elem_hashable(e) for e in elem)
    elif isinstance(elem, dict):
        return tuple((_make_elem_hashable(k), _make_elem_hashable(v)) for k, v in elem.items())
    elif isinstance(elem, np.ndarray):
        return elem.shape, tuple(elem.flatten(order="C"))
    else:
        return elem


def _run_split(task, input_df, split_ratios, target_col):
    if task == "classification":
        return _perform_stratified_split_per_class(input_df, split_ratios, target_col)
    elif task == "regression":
        return _perform_split(input_df, split_ratios)


def _perform_stratified_split_per_class(input_df, split_ratios, target_col):
    classes = np.unique(input_df[target_col])
    partial_func = partial(
        _perform_split_for_one_class,
        input_df=input_df,
        split_ratios=split_ratios,
        target_col=target_col,
    )

    with ThreadPool(os.cpu_count() or _NUM_DEFAULT_CPUS) as p:
        zipped_dfs = p.map(partial_func, classes)
        train_df, validation_df, test_df = [pd.concat(x) for x in list(zip(*zipped_dfs))]
        return train_df, validation_df, test_df


def _perform_split_for_one_class(
    class_value,
    input_df,
    split_ratios,
    target_col,
):
    filtered_df = input_df[input_df[target_col] == class_value]
    return _perform_split(filtered_df, split_ratios, n_jobs=2)


def _perform_split(input_df, split_ratios, n_jobs=-1):
    hash_buckets = _create_hash_buckets(input_df, n_jobs=n_jobs)
    train_df, validation_df, test_df = _get_split_df(input_df, hash_buckets, split_ratios)
    return train_df, validation_df, test_df


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


def _parallelize(data, func, n_jobs=-1):
    n_jobs = n_jobs if 0 < n_jobs <= _NUM_DEFAULT_CPUS else _NUM_DEFAULT_CPUS
    data_split = np.array_split(data, n_jobs)
    pool = Pool(n_jobs)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def _run_on_subset(func, data_subset):
    if Version(pd.__version__) >= Version("2.1.0"):
        return data_subset.map(func)
    return data_subset.applymap(func)


def _parallelize_on_rows(data, func, n_jobs=-1):
    return _parallelize(data, partial(_run_on_subset, func), n_jobs=n_jobs)


def _hash_pandas_dataframe(input_df, n_jobs=-1):
    from pandas.util import hash_pandas_object

    hashed_input_df = _parallelize_on_rows(input_df, _make_elem_hashable, n_jobs=n_jobs)
    return hash_pandas_object(hashed_input_df)


def _create_hash_buckets(input_df, n_jobs=-1):
    # Create hash bucket used for splitting dataset
    # Note: use `hash_pandas_object` instead of python builtin hash because it is stable
    # across different process runs / different python versions
    with Timer() as t:
        hash_buckets = _hash_pandas_dataframe(input_df, n_jobs=n_jobs).map(
            lambda x: (x % _SPLIT_HASH_BUCKET_NUM) / _SPLIT_HASH_BUCKET_NUM
        )
    _logger.debug(
        f"Creating hash buckets on input dataset containing {len(input_df)} "
        f"rows consumes {t:.3f} seconds."
    )
    return hash_buckets


def _validate_user_code_output(post_split, train_df, validation_df, test_df):
    try:
        (
            post_filter_train_df,
            post_filter_validation_df,
            post_filter_test_df,
        ) = post_split(train_df, validation_df, test_df)
    except Exception:
        raise MlflowException(
            message="Error in cleaning up the data frame post split step."
            " Expected output is a tuple with (train_df, validation_df, test_df)"
        ) from None

    for post_split_df, pre_split_df, split_type in [
        [post_filter_train_df, train_df, "train"],
        [post_filter_validation_df, validation_df, "validation"],
        [post_filter_test_df, test_df, "test"],
    ]:
        if not isinstance(post_split_df, pd.DataFrame):
            raise MlflowException(
                message="The split data is not a DataFrame, please return the correct data."
            ) from None
        if list(pre_split_df.columns) != list(post_split_df.columns):
            raise MlflowException(
                message="The number of columns post split step are different."
                f" Column list for {split_type} dataset pre-slit is {list(pre_split_df.columns)}"
                f" and post split is {list(post_split_df.columns)}. "
                "Split filter function should be used to filter rows rather than filtering columns."
            ) from None

    return (
        post_filter_train_df,
        post_filter_validation_df,
        post_filter_test_df,
    )


class SplitStep(BaseStep):
    def _validate_and_apply_step_config(self):
        self.run_end_time = None
        self.execution_duration = None
        self.num_dropped_rows = None

        self.target_col = self.step_config.get("target_col")
        self.positive_class = self.step_config.get("positive_class")
        self.skip_data_profiling = self.step_config.get("skip_data_profiling", False)
        if self.target_col is None:
            raise MlflowException(
                "Missing target_col config in recipe config.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.skip_data_profiling = self.step_config.get("skip_data_profiling", False)

        if "using" in self.step_config:
            if self.step_config["using"] not in ["custom", "split_ratios"]:
                raise MlflowException(
                    f"Invalid split step configuration value {self.step_config['using']} for "
                    f"key 'using'. Supported values are: ['custom', 'split_ratios']",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            self.step_config["using"] = "split_ratios"

        if self.step_config["using"] == "split_ratios":
            self.split_ratios = self.step_config.get("split_ratios", [0.75, 0.125, 0.125])
            if not (
                isinstance(self.split_ratios, list)
                and len(self.split_ratios) == 3
                and all(isinstance(x, (int, float)) and x > 0 for x in self.split_ratios)
            ):
                raise MlflowException(
                    "Config split_ratios must be a list containing 3 positive numbers."
                )

        if "split_method" not in self.step_config and self.step_config["using"] == "custom":
            raise MlflowException(
                "Missing 'split_method' configuration in the split step, "
                "which is using 'custom'.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _build_profiles_and_card(self, train_df, validation_df, test_df) -> BaseCard:
        from sklearn.utils import compute_class_weight

        def _set_target_col_as_first(df, target_col):
            columns = list(df.columns)
            col = columns.pop(columns.index(target_col))
            return df[[col] + columns]

        # Build card
        card = BaseCard(self.recipe_name, self.name)

        if not self.skip_data_profiling:
            # Build profiles for input dataset, and train / validation / test splits
            train_df = _set_target_col_as_first(train_df, self.target_col)
            validation_df = _set_target_col_as_first(validation_df, self.target_col)
            test_df = _set_target_col_as_first(test_df, self.target_col)
            data_profile = get_pandas_data_profiles(
                [
                    ["Train", train_df.reset_index(drop=True)],
                    ["Validation", validation_df.reset_index(drop=True)],
                    ["Test", test_df.reset_index(drop=True)],
                ]
            )

            # Tab #1 - #3: data profiles for train/validation and test.
            card.add_tab("Compare Splits", "{{PROFILE}}").add_pandas_profile(
                "PROFILE", data_profile
            )

            if self.task == "classification":
                if self.positive_class is not None:
                    mask = train_df[self.target_col] == self.positive_class
                    dfs_for_profiles = [
                        ("Positive", train_df[mask]),
                        ("Negative", train_df[~mask]),
                    ]
                    sub_title = "Positive vs Negative"
                else:
                    classes = np.unique(train_df[self.target_col])
                    class_weights = compute_class_weight(
                        class_weight="balanced",
                        classes=classes,
                        y=train_df[self.target_col],
                    )
                    class_weights = list(zip(classes, class_weights))
                    class_weights = sorted(class_weights, key=lambda x: x[1], reverse=True)
                    if len(class_weights) > _MAX_CLASSES_TO_PROFILE:
                        class_weights = class_weights[:_MAX_CLASSES_TO_PROFILE]
                    dfs_for_profiles = [
                        (name, train_df[(train_df[self.target_col] == name)])
                        for name, _ in class_weights
                    ]
                    sub_title = f"Top {min(5, len(class_weights))} Classes"

                profiles = [
                    [
                        str(p[0]),
                        p[1].drop(columns=[self.target_col]).reset_index(drop=True),
                    ]
                    for p in dfs_for_profiles
                ]
                generated_profile = get_pandas_data_profiles(profiles)
                # Tab #4: data profiles positive negative training split.
                card.add_tab(
                    f"Compare Training Data ({sub_title})", "{{PROFILE}}"
                ).add_pandas_profile("PROFILE", generated_profile)

        # Tab #5: run summary.
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

    def _validate_and_execute_custom_split(self, split_fn, input_df):
        custom_split_mapping_series = split_fn(input_df)
        if not isinstance(custom_split_mapping_series, pd.Series):
            raise MlflowException(
                "Return type of the custom split function should be a pandas series",
                error_code=INVALID_PARAMETER_VALUE,
            )

        copy_df = input_df.copy()
        copy_df["split"] = custom_split_mapping_series
        train_df = input_df[copy_df["split"] == SplitValues.TRAINING.value].reset_index(drop=True)
        validation_df = input_df[copy_df["split"] == SplitValues.VALIDATION.value].reset_index(
            drop=True
        )
        test_df = input_df[copy_df["split"] == SplitValues.TEST.value].reset_index(drop=True)

        if train_df.size + validation_df.size + test_df.size != input_df.size:
            incorrect_args = custom_split_mapping_series[
                ~custom_split_mapping_series.isin(
                    [
                        SplitValues.TRAINING.value,
                        SplitValues.VALIDATION.value,
                        SplitValues.TEST.value,
                    ]
                )
            ].unique()
            raise MlflowException(
                f"Returned pandas series from custom split step should only contain "
                f"{SplitValues.TRAINING.value}, {SplitValues.VALIDATION.value} or "
                f"{SplitValues.TEST.value} as values. Value returned back: {incorrect_args}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return train_df, validation_df, test_df

    def _run_custom_split(self, input_df):
        split_fn = getattr(
            importlib.import_module(_USER_DEFINED_SPLIT_STEP_MODULE),
            self.step_config["split_method"],
        )
        return self._validate_and_execute_custom_split(split_fn, input_df)

    def _run(self, output_directory):
        run_start_time = time.time()

        # read ingested dataset
        ingested_data_path = get_step_output_path(
            recipe_root_path=self.recipe_root,
            step_name="ingest",
            relative_path=_INPUT_FILE_NAME,
        )
        input_df = pd.read_parquet(ingested_data_path)
        validate_classification_config(self.task, self.positive_class, input_df, self.target_col)

        # drop rows which target value is missing
        raw_input_num_rows = len(input_df)
        # Make sure the target column is actually present in the input DF.
        if self.target_col not in input_df.columns:
            raise MlflowException(
                f"Target column '{self.target_col}' not found in ingested dataset.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        input_df = input_df.dropna(how="any", subset=[self.target_col])
        self.num_dropped_rows = raw_input_num_rows - len(input_df)

        # split dataset
        if self.step_config["using"] == "custom":
            train_df, validation_df, test_df = self._run_custom_split(input_df)
        else:
            train_df, validation_df, test_df = _run_split(
                self.task, input_df, self.split_ratios, self.target_col
            )
        # Import from user function module to process dataframes
        post_split_config = self.step_config.get("post_split_method", None)
        post_split_filter_config = self.step_config.get("post_split_filter_method", None)
        if post_split_config is not None:
            sys.path.append(self.recipe_root)
            post_split = getattr(
                importlib.import_module(_USER_DEFINED_SPLIT_STEP_MODULE), post_split_config
            )
            _logger.debug(f"Running {post_split_config} on train, validation and test datasets.")
            (
                train_df,
                validation_df,
                test_df,
            ) = _validate_user_code_output(post_split, train_df, validation_df, test_df)

        elif post_split_filter_config is not None:
            sys.path.append(self.recipe_root)
            post_split_filter = getattr(
                importlib.import_module(_USER_DEFINED_SPLIT_STEP_MODULE), post_split_filter_config
            )
            _logger.debug(
                f"Running {post_split_filter_config} on train, validation and test datasets."
            )
            train_df = train_df[post_split_filter(train_df)]

        if min(len(train_df), len(validation_df), len(test_df)) < 4:
            raise MlflowException(
                f"Train, validation, and testing datasets cannot be less than 4 rows. Train has "
                f"{len(train_df)} rows, validation has {len(validation_df)} rows, and test has "
                f"{len(test_df)} rows.",
                error_code=BAD_REQUEST,
            )
        # Output train / validation / test splits
        train_df.to_parquet(os.path.join(output_directory, _OUTPUT_TRAIN_FILE_NAME))
        validation_df.to_parquet(os.path.join(output_directory, _OUTPUT_VALIDATION_FILE_NAME))
        test_df.to_parquet(os.path.join(output_directory, _OUTPUT_TEST_FILE_NAME))

        self.run_end_time = time.time()
        self.execution_duration = self.run_end_time - run_start_time
        return self._build_profiles_and_card(train_df, validation_df, test_df)

    @classmethod
    def from_recipe_config(cls, recipe_config, recipe_root):
        step_config = {}
        if recipe_config.get("steps", {}).get("split", {}) is not None:
            step_config.update(recipe_config.get("steps", {}).get("split", {}))
        step_config["target_col"] = recipe_config.get("target_col")
        step_config["positive_class"] = recipe_config.get("positive_class")
        step_config["recipe"] = recipe_config.get("recipe", "regression/v1")
        return cls(step_config, recipe_root)

    @property
    def name(self):
        return "split"

    def get_artifacts(self):
        return [
            DataframeArtifact(
                "training_data", self.recipe_root, self.name, _OUTPUT_TRAIN_FILE_NAME
            ),
            DataframeArtifact(
                "validation_data", self.recipe_root, self.name, _OUTPUT_VALIDATION_FILE_NAME
            ),
            DataframeArtifact("test_data", self.recipe_root, self.name, _OUTPUT_TEST_FILE_NAME),
        ]

    def step_class(self):
        return StepClass.TRAINING
