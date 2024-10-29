from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import mlflow
from mlflow.environment_variables import MLFLOW_RECIPES_EXECUTION_DIRECTORY
from mlflow.exceptions import MlflowException
from mlflow.recipes.steps.split import (
    _OUTPUT_TEST_FILE_NAME,
    _OUTPUT_TRAIN_FILE_NAME,
    _OUTPUT_VALIDATION_FILE_NAME,
    SplitStep,
    SplitValues,
    _get_split_df,
    _hash_pandas_dataframe,
    _make_elem_hashable,
    _validate_user_code_output,
)
from mlflow.recipes.utils import _RECIPE_CONFIG_FILE_NAME
from mlflow.utils.file_utils import read_yaml


def set_up_dataset(tmp_path, num_classes=2):
    ingest_output_dir = tmp_path / "steps" / "ingest" / "outputs"
    ingest_output_dir.mkdir(parents=True)
    split_output_dir = tmp_path / "steps" / "split" / "outputs"
    split_output_dir.mkdir(parents=True)
    num_rows = 1000
    num_good_rows = 900
    input_dataframe = pd.DataFrame(
        {
            "a": list(range(num_rows)),
            "b": [str(i) for i in range(num_rows)],
            "y": [float(i % num_classes) if i < num_good_rows else None for i in range(num_rows)],
        }
    )
    input_dataframe.to_parquet(str(ingest_output_dir / "dataset.parquet"))
    return num_good_rows, split_output_dir, input_dataframe


def test_split_step_run(tmp_path, monkeypatch):
    num_good_rows, split_output_dir, _ = set_up_dataset(tmp_path)

    split_ratios = [0.6, 0.3, 0.1]

    monkeypatch.setenv(MLFLOW_RECIPES_EXECUTION_DIRECTORY.name, str(tmp_path))
    with mock.patch("mlflow.recipes.step.get_recipe_name", return_value="fake_name"):
        split_step = SplitStep(
            {"split_ratios": split_ratios, "target_col": "y", "recipe": "classification/v1"},
            "fake_root",
        )
        split_step.run(str(split_output_dir))

    (split_output_dir / "summary.html").exists()

    split_card_file_path = split_output_dir / "card.html"
    split_card_file_path.exists()

    with open(split_card_file_path, errors="ignore") as f:
        step_card_content = f.read()

    assert "Compare Training Data" in step_card_content

    output_train_df = pd.read_parquet(str(split_output_dir / "train.parquet"))
    output_validation_df = pd.read_parquet(str(split_output_dir / "validation.parquet"))
    output_test_df = pd.read_parquet(str(split_output_dir / "test.parquet"))

    assert len(output_train_df) == 551
    assert len(output_validation_df) == 266
    assert len(output_test_df) == 83

    merged_output_df = pd.concat([output_train_df, output_validation_df, output_test_df])
    assert merged_output_df.columns.tolist() == ["a", "b", "y"]
    assert set(merged_output_df.a.tolist()) == set(range(num_good_rows))
    assert set(merged_output_df.b.tolist()) == {str(i) for i in range(num_good_rows)}
    assert set(merged_output_df.y.tolist()) == {0.0, 1.0}


def test_split_step_run_with_multiple_classes(tmp_path, monkeypatch):
    num_good_rows, split_output_dir, _ = set_up_dataset(tmp_path, num_classes=10)

    split_ratios = [0.6, 0.3, 0.1]

    monkeypatch.setenv(MLFLOW_RECIPES_EXECUTION_DIRECTORY.name, str(tmp_path))
    with mock.patch("mlflow.recipes.step.get_recipe_name", return_value="fake_name"):
        split_step = SplitStep(
            {"split_ratios": split_ratios, "target_col": "y", "recipe": "classification/v1"},
            "fake_root",
        )
        split_step.run(str(split_output_dir))

    output_train_df = pd.read_parquet(str(split_output_dir / "train.parquet"))
    output_validation_df = pd.read_parquet(str(split_output_dir / "validation.parquet"))
    output_test_df = pd.read_parquet(str(split_output_dir / "test.parquet"))

    assert len(output_train_df) + len(output_validation_df) + len(output_test_df) == num_good_rows

    merged_output_df = pd.concat([output_train_df, output_validation_df, output_test_df])
    assert merged_output_df.columns.tolist() == ["a", "b", "y"]
    assert set(merged_output_df.a.tolist()) == set(range(num_good_rows))
    assert set(merged_output_df.b.tolist()) == {str(i) for i in range(num_good_rows)}


def test_make_elem_hashable():
    assert _make_elem_hashable(3) == 3
    assert _make_elem_hashable("abc") == "abc"
    assert _make_elem_hashable((2, 3)) == (2, 3)
    assert _make_elem_hashable([2, 3]) == (2, 3)
    assert _make_elem_hashable([[2, 3], [4, 5]]) == ((2, 3), (4, 5))
    assert _make_elem_hashable({"a": 2, "b": 3}) == (("a", 2), ("b", 3))
    assert _make_elem_hashable({"a": [2, 3]}) == (("a", (2, 3)),)
    assert _make_elem_hashable(np.array([2, 3])) == ((2,), (2, 3))
    assert _make_elem_hashable(np.array([[2, 3, 4], [5, 6, 7]])) == ((2, 3), (2, 3, 4, 5, 6, 7))


def test_hash_pandas_dataframe_deterministic():
    pdf = pd.DataFrame(
        {
            "f1": [2, 3],
            "f2": [2.5, 3.5],
            "f3": [[6, 7], [8, 9]],
            "f4": [np.array([12, 13]), np.array([14, 15])],
            "f5": [np.array([12.5, 13.5]), np.array([14.5, 15.5])],
            "f6": [np.array([[22], [23]]), np.array([[24], [25]])],
            "f7": ["abc", "def"],
            "f8": [b"ghi", b"jkl"],
            "f9": [{"a1": 32, "b1": 33}, {"a2": 34, "b2": 35}],
            "f10": [{"a3": [42, 43]}, {"a4": [44, 45]}],
            "f11": pd.Series(["a5", "a6"], dtype="category"),
            "f13": [pd.Timestamp("2017-01-01T12"), pd.Timestamp("2017-02-01T11")],
            "f14": [True, False],
            "f15": [pd.Period("2000-01-02", freq="D"), pd.Period("2001-01-02", freq="M")],
            "f16": [pd.Interval(22.5, 23.5), pd.Interval(24.5, 25.5)],
        }
    )
    result = _hash_pandas_dataframe(pdf)
    assert result.tolist() == [17628718130518205164, 1442560710956928490]


def test_get_split_df():
    with mock.patch("mlflow.recipes.steps.split._SPLIT_HASH_BUCKET_NUM", 6):
        split_ratios = [3, 2, 1]
        hash_buckets = pd.Series([0.3, 0.9, 0.1, 0.7, 0.2, 0.6])
        dataset = pd.DataFrame({"v": [10, 20, 30, 40, 50, 60]})

        train_df, val_df, test_df = _get_split_df(dataset, hash_buckets, split_ratios)

        assert train_df.v.tolist() == [10, 30, 50]
        assert val_df.v.tolist() == [40, 60]
        assert test_df.v.tolist() == [20]


def test_from_recipe_config_fails_without_target_col(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_RECIPES_EXECUTION_DIRECTORY.name, str(tmp_path))
    with mock.patch("mlflow.recipes.step.get_recipe_name", return_value="fake_name"):
        split_step = SplitStep.from_recipe_config({}, "fake_root")
        with pytest.raises(MlflowException, match="Missing target_col config"):
            split_step._validate_and_apply_step_config()


def test_from_recipe_config_works_with_target_col(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_RECIPES_EXECUTION_DIRECTORY.name, str(tmp_path))
    with mock.patch("mlflow.recipes.step.get_recipe_name", return_value="fake_name"):
        assert SplitStep.from_recipe_config({"target_col": "fake_col"}, "fake_root") is not None


def test_split_step_skips_profiling_when_specified(tmp_path, monkeypatch):
    ingest_output_dir = tmp_path / "steps" / "ingest" / "outputs"
    ingest_output_dir.mkdir(parents=True)
    split_output_dir = tmp_path / "steps" / "split" / "outputs"
    split_output_dir.mkdir(parents=True)

    num_rows = 1000
    num_good_rows = 900
    input_dataframe = pd.DataFrame(
        {
            "a": list(range(num_rows)),
            "b": [str(i) for i in range(num_rows)],
            "y": [float(i % 2) if i < num_good_rows else None for i in range(num_rows)],
        }
    )
    input_dataframe.to_parquet(str(ingest_output_dir / "dataset.parquet"))

    monkeypatch.setenv(MLFLOW_RECIPES_EXECUTION_DIRECTORY.name, str(tmp_path))
    with mock.patch(
        "mlflow.recipes.utils.step.get_pandas_data_profiles"
    ) as mock_profiling, mock.patch(
        "mlflow.recipes.step.get_recipe_name", return_value="fake_name"
    ):
        split_step = SplitStep({"target_col": "y", "skip_data_profiling": True}, "fake_root")
        split_step.run(str(split_output_dir))

    mock_profiling.assert_not_called()


def test_validation_split_step_validates_split_correctly():
    train_df = pd.DataFrame({"v": [10, 20, 30], "w": [1, 2, 3]})
    validation_df = pd.DataFrame({"v": [40, 50, 60], "w": [4, 5, 6]})
    test_df = pd.DataFrame({"v": [70, 80, 90], "w": [7, 8, 9]})

    def correct_post_split(train_df, validation_df, test_df):
        return (train_df, validation_df, test_df)

    (out_train_df, out_validation_df, out_test_df) = _validate_user_code_output(
        correct_post_split, train_df, validation_df, test_df
    )

    assert train_df.equals(out_train_df)
    assert validation_df.equals(out_validation_df)
    assert test_df.equals(out_test_df)

    def drop_post_split(train_df, validation_df, test_df):
        train_df = train_df.drop(columns=["w"])
        return (train_df, validation_df, test_df)

    with pytest.raises(
        MlflowException,
        match="Column list for train dataset pre-slit .* and post split is .*",
    ):
        (out_train_df, out_validation_df, out_test_df) = _validate_user_code_output(
            drop_post_split, train_df, validation_df, test_df
        )

    def incorrect_post_split(_, validation_df, test_df):
        return ([], validation_df, test_df)

    with pytest.raises(
        MlflowException,
        match="The split data is not a DataFrame, please return the correct data.",
    ):
        (out_train_df, out_validation_df, out_test_df) = _validate_user_code_output(
            incorrect_post_split, train_df, validation_df, test_df
        )


def custom_split(df):
    from pandas import Series

    splits = Series(SplitValues.TRAINING.value, index=range(len(df)))
    splits[df["a"] >= 500] = SplitValues.VALIDATION.value
    splits[df["a"] >= 800] = SplitValues.TEST.value

    return splits


def test_custom_split_method(tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path):
    ingest_output_dir = tmp_recipe_exec_path / "steps" / "ingest" / "outputs"
    ingest_output_dir.mkdir(parents=True)
    split_output_dir = tmp_recipe_exec_path / "steps" / "split" / "outputs"
    split_output_dir.mkdir(parents=True)
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True)

    num_rows = 1000
    input_dataframe = pd.DataFrame(
        {
            "a": list(range(num_rows)),
            "y": [float(i % 2) for i in range(num_rows)],
        }
    )
    input_dataframe.to_parquet(str(ingest_output_dir / "dataset.parquet"))

    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        f"""
        recipe: "regression/v1"
        target_col: "y"
        primary_metric: "f1_score"
        profile: "test_profile"
        run_args:
            step: "split"
        experiment:
            name: "demo"
            tracking_uri: {mlflow.get_tracking_uri()}
        steps:
            split:
                using: custom
                split_method: split_method
        """
    )

    m_split = Mock()
    m_split.split_method = custom_split

    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    split_step = SplitStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    with mock.patch.dict("sys.modules", {"steps.split": m_split}):
        split_step.run(str(split_output_dir))

    train_dataframe = pd.read_parquet(split_output_dir / _OUTPUT_TRAIN_FILE_NAME)
    validation_dataframe = pd.read_parquet(split_output_dir / _OUTPUT_VALIDATION_FILE_NAME)
    test_dataframe = pd.read_parquet(split_output_dir / _OUTPUT_TEST_FILE_NAME)
    assert len(train_dataframe.index) == 500
    assert len(validation_dataframe.index) == 300
    assert len(test_dataframe.index) == 200


def custom_error_split(df):
    from pandas import Series

    splits = Series(SplitValues.TRAINING.value, index=range(len(df)))
    splits[df["a"] >= 500] = "VALIDATE"
    splits[df["a"] >= 800] = "TESTING"

    return splits


def test_custom_error_split_method(tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path):
    ingest_output_dir = tmp_recipe_exec_path / "steps" / "ingest" / "outputs"
    ingest_output_dir.mkdir(parents=True)
    split_output_dir = tmp_recipe_exec_path / "steps" / "split" / "outputs"
    split_output_dir.mkdir(parents=True)
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True)

    num_rows = 1000
    input_dataframe = pd.DataFrame(
        {
            "a": list(range(num_rows)),
            "y": [float(i % 2) for i in range(num_rows)],
        }
    )
    input_dataframe.to_parquet(str(ingest_output_dir / "dataset.parquet"))

    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        f"""
        recipe: "regression/v1"
        target_col: "y"
        primary_metric: "f1_score"
        profile: "test_profile"
        run_args:
            step: "split"
        experiment:
            name: "demo"
            tracking_uri: {mlflow.get_tracking_uri()}
        steps:
            split:
                using: custom
                split_method: split_method
        """
    )

    m_split = Mock()
    m_split.split_method = custom_error_split

    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    split_step = SplitStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    with mock.patch.dict("sys.modules", {"steps.split": m_split}), pytest.raises(
        MlflowException,
        match=r"Value returned back: \['VALIDATE' 'TESTING'\]",
    ):
        split_step.run(str(split_output_dir))
