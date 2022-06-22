import os
import pytest

import numpy as np
import pandas as pd

from mlflow.exceptions import MlflowException
from mlflow.pipelines.utils.execution import _MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR
from mlflow.pipelines.steps.split import _get_split_df, _make_elem_hashable, SplitStep
from unittest import mock


def test_split_step_run(tmp_path):
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

    split_ratios = [0.6, 0.3, 0.1]

    with mock.patch.dict(
        os.environ, {_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_path)}
    ), mock.patch("mlflow.pipelines.step.get_pipeline_name", return_value="fake_name"):
        split_step = SplitStep({"split_ratios": split_ratios, "target_col": "y"}, "fake_root")
        split_step._run(str(split_output_dir))

    (split_output_dir / "summary.html").exists()
    (split_output_dir / "card.html").exists()

    output_train_df = pd.read_parquet(str(split_output_dir / "train.parquet"))
    output_validation_df = pd.read_parquet(str(split_output_dir / "validation.parquet"))
    output_test_df = pd.read_parquet(str(split_output_dir / "test.parquet"))

    assert np.isclose(len(output_train_df) / num_good_rows, split_ratios[0], rtol=0.1)
    assert np.isclose(len(output_validation_df) / num_good_rows, split_ratios[1], rtol=0.1)
    assert np.isclose(len(output_test_df) / num_good_rows, split_ratios[2], rtol=0.1)

    merged_output_df = pd.concat([output_train_df, output_validation_df, output_test_df])
    assert merged_output_df.columns.tolist() == ["a", "b", "y"]
    assert set(merged_output_df.a.tolist()) == set(range(num_good_rows))
    assert set(merged_output_df.b.tolist()) == set(str(i) for i in range(num_good_rows))
    assert set(merged_output_df.y.tolist()) == {0.0, 1.0}


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


def test_get_split_df():
    with mock.patch("mlflow.pipelines.steps.split._SPLIT_HASH_BUCKET_NUM", 6):
        split_ratios = [3, 2, 1]
        hash_buckets = pd.Series([0.3, 0.9, 0.1, 0.7, 0.2, 0.6])
        dataset = pd.DataFrame({"v": [10, 20, 30, 40, 50, 60]})

        train_df, val_df, test_df = _get_split_df(dataset, hash_buckets, split_ratios)

        assert train_df.v.tolist() == [10, 30, 50]
        assert val_df.v.tolist() == [40, 60]
        assert test_df.v.tolist() == [20]


def test_from_pipeline_config_fails_without_target_col(tmp_path):
    with mock.patch.dict(
        os.environ, {_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_path)}
    ), mock.patch(
        "mlflow.pipelines.step.get_pipeline_name", return_value="fake_name"
    ), pytest.raises(
        MlflowException, match="Missing target_col config"
    ):
        _ = SplitStep.from_pipeline_config({}, "fake_root")


def test_from_pipeline_config_works_with_target_col(tmp_path):
    with mock.patch.dict(
        os.environ, {_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_path)}
    ), mock.patch("mlflow.pipelines.step.get_pipeline_name", return_value="fake_name"):
        assert SplitStep.from_pipeline_config({"target_col": "fake_col"}, "fake_root") is not None
