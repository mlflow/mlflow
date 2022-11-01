import mlflow.pipelines.utils.step as step_utils
import numpy as np
import pytest

from mlflow.exceptions import MlflowException
from mlflow.pipelines.cards import pandas_renderer
from mlflow.pipelines.utils.step import (
    display_html,
    get_merged_eval_metrics,
    truncate_pandas_data_profile,
    get_pandas_data_profiles,
)
from pandas import DataFrame
from unittest import mock


def test_display_html_raises_without_input():
    with pytest.raises(MlflowException, match="At least one HTML source must be provided"):
        display_html()


def test_display_html_opens_html_data():
    html_data = "<!DOCTYPE html><html><body><p>Hey</p></body></html>"
    with mock.patch(
        "mlflow.pipelines.utils.step.is_running_in_ipython_environment", return_value=True
    ):
        with mock.patch("IPython.display.display") as patched_display:
            display_html(html_data=html_data)
            patched_display.assert_called_once()


def test_display_html_opens_html_file(tmp_path):
    html_file = tmp_path / "test.html"
    html_file.write_text("<!DOCTYPE html><html><body><p>Hey</p></body></html>")
    with mock.patch("subprocess.run") as patched_subprocess, mock.patch(
        "shutil.which", return_value=True
    ):
        display_html(html_file_path=html_file)
        patched_subprocess.assert_called_once()


def test_display_html_throws_error_on_old_dbr():
    html_data = "<!DOCTYPE html><html><body><p>Hey</p></body></html>"
    with mock.patch(
        "mlflow.pipelines.utils.step.is_running_in_ipython_environment", return_value=True
    ), mock.patch(
        "mlflow.pipelines.utils.step.is_in_databricks_runtime", return_value=True
    ), mock.patch(
        "mlflow.pipelines.utils.step.get_databricks_runtime",
        return_value="10.4.x-snapshot-cpu-ml-scala2.12",
    ), pytest.raises(
        MlflowException, match="Use Databricks Runtime 11 or newer with MLflow Pipelines"
    ):
        display_html(html_data=html_data)


@pytest.mark.parametrize(
    ("eval_metrics", "ordered_metric_names", "expected_metric_names", "expected_columns"),
    [
        ({"src1": {}}, None, [], ["src1"]),
        ({"src1": {}, "src2": {"m1": 1}}, None, ["m1"], ["src1", "src2"]),
        (
            {"src1": {"m1": 1, "m2": 2, "m3": 3}, "src2": {"m1": 10, "m3": 30, "m4": 40}},
            None,
            ["m1", "m2", "m3", "m4"],
            ["src1", "src2"],
        ),
        (
            {"src1": {"m1": 1, "m2": 2, "m3": 3}, "src2": {"m1": 10, "m3": 30, "m4": 40}},
            ["m4", "m3", "M100"],
            ["m4", "m3", "m1", "m2"],
            ["src1", "src2"],
        ),
    ],
)
def test_get_merged_eval_metrics_works(
    eval_metrics, ordered_metric_names, expected_metric_names, expected_columns
):
    merged_df = get_merged_eval_metrics(eval_metrics, ordered_metric_names=ordered_metric_names)
    assert list(merged_df.index) == expected_metric_names
    assert list(merged_df.columns) == expected_columns


@pytest.mark.parametrize(
    ("data_frame", "max_cells", "max_cols", "max_rows", "expected_cols", "expected_rows"),
    [
        (DataFrame(np.arange(160).reshape(8, 20)), 1000, 1, 1, 1, 1),
        (DataFrame(np.arange(160).reshape(8, 20)), 1000, 5, 4, 5, 4),
        (DataFrame(np.arange(160).reshape(8, 20)), 1000, 20, 4, 20, 4),
        (DataFrame(np.arange(160).reshape(8, 20)), 1000, 5, 8, 5, 8),
        (DataFrame(np.arange(160).reshape(8, 20)), 1000, 20, 8, 20, 8),
        (DataFrame(np.arange(160).reshape(8, 20)), 1000, 21, 8, 20, 8),
        (DataFrame(np.arange(160).reshape(8, 20)), 1000, 20, 9, 20, 8),
        (DataFrame(np.arange(160).reshape(8, 20)), 10, 5, 9, 5, 2),
        (DataFrame(np.arange(160).reshape(8, 20)), 10, 5, 1, 5, 1),
        (DataFrame(np.arange(160).reshape(8, 20)), 10, 30, 1, 20, 1),
        (DataFrame(np.arange(160).reshape(2, 80)), 80, 8, 10, 8, 2),
    ],
)
def test_get_data_profile_truncates_large_data_frame(
    data_frame, max_cells, max_cols, max_rows, expected_cols, expected_rows
):
    step_utils._MAX_PROFILE_CELL_SIZE = max_cells
    step_utils._MAX_PROFILE_COL_SIZE = max_cols
    step_utils._MAX_PROFILE_ROW_SIZE = max_rows
    with mock.patch.object(pandas_renderer, "get_html") as mock_pandas_renderer_html:
        truncated_df = truncate_pandas_data_profile("fake profile", data_frame)
        assert truncated_df[1].shape == (expected_rows, expected_cols)
        get_pandas_data_profiles([["fake profile", data_frame]])
        # Initial index of [0][0][0] are from call_args_list to get the 0th call.
        # The next [0][1] are because of the pandas_renderer get_html API
        assert mock_pandas_renderer_html.call_args_list[0][0][0][0][1].shape == (
            expected_rows,
            expected_cols,
        )


def test_get_data_profile_works_for_empty_data_frame():
    with mock.patch.object(pandas_renderer, "get_html") as mock_pandas_renderer_html:
        data_frame = DataFrame()
        truncated_df = truncate_pandas_data_profile("fake profile", data_frame)
        assert truncated_df[1].shape == (0, 0)
        get_pandas_data_profiles([["fake profile", data_frame]])
        # Initial index of [0][0][0] are from call_args_list to get the 0th call.
        # The next [0][1] are because of the pandas_renderer get_html API
        assert mock_pandas_renderer_html.call_args_list[0][0][0][0][1].shape == (0, 0)
