import pytest

from mlflow.exceptions import MlflowException
from mlflow.pipelines.utils.step import display_html, get_merged_eval_metrics
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


@pytest.mark.parametrize(
    "eval_metrics, ordered_metric_names, expected_metric_names, expected_columns",
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
