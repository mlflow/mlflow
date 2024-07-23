import logging
import os
import shutil
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from mlflow.exceptions import BAD_REQUEST, INVALID_PARAMETER_VALUE, MlflowException
from mlflow.recipes.cards import pandas_renderer
from mlflow.utils.databricks_utils import (
    get_databricks_runtime_version,
    is_in_databricks_runtime,
    is_running_in_ipython_environment,
)
from mlflow.utils.os import is_windows

_logger = logging.getLogger(__name__)

_MAX_PROFILE_CELL_SIZE = 10000000  # 10M Cells
_MAX_PROFILE_ROW_SIZE = 1000000  # 1M Rows
_MAX_PROFILE_COL_SIZE = 10000  # 10k Cols


def get_merged_eval_metrics(
    eval_metrics: Dict[str, Dict], ordered_metric_names: Optional[List[str]] = None
):
    """
    Returns a merged Pandas DataFrame from a map of dataset to evaluation metrics.
    Optionally, the rows in the DataFrame are ordered by input ordered metric names.

    Args:
        eval_metrics: Dict maps from dataset name to a Dict of evaluation metrics, which itself
            is a map from metric name to metric value.
        ordered_metric_names: List containing metric names. The ordering of the output is
            determined by this list, if provided.

    Returns:
        Pandas DataFrame containing evaluation metrics. The DataFrame is indexed by metric
        name. Columns are dataset names.
    """
    from pandas import DataFrame

    merged_metrics = {}
    for src, metrics in eval_metrics.items():
        if src not in merged_metrics:
            merged_metrics[src] = {}
        merged_metrics[src].update(metrics)

    if ordered_metric_names is None:
        ordered_metric_names = []

    metric_names = set()
    for val in merged_metrics.values():
        metric_names.update(val.keys())
    missing_metrics = set(ordered_metric_names) - metric_names
    if len(missing_metrics) > 0:
        _logger.warning(
            "Input metric names %s not found in eval metrics: %s", missing_metrics, metric_names
        )
        ordered_metric_names = [
            name for name in ordered_metric_names if name not in missing_metrics
        ]
    ordered_metric_names.extend(sorted(metric_names - set(ordered_metric_names)))
    return DataFrame.from_dict(merged_metrics).reindex(ordered_metric_names)


def display_html(html_data: Optional[str] = None, html_file_path: Optional[str] = None) -> None:
    if html_file_path is None and html_data is None:
        raise MlflowException(
            "At least one HTML source must be provided. html_data and html_file_path are empty.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if is_running_in_ipython_environment():
        from IPython.display import HTML
        from IPython.display import display as ip_display

        html_file_path = html_file_path if html_data is None else None

        if is_in_databricks_runtime():
            dbr_version = get_databricks_runtime_version()
            if int(dbr_version.split(".")[0]) < 11:
                raise MlflowException(
                    f"Use Databricks Runtime 11 or newer with MLflow Recipes. "
                    f"Current version is {dbr_version} ",
                    error_code=BAD_REQUEST,
                )
            # Patch IPython display with Databricks display before showing the HTML.
            import IPython.core.display as icd

            orig_display = icd.display
            icd.display = display  # noqa: F821
            ip_display(HTML(data=html_data, filename=html_file_path))
            icd.display = orig_display
        else:
            ip_display(HTML(data=html_data, filename=html_file_path))
    else:
        # Use xdg-open in Linux environment
        if shutil.which("xdg-open") is not None:
            open_tool = shutil.which("xdg-open")
        elif shutil.which("open") is not None:
            open_tool = shutil.which("open")
        else:
            open_tool = None

        if (
            os.path.exists(html_file_path)
            and open_tool is not None
            # On Windows, attempting to clean up the card while it's being accessed by
            # the process running `open_tool` results in a PermissionError. To avoid this,
            # skip displaying the card.
            and "GITHUB_ACTIONS" not in os.environ
        ):
            _logger.info(f"Opening HTML file at: '{html_file_path}'")
            try:
                subprocess.run([open_tool, html_file_path], check=True)
            except Exception as e:
                _logger.warning(
                    f"Encountered unexpected error opening the html page."
                    f" The file may be manually accessed at {html_file_path}. Exception: {e}"
                )


# Prevent pandas_profiling from using multiprocessing on Windows while running tests.
# multiprocessing and pytest don't play well together on Windows.
# Relevant code: https://github.com/ydataai/pandas-profiling/blob/f8bad5dde27e3f87f11ac74fb8966c034bc22db8/src/pandas_profiling/model/pandas/summary_pandas.py#L76-L97
def _get_pool_size():
    return 1 if "PYTEST_CURRENT_TEST" in os.environ and is_windows() else 0


def get_pandas_data_profiles(inputs: Iterable[Tuple[str, pd.DataFrame]]) -> str:
    """
    Returns a data profiling string over input data frame.

    Args:
        inputs: Either a single "glimpse" DataFrame that contains the statistics, or a
            collection of (title, DataFrame) pairs where each pair names a separate "glimpse"
            and they are all visualized in comparison mode.

    Returns:
        a data profiling string such as Pandas profiling ProfileReport.
    """
    truncated_input = [truncate_pandas_data_profile(*input) for input in inputs]
    return pandas_renderer.get_html(truncated_input)


def truncate_pandas_data_profile(title: str, data_frame) -> str:
    """
    Returns a data profiling string over input data frame.

    Args:
        title: The title of the data profile.
        data_frame: Contains data to be profiled.

    Returns:
        A data profiling string such as Pandas profiling ProfileReport.
    """
    if len(data_frame) == 0:
        return (title, data_frame)

    max_cells = min(data_frame.size, _MAX_PROFILE_CELL_SIZE)
    max_cols = min(data_frame.columns.size, _MAX_PROFILE_COL_SIZE)
    max_rows = min(max(max_cells // max_cols, 1), len(data_frame), _MAX_PROFILE_ROW_SIZE)
    truncated_df = data_frame.drop(columns=data_frame.columns[max_cols:]).sample(
        n=max_rows, ignore_index=True, random_state=42
    )
    if (
        max_cells == _MAX_PROFILE_CELL_SIZE
        or max_cols == _MAX_PROFILE_COL_SIZE
        or max_rows == _MAX_PROFILE_ROW_SIZE
    ):
        _logger.info(
            "Truncating the data frame for %s to %d cells, %d columns and %d rows",
            title,
            max_cells,
            max_cols,
            max_rows,
        )
    return (title, truncated_df)


def validate_classification_config(
    task: str, positive_class: str, input_df: pd.DataFrame, target_col: str
):
    """
    Args:
        task:
        positive_class:
        input_df:
        target_col:
    """
    if task == "classification":
        classes = np.unique(input_df[target_col])
        num_classes = len(classes)
        if num_classes <= 1:
            raise MlflowException(
                f"Classification tasks require at least two tasks. "
                f"Your dataset contains {num_classes}."
            )
        elif positive_class is None and num_classes == 2:
            raise MlflowException(
                "`positive_class` must be specified for classification/v1 recipes.",
                error_code=INVALID_PARAMETER_VALUE,
            )
