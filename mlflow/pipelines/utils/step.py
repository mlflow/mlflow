import logging
import os

from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.utils.databricks_utils import (
    is_running_in_ipython_environment,
    is_in_databricks_runtime,
)
from typing import Dict, List

_logger = logging.getLogger(__name__)


def get_merged_eval_metrics(eval_metrics: Dict[str, Dict], ordered_metric_names: List[str] = None):
    """Returns a merged Pandas DataFrame from a map of dataset to evaluation metrics.
    Optionally, the rows in the DataFrame are ordered by input ordered metric names.

    :param eval_metrics: Dict maps from dataset name to a Dict of evaluation metrics, which itself
                         is a map from metric name to metric value.
    :param ordered_metric_names: List containing metric names. The ordering of the output is
                                 determined by this list, if provided.
    :return: Pandas DataFrame containing evaluation metrics. The DataFrame is indexed by metric
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
    ordered_metric_names.extend(sorted(list(metric_names - set(ordered_metric_names))))
    return DataFrame.from_dict(merged_metrics).reindex(ordered_metric_names)


def display_html(html_data: str = None, html_file_path: str = None) -> None:
    if html_file_path is None and html_data is None:
        raise MlflowException(
            "At least one HTML source must be provided. html_data and html_file_path are empty.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if is_running_in_ipython_environment():
        from IPython.display import display as ip_display, HTML

        if is_in_databricks_runtime():
            # Patch IPython display with Databricks display before showing the HTML.
            import IPython.core.display as icd

            orig_display = icd.display
            icd.display = display  # pylint: disable=undefined-variable
            ip_display(HTML(data=html_data, filename=html_file_path))
            icd.display = orig_display
        else:
            ip_display(HTML(data=html_data, filename=html_file_path))
    else:
        import shutil
        import subprocess

        if os.path.exists(html_file_path) and shutil.which("open") is not None:
            _logger.info(f"Opening HTML file at: '{html_file_path}'")
            subprocess.run(["open", html_file_path], check=True)
