import copy
import inspect
import json
import logging
import pathlib
import pickle
import shutil
import tempfile
import traceback
import warnings
from abc import abstractmethod
from typing import Any, Callable, NamedTuple, Optional, Union

import numpy as np
import pandas as pd

import mlflow
from mlflow import MlflowClient, MlflowException
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.entities.metric import Metric
from mlflow.metrics.base import MetricValue
from mlflow.models.evaluation.artifacts import (
    CsvEvaluationArtifact,
    ImageEvaluationArtifact,
    JsonEvaluationArtifact,
    NumpyEvaluationArtifact,
    _infer_artifact_type_and_ext,
)
from mlflow.models.evaluation.base import EvaluationMetric, EvaluationResult, ModelEvaluator
from mlflow.models.evaluation.utils.metric import MetricDefinition
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.pyfunc import _ServedPyFuncModel
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.time import get_current_time_millis

_logger = logging.getLogger(__name__)

_EVAL_TABLE_FILE_NAME = "eval_results_table.json"
_TOKEN_COUNT_METRIC_NAME = "token_count"
_LATENCY_METRIC_NAME = "latency"


def _extract_raw_model(model):
    if not getattr(model, "metadata", None):
        return None, None

    model_loader_module = model.metadata.flavors["python_function"]["loader_module"]
    # If we load a model with mlflow.pyfunc.load_model, the model will be wrapped
    # with a pyfunc wrapper. We need to extract the raw model so that shap
    # explainer uses the raw model instead of the wrapper and skips data schema validation.
    if model_loader_module in ["mlflow.sklearn", "mlflow.xgboost"] and not isinstance(
        model, _ServedPyFuncModel
    ):
        if hasattr(model._model_impl, "get_raw_model"):
            return model_loader_module, model._model_impl.get_raw_model()
        return model_loader_module, model._model_impl
    else:
        return model_loader_module, None


def _extract_predict_fn(model: Any) -> Optional[Callable]:
    """
    Extracts the predict function from the given model or raw_model.

    Precedence order:
    1. If raw_model is specified, its predict function is used.
    2. If model is specified, its predict function is used.
    3. If none of the above, predict function is None.

    Args:
        model: A model object that has a predict method.
        raw_model: A raw model object that has a predict method.

    Returns: The predict function.
    """
    _, raw_model = _extract_raw_model(model)
    predict_fn = None

    if raw_model is not None:
        predict_fn = raw_model.predict
        try:
            from mlflow.xgboost import _wrapped_xgboost_model_predict_fn

            # Because shap evaluation will pass evaluation data in ndarray format
            # (without feature names), if set validate_features=True it will raise error.
            predict_fn = _wrapped_xgboost_model_predict_fn(raw_model, validate_features=False)
        except ImportError:
            pass

    elif model is not None:
        predict_fn = model.predict

    return predict_fn


def _get_dataframe_with_renamed_columns(x, new_column_names):
    """
    Downstream inference functions may expect a pd.DataFrame to be created from x. However,
    if x is already a pd.DataFrame, and new_column_names != x.columns, we cannot simply call
    pd.DataFrame(x, columns=new_column_names) because the resulting pd.DataFrame will contain
    NaNs for every column in new_column_names that does not exist in x.columns. This function
    instead creates a new pd.DataFrame object from x, and then explicitly renames the columns
    to avoid NaNs.

    Args:
        x: A data object, such as a Pandas DataFrame, numPy array, or list
        new_column_names: Column names for the output Pandas DataFrame

    Returns:
        A pd.DataFrame with x as data, with columns new_column_names
    """
    df = pd.DataFrame(x)
    return df.rename(columns=dict(zip(df.columns, new_column_names)))


def _get_aggregate_metrics_values(metrics):
    return {name: MetricValue(aggregate_results={name: value}) for name, value in metrics.items()}


_matplotlib_config = {
    "figure.dpi": 175,
    "figure.figsize": [6.0, 4.0],
    "figure.autolayout": True,
    "font.size": 8,
}


class _CustomArtifact(NamedTuple):
    """
    A namedtuple representing a custom artifact function and its properties.

    function : the custom artifact function
    name : the name of the custom artifact function
    index : the index of the function in the ``custom_artifacts`` argument of mlflow.evaluate
    artifacts_dir : the path to a temporary directory to store produced artifacts of the function
    """

    function: Callable
    name: str
    index: int
    artifacts_dir: str


def _is_valid_artifacts(artifacts):
    return isinstance(artifacts, dict) and all(isinstance(k, str) for k in artifacts.keys())


def _evaluate_custom_artifacts(custom_artifact_tuple, eval_df, builtin_metrics):
    """
    This function calls the `custom_artifact` function and performs validations on the returned
    result to ensure that they are in the expected format. It will raise a MlflowException if
    the result is not in the expected format.

    Args:
        custom_artifact_tuple: Containing a user provided function and its index in the
            ``custom_artifacts`` parameter of ``mlflow.evaluate``
        eval_df: A Pandas dataframe object containing a prediction and a target column.
        builtin_metrics: A dictionary of metrics produced by the default evaluator.

    Returns:
        A dictionary of artifacts.
    """
    exception_header = (
        f"Custom artifact function '{custom_artifact_tuple.name}' "
        " at index {custom_artifact_tuple.index}"
        " in the `custom_artifacts` parameter"
    )
    artifacts = custom_artifact_tuple.function(
        eval_df, builtin_metrics, custom_artifact_tuple.artifacts_dir
    )

    if artifacts is None:
        _logger.warning(f"{exception_header} returned None.")
        return

    if not _is_valid_artifacts(artifacts):
        _logger.warning(
            f"{exception_header} did not return artifacts as a dictionary of string artifact "
            "names with their corresponding objects."
        )
        return

    return artifacts


# TODO: Move this to the /evaluators directory
class BuiltInEvaluator(ModelEvaluator):
    """
    The base class for all evaluators that are built-in to MLflow.

    Each evaluator is responsible for implementing the `_evaluate()` method, which is called by
    the `evaluate()` method of this base class. This class contains many helper methods used
    across built-in evaluators, such as logging metrics, artifacts, and ordering metrics.
    """

    def __init__(self):
        self.client = MlflowClient()

    @abstractmethod
    def _evaluate(
        self,
        model: Optional["mlflow.pyfunc.PyFuncModel"],
        extra_metrics: list[EvaluationMetric],
        custom_artifacts=None,
        **kwargs,
    ) -> Optional[EvaluationResult]:
        """Implement the evaluation logic for each evaluator."""

    def log_metrics(self):
        """
        Helper method to log metrics into specified run.
        """
        self._add_prefix_to_metrics()

        timestamp = get_current_time_millis()
        self.client.log_batch(
            self.run_id,
            metrics=[
                Metric(
                    key=key,
                    value=value,
                    timestamp=timestamp,
                    step=0,
                )
                for key, value in self.aggregate_metrics.items()
            ],
        )

    def _log_image_artifact(
        self,
        do_plot,
        artifact_name,
    ):
        from matplotlib import pyplot

        prefix = self.evaluator_config.get("metric_prefix", "")
        artifact_file_name = f"{prefix}{artifact_name}.png"
        artifact_file_local_path = self.temp_dir.path(artifact_file_name)

        try:
            pyplot.clf()
            do_plot()
            pyplot.savefig(artifact_file_local_path, bbox_inches="tight")
        except Exception as e:
            _logger.warning(f"Failed to log image artifact {artifact_name!r}: {e!r}")
        else:
            mlflow.log_artifact(artifact_file_local_path)
            artifact = ImageEvaluationArtifact(uri=mlflow.get_artifact_uri(artifact_file_name))
            artifact._load(artifact_file_local_path)
            self.artifacts[artifact_name] = artifact
        finally:
            pyplot.close(pyplot.gcf())

    def _evaluate_sklearn_model_score_if_scorable(self, model, y_true, sample_weights):
        model_loader_module, raw_model = _extract_raw_model(model)
        if model_loader_module == "mlflow.sklearn" and raw_model is not None:
            try:
                score = raw_model.score(
                    self.X.copy_to_avoid_mutation(), y_true, sample_weight=sample_weights
                )
                self.metrics_values.update(_get_aggregate_metrics_values({"score": score}))
            except Exception as e:
                _logger.warning(
                    f"Computing sklearn model score failed: {e!r}. Set logging level to "
                    "DEBUG to see the full traceback."
                )
                _logger.debug("", exc_info=True)

    def _log_custom_metric_artifact(self, artifact_name, raw_artifact, custom_metric_tuple):
        """
        This function logs and returns a custom metric artifact. Two cases:
            - The provided artifact is a path to a file, the function will make a copy of it with
              a formatted name in a temporary directory and call mlflow.log_artifact.
            - Otherwise: will attempt to save the artifact to an temporary path with an inferred
              type. Then call mlflow.log_artifact.

        Args:
            artifact_name: the name of the artifact
            raw_artifact: the object representing the artifact
            custom_metric_tuple: an instance of the _CustomMetric namedtuple

        Returns:
            EvaluationArtifact
        """

        exception_and_warning_header = (
            f"Custom artifact function '{custom_metric_tuple.name}' at index "
            f"{custom_metric_tuple.index} in the `custom_artifacts` parameter"
        )

        inferred_from_path, inferred_type, inferred_ext = _infer_artifact_type_and_ext(
            artifact_name, raw_artifact, custom_metric_tuple
        )
        artifact_file_local_path = self.temp_dir.path(artifact_name + inferred_ext)

        if pathlib.Path(artifact_file_local_path).exists():
            raise MlflowException(
                f"{exception_and_warning_header} produced an artifact '{artifact_name}' that "
                "cannot be logged because there already exists an artifact with the same name."
            )

        # ParquetEvaluationArtifact isn't explicitly stated here because such artifacts can only
        # be supplied through file. Which is handled by the first if clause. This is because
        # DataFrame objects default to be stored as CsvEvaluationArtifact.
        if inferred_from_path:
            shutil.copy2(raw_artifact, artifact_file_local_path)
        elif inferred_type is JsonEvaluationArtifact:
            with open(artifact_file_local_path, "w") as f:
                if isinstance(raw_artifact, str):
                    f.write(raw_artifact)
                else:
                    json.dump(raw_artifact, f, cls=NumpyEncoder)
        elif inferred_type is CsvEvaluationArtifact:
            raw_artifact.to_csv(artifact_file_local_path, index=False)
        elif inferred_type is NumpyEvaluationArtifact:
            np.save(artifact_file_local_path, raw_artifact, allow_pickle=False)
        elif inferred_type is ImageEvaluationArtifact:
            raw_artifact.savefig(artifact_file_local_path)
        else:
            # storing as pickle
            try:
                with open(artifact_file_local_path, "wb") as f:
                    pickle.dump(raw_artifact, f)
                _logger.warning(
                    f"{exception_and_warning_header} produced an artifact '{artifact_name}'"
                    f" with type '{type(raw_artifact)}' that is logged as a pickle artifact."
                )
            except pickle.PickleError:
                raise MlflowException(
                    f"{exception_and_warning_header} produced an unsupported artifact "
                    f"'{artifact_name}' with type '{type(raw_artifact)}' that cannot be pickled. "
                    "Supported object types for artifacts are:\n"
                    "- A string uri representing the file path to the artifact. MLflow"
                    "  will infer the type of the artifact based on the file extension.\n"
                    "- A string representation of a JSON object. This will be saved as a "
                    ".json artifact.\n"
                    "- Pandas DataFrame. This will be saved as a .csv artifact."
                    "- Numpy array. This will be saved as a .npy artifact."
                    "- Matplotlib Figure. This will be saved as an .png image artifact."
                    "- Other objects will be attempted to be pickled with default protocol."
                )

        mlflow.log_artifact(artifact_file_local_path)
        artifact = inferred_type(uri=mlflow.get_artifact_uri(artifact_name + inferred_ext))
        artifact._load(artifact_file_local_path)
        return artifact

    def _get_column_in_metrics_values(self, column):
        for metric_name, metric_value in self.metrics_values.items():
            if metric_name.split("/")[0] == column:
                return metric_value

    def _get_args_for_metrics(
        self,
        metric: MetricDefinition,
        eval_df: pd.DataFrame,
        input_df: pd.DataFrame,
        other_output_df: Optional[pd.DataFrame],
    ) -> tuple[bool, list[Union[str, pd.DataFrame]]]:
        """
        Given a metric_tuple, read the signature of the metric function and get the appropriate
        arguments from the input/output columns, other calculated metrics, and evaluator_config.

        Args:
            metric: The metric definition containing a user provided function and its index
                in the ``extra_metrics`` parameter of ``mlflow.evaluate``.
            eval_df: The evaluation dataframe containing the prediction and target columns.
            input_df: The input dataframe containing the features used to make predictions.
            other_output_df: A dataframe containing all model output columns but the predictions.

        Returns:
            tuple: A tuple of (bool, list) where the bool indicates if the given metric can
            be calculated with the given eval_df, metrics, and input_df.
                - If the user is missing "targets" or "predictions" parameters when needed, or we
                cannot find a column or metric for a parameter to the metric, return
                    (False, list of missing parameters)
                - If all arguments to the metric function were found, return
                    (True, list of arguments).
        """
        # deepcopying eval_df and builtin_metrics for each custom metric function call,
        # in case the user modifies them inside their function(s).
        eval_df_copy = eval_df.copy()
        parameters = inspect.signature(metric.function).parameters
        eval_fn_args = []
        params_not_found = []
        if len(parameters) == 2:
            param_0_name, param_1_name = parameters.keys()

        # eval_fn has parameters (eval_df, builtin_metrics) for backwards compatibility
        if len(parameters) == 2 and param_0_name != "predictions" and param_1_name != "targets":
            eval_fn_args.append(eval_df_copy)
            self._update_aggregate_metrics()
            eval_fn_args.append(copy.deepcopy(self.aggregate_metrics))
        # eval_fn can have parameters like (predictions, targets, metrics, random_col)
        else:
            for param_name, param in parameters.items():
                column = self.col_mapping.get(param_name, param_name)

                if (
                    column == "predictions"
                    or column == self.predictions
                    or column == self.dataset.predictions_name
                ):
                    eval_fn_args.append(eval_df_copy["prediction"])
                elif column == "targets" or column == self.dataset.targets_name:
                    if "target" in eval_df_copy:
                        eval_fn_args.append(eval_df_copy["target"])
                    else:
                        if param.default == inspect.Parameter.empty:
                            params_not_found.append(param_name)
                        else:
                            eval_fn_args.append(param.default)
                elif column == "metrics":
                    eval_fn_args.append(copy.deepcopy(self.metrics_values))
                else:
                    # case when column passed in col_mapping contains the entire column
                    if not isinstance(column, str):
                        eval_fn_args.append(column)

                    # case column in col_mapping is string and the column value
                    # is part of the input_df
                    elif column in input_df.columns:
                        eval_fn_args.append(input_df[column])

                    # case column in col_mapping is string and the column value
                    # is part of the output_df(other than predictions)
                    elif other_output_df is not None and column in other_output_df.columns:
                        self.other_output_columns_for_eval.add(column)
                        eval_fn_args.append(other_output_df[column])

                    # case where the param is defined as part of the evaluator_config
                    elif column in self.evaluator_config:
                        eval_fn_args.append(self.evaluator_config.get(column))

                    # case where this is the name of another metric
                    elif metric_value := self._get_column_in_metrics_values(column):
                        eval_fn_args.append(metric_value)

                    # in the case that:
                    # the metric has not been calculated yet, but is scheduled to be calculated
                    # "before" this metric in self.ordered_metrics, we append None to indicate
                    # that there is not an error in the dependencies
                    elif column in [metric_tuple.name for metric_tuple in self.ordered_metrics]:
                        eval_fn_args.append(None)

                    elif param.default == inspect.Parameter.empty:
                        params_not_found.append(param_name)
                    else:
                        eval_fn_args.append(param.default)

        if len(params_not_found) > 0:
            return False, params_not_found
        return True, eval_fn_args

    def evaluate_and_log_custom_artifacts(
        self,
        custom_artifacts: list[_CustomArtifact],
        prediction: pd.Series,
        target: Optional[np.array] = None,
    ):
        """Evaluate custom artifacts provided by users."""
        if not custom_artifacts:
            return

        eval_df = self._get_eval_df(prediction, target)
        for index, custom_artifact in enumerate(custom_artifacts):
            with tempfile.TemporaryDirectory() as artifacts_dir:
                # deepcopying eval_df and builtin_metrics for each custom artifact function call,
                # in case the user modifies them inside their function(s).
                custom_artifact_tuple = _CustomArtifact(
                    function=custom_artifact,
                    index=index,
                    name=getattr(custom_artifact, "__name__", repr(custom_artifact)),
                    artifacts_dir=artifacts_dir,
                )
                artifact_results = _evaluate_custom_artifacts(
                    custom_artifact_tuple,
                    eval_df.copy(),
                    copy.deepcopy(self.metrics_values),
                )
                if artifact_results:
                    for artifact_name, raw_artifact in artifact_results.items():
                        self.artifacts[artifact_name] = self._log_custom_metric_artifact(
                            artifact_name,
                            raw_artifact,
                            custom_artifact_tuple,
                        )

    def _get_error_message_missing_columns(self, metric_name, param_names):
        error_message_parts = [f"Metric '{metric_name}' requires the following:"]

        special_params = ["targets", "predictions"]
        for param in special_params:
            if param in param_names:
                error_message_parts.append(f"  - the '{param}' parameter needs to be specified")

        remaining_params = [param for param in param_names if param not in special_params]

        if remaining_params:
            error_message_parts.append(
                f"  - missing columns {remaining_params} need to be defined or mapped"
            )

        return "\n".join(error_message_parts)

    def _construct_error_message_for_malformed_metrics(
        self, malformed_results, input_columns, output_columns
    ):
        error_messages = [
            self._get_error_message_missing_columns(metric_name, param_names)
            for metric_name, param_names in malformed_results
        ]
        joined_error_message = "\n".join(error_messages)

        full_message = f"""Error: Metric calculation failed for the following metrics:
        {joined_error_message}

        Below are the existing column names for the input/output data:
        Input Columns: {input_columns}
        Output Columns: {output_columns}

        To resolve this issue, you may need to:
         - specify any required parameters
         - if you are missing columns, check that there are no circular dependencies among your
         metrics, and you may want to map them to an existing column using the following
         configuration:
        evaluator_config={{'col_mapping': {{<missing column name>: <existing column name>}}}}"""

        return "\n".join(l.lstrip() for l in full_message.splitlines())

    def _raise_exception_for_malformed_metrics(self, malformed_results, eval_df, other_output_df):
        output_columns = [] if other_output_df is None else list(other_output_df.columns)
        if self.predictions:
            output_columns.append(self.predictions)
        elif self.dataset.predictions_name:
            output_columns.append(self.dataset.predictions_name)
        else:
            output_columns.append("predictions")

        input_columns = list(self.X.copy_to_avoid_mutation().columns)
        if "target" in eval_df:
            if self.dataset.targets_name:
                input_columns.append(self.dataset.targets_name)
            else:
                input_columns.append("targets")

        error_message = self._construct_error_message_for_malformed_metrics(
            malformed_results, input_columns, output_columns
        )

        raise MlflowException(error_message, error_code=INVALID_PARAMETER_VALUE)

    def _get_eval_df(self, prediction: pd.Series, target: Optional[np.array] = None):
        """
        Create a DataFrame with "prediction" and "target" columns.

        This is a standard format that can be passed to the metric functions.
        """
        eval_df = pd.DataFrame({"prediction": copy.deepcopy(prediction)})
        if target is not None:
            eval_df["target"] = target
        return eval_df

    def _order_metrics(
        self,
        metrics: list[EvaluationMetric],
        eval_df: pd.DataFrame,
        other_output_df: Optional[pd.DataFrame],
    ):
        """
        Order the list metrics so they can be computed in sequence.

        Some metrics might use the results of other metrics to compute their own results. This
        function iteratively resolve this dependency, by checking if each metric can be computed
        with the current available columns and metrics values.
        """
        remaining_metrics = metrics
        input_df = self.X.copy_to_avoid_mutation()

        while len(remaining_metrics) > 0:
            pending_metrics = []
            failed_results = []
            did_append_metric = False
            for metric_tuple in remaining_metrics:
                can_calculate, eval_fn_args = self._get_args_for_metrics(
                    metric_tuple, eval_df, input_df, other_output_df
                )
                if can_calculate:
                    self.ordered_metrics.append(metric_tuple)
                    did_append_metric = True
                else:  # cannot calculate the metric yet
                    pending_metrics.append(metric_tuple)
                    failed_results.append((metric_tuple.name, eval_fn_args))

            # cant calculate any more metrics
            if not did_append_metric:
                self._raise_exception_for_malformed_metrics(
                    failed_results, eval_df, other_output_df
                )

            remaining_metrics = pending_metrics

        return self.ordered_metrics

    def _test_first_row(
        self,
        metrics: list[MetricDefinition],
        eval_df: pd.DataFrame,
        other_output_df: Optional[pd.DataFrame],
    ):
        # test calculations on first row of eval_df
        _logger.info("Testing metrics on first row...")
        exceptions = []
        first_row_df = eval_df.iloc[[0]]
        first_row_input_df = self.X.copy_to_avoid_mutation().iloc[[0]]
        for metric in metrics:
            try:
                _, eval_fn_args = self._get_args_for_metrics(
                    metric, first_row_df, first_row_input_df, other_output_df
                )
                metric_value = metric.evaluate(eval_fn_args)
                if metric_value:
                    name = f"{metric.name}/{metric.version}" if metric.version else metric.name
                    self.metrics_values.update({name: metric_value})
            except Exception as e:
                stacktrace_str = traceback.format_exc()
                if isinstance(e, MlflowException):
                    exceptions.append(
                        f"Metric '{metric.name}': Error:\n{e.message}\n{stacktrace_str}"
                    )
                else:
                    exceptions.append(f"Metric '{metric.name}': Error:\n{e!r}\n{stacktrace_str}")

        if len(exceptions) > 0:
            raise MlflowException("\n".join(exceptions))

    def evaluate_metrics(
        self,
        metrics: list[EvaluationMetric],
        prediction: pd.Series,
        target: Optional[np.array] = None,
        other_output_df: Optional[pd.DataFrame] = None,
    ):
        """
        Evaluate the metrics on the given prediction and target data.

        Args:
            metrics: A list of metrics to evaluate.
            prediction: A Pandas Series containing the predictions.
            target: A numpy array containing the target values.
            other_output_df: A Pandas DataFrame containing other output columns from the model.

        Returns:
            None, the metrics values are recorded in the self.metrics_values dictionary.
        """

        eval_df = self._get_eval_df(prediction, target)
        metrics = [
            MetricDefinition.from_index_and_metric(i, metric) for i, metric in enumerate(metrics)
        ]
        metrics = self._order_metrics(metrics, eval_df, other_output_df)

        self._test_first_row(metrics, eval_df, other_output_df)

        # calculate metrics for the full eval_df
        input_df = self.X.copy_to_avoid_mutation()
        for metric in metrics:
            _, eval_fn_args = self._get_args_for_metrics(metric, eval_df, input_df, other_output_df)
            metric_value = metric.evaluate(eval_fn_args)

            if metric_value:
                name = f"{metric.name}/{metric.version}" if metric.version else metric.name
                self.metrics_values.update({name: metric_value})

    def log_eval_table(self, y_pred, other_output_columns=None):
        # only log eval table if there are per row metrics recorded
        if not any(
            metric_value.scores is not None or metric_value.justifications is not None
            for _, metric_value in self.metrics_values.items()
        ):
            return

        metric_prefix = self.evaluator_config.get("metric_prefix", "")
        if not isinstance(metric_prefix, str):
            metric_prefix = ""
        if isinstance(self.dataset.features_data, pd.DataFrame):
            # Handle DataFrame case
            if self.dataset.has_targets:
                data = self.dataset.features_data.assign(
                    **{
                        self.dataset.targets_name or "target": self.dataset.labels_data,
                        self.dataset.predictions_name or self.predictions or "outputs": y_pred,
                    }
                )
            else:
                data = self.dataset.features_data.assign(outputs=y_pred)
        else:
            # Handle NumPy array case, converting it to a DataFrame
            data = pd.DataFrame(self.dataset.features_data, columns=self.dataset.feature_names)
            if self.dataset.has_targets:
                data = data.assign(
                    **{
                        self.dataset.targets_name or "target": self.dataset.labels_data,
                        self.dataset.predictions_name or self.predictions or "outputs": y_pred,
                    }
                )
            else:
                data = data.assign(outputs=y_pred)

        # Include other_output_columns used in evaluation to the eval table
        if other_output_columns is not None and len(self.other_output_columns_for_eval) > 0:
            for column in self.other_output_columns_for_eval:
                data[column] = other_output_columns[column]

        columns = {}
        for metric_name, metric_value in self.metrics_values.items():
            scores = metric_value.scores
            justifications = metric_value.justifications

            if scores:
                if metric_name.startswith(metric_prefix) and metric_name[len(metric_prefix) :] in [
                    _TOKEN_COUNT_METRIC_NAME,
                    _LATENCY_METRIC_NAME,
                ]:
                    columns[metric_name] = scores
                else:
                    columns[f"{metric_name}/score"] = scores
            if justifications:
                columns[f"{metric_name}/justification"] = justifications
        data = data.assign(**columns)
        artifact_file_name = f"{metric_prefix}{_EVAL_TABLE_FILE_NAME}"
        mlflow.log_table(data, artifact_file=artifact_file_name)
        if self.eval_results_path:
            eval_table_spark = self.spark_session.createDataFrame(data)
            try:
                eval_table_spark.write.mode(self.eval_results_mode).option(
                    "mergeSchema", "true"
                ).format("delta").saveAsTable(self.eval_results_path)
            except Exception as e:
                _logger.info(f"Saving eval table to delta table failed. Reason: {e}")

        name = _EVAL_TABLE_FILE_NAME.split(".", 1)[0]
        self.artifacts[name] = JsonEvaluationArtifact(
            uri=mlflow.get_artifact_uri(artifact_file_name)
        )

    def _update_aggregate_metrics(self):
        self.aggregate_metrics = {}
        for metric_name, metric_value in self.metrics_values.items():
            if metric_value.aggregate_results:
                for agg_name, agg_value in metric_value.aggregate_results.items():
                    if agg_value is not None:
                        if agg_name == metric_name.split("/")[0]:
                            self.aggregate_metrics[metric_name] = agg_value
                        else:
                            self.aggregate_metrics[f"{metric_name}/{agg_name}"] = agg_value

    def _add_prefix_to_metrics(self):
        def _prefix_value(value):
            aggregate = (
                {f"{prefix}{k}": v for k, v in value.aggregate_results.items()}
                if value.aggregate_results
                else None
            )
            return MetricValue(value.scores, value.justifications, aggregate)

        if prefix := self.evaluator_config.get("metric_prefix"):
            self.metrics_values = {
                f"{prefix}{k}": _prefix_value(v) for k, v in self.metrics_values.items()
            }

        self._update_aggregate_metrics()

    def evaluate(
        self,
        *,
        model_type,
        dataset,
        run_id,
        evaluator_config,
        model: "mlflow.pyfunc.PyFuncModel" = None,
        custom_metrics=None,
        extra_metrics=None,
        custom_artifacts=None,
        predictions=None,
        **kwargs,
    ) -> EvaluationResult:
        if model is None and predictions is None and dataset.predictions_data is None:
            raise MlflowException(
                message=(
                    "Either a model or set of predictions must be specified in order to use the"
                    " default evaluator. Either specify the `model` parameter, the `predictions`"
                    " parameter, an MLflow dataset containing the `predictions` column name"
                    " (via the `data` parameter), or a different evaluator (via the `evaluators`"
                    " parameter)."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

        self.artifacts = {}
        self.aggregate_metrics = {}
        self.metrics_values = {}
        self.ordered_metrics = []
        self.other_output_columns_for_eval = set()

        self.dataset: EvaluationDataset = dataset
        self.run_id = run_id
        self.model_type = model_type
        self.evaluator_config = evaluator_config

        self.predictions = predictions
        self.col_mapping = self.evaluator_config.get("col_mapping", {})
        self.eval_results_path = self.evaluator_config.get("eval_results_path")
        self.eval_results_mode = self.evaluator_config.get("eval_results_mode", "overwrite")

        if self.eval_results_path:
            from mlflow.utils._spark_utils import _get_active_spark_session

            self.spark_session = _get_active_spark_session()
            if not self.spark_session:
                raise MlflowException(
                    message="eval_results_path is only supported in Spark environment. ",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            if self.eval_results_mode not in ["overwrite", "append"]:
                raise MlflowException(
                    message="eval_results_mode can only be 'overwrite' or 'append'. ",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        if extra_metrics and custom_metrics:
            raise MlflowException(
                "The 'custom_metrics' parameter in mlflow.evaluate is deprecated. Please update "
                "your code to only use the 'extra_metrics' parameter instead."
            )
        if custom_metrics:
            warnings.warn(
                "The 'custom_metrics' parameter in mlflow.evaluate is deprecated. "
                "Please update your code to use the 'extra_metrics' parameter instead.",
                FutureWarning,
                stacklevel=2,
            )
            extra_metrics = custom_metrics

        if extra_metrics is None:
            extra_metrics = []

        bad_metrics = []
        for metric in extra_metrics:
            if not isinstance(metric, EvaluationMetric):
                bad_metrics.append(metric)
        if len(bad_metrics) > 0:
            message = "\n".join(
                [f"- Metric '{m}' has type '{type(m).__name__}'" for m in bad_metrics]
            )
            raise MlflowException(
                f"In the 'extra_metrics' parameter, the following metrics have the wrong type:\n"
                f"{message}\n"
                f"Please ensure that all extra metrics are instances of "
                f"mlflow.metrics.EvaluationMetric."
            )

        import matplotlib

        with TempDir() as temp_dir, matplotlib.rc_context(_matplotlib_config):
            self.temp_dir = temp_dir
            return self._evaluate(model, extra_metrics, custom_artifacts)

    @property
    def X(self) -> pd.DataFrame:
        """
        The features (`X`) portion of the dataset, guarded against accidental mutations.
        """
        return BuiltInEvaluator._MutationGuardedData(
            _get_dataframe_with_renamed_columns(
                self.dataset.features_data, self.dataset.feature_names
            )
        )

    class _MutationGuardedData:
        """
        Wrapper around a data object that requires explicit API calls to obtain either a copy
        of the data object, or, in cases where the caller can guaranteed that the object will not
        be mutated, the original data object.
        """

        def __init__(self, data):
            """
            Args:
                data: A data object, such as a Pandas DataFrame, numPy array, or list.
            """
            self._data = data

        def copy_to_avoid_mutation(self):
            """
            Obtain a copy of the data. This method should be called every time the data needs
            to be used in a context where it may be subsequently mutated, guarding against
            accidental reuse after mutation.

            Returns:
                A copy of the data object.
            """
            if isinstance(self._data, pd.DataFrame):
                return self._data.copy(deep=True)
            else:
                return copy.deepcopy(self._data)

        def get_original(self):
            """
            Obtain the original data object. This method should only be called if the caller
            can guarantee that it will not mutate the data during subsequent operations.

            Returns:
                The original data object.
            """
            return self._data
