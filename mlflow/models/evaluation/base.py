import json
import keyword
import logging
import operator
import os
import pathlib
import signal
import urllib
import urllib.parse
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from decimal import Decimal
from types import FunctionType
from typing import Any, Dict, Optional

import mlflow
from mlflow.data.dataset import Dataset
from mlflow.data.evaluation_dataset import EvaluationDataset, convert_data_to_mlflow_dataset
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.validation import (
    MetricThreshold,
    ModelValidationFailedException,
    _MetricValidationResult,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.annotations import developer_stable, experimental
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
from mlflow.utils.proto_json_utils import NumpyEncoder

try:
    # `pandas` is not required for `mlflow-skinny`.
    import pandas as pd
except ImportError:
    pass

_logger = logging.getLogger(__name__)


class _ModelType:
    REGRESSOR = "regressor"
    CLASSIFIER = "classifier"
    QUESTION_ANSWERING = "question-answering"
    TEXT_SUMMARIZATION = "text-summarization"
    TEXT = "text"
    RETRIEVER = "retriever"

    def __init__(self):
        raise NotImplementedError("This class is not meant to be instantiated.")

    @classmethod
    def values(cls):
        return (
            cls.REGRESSOR,
            cls.CLASSIFIER,
            cls.QUESTION_ANSWERING,
            cls.TEXT_SUMMARIZATION,
            cls.TEXT,
            cls.RETRIEVER,
        )


class EvaluationMetric:
    '''
    An evaluation metric.

    Args:
        eval_fn: A function that computes the metric with the following signature:

            .. code-block:: python

                def eval_fn(
                    predictions: pandas.Series,
                    targets: pandas.Series,
                    metrics: Dict[str, MetricValue],
                    **kwargs,
                ) -> Union[float, MetricValue]:
                    """
                    Args:
                        predictions: A pandas Series containing the predictions made by the model.
                        targets: (Optional) A pandas Series containing the corresponding labels
                            for the predictions made on that input.
                        metrics: (Optional) A dictionary containing the metrics calculated by the
                            default evaluator.  The keys are the names of the metrics and the values
                            are the metric values.  To access the MetricValue for the metrics
                            calculated by the system, make sure to specify the type hint for this
                            parameter as Dict[str, MetricValue].  Refer to the DefaultEvaluator
                            behavior section for what metrics will be returned based on the type of
                            model (i.e. classifier or regressor).
                        kwargs: Includes a list of args that are used to compute the metric. These
                            args could be information coming from input data, model outputs,
                            other metrics, or parameters specified in the `evaluator_config`
                            argument of the `mlflow.evaluate` API.

                    Returns: MetricValue with per-row scores, per-row justifications, and aggregate
                        results.
                    """
                    ...

        name: The name of the metric.
        greater_is_better: Whether a higher value of the metric is better.
        long_name: (Optional) The long name of the metric. For example,
            ``"root_mean_squared_error"`` for ``"mse"``.
        version: (Optional) The metric version. For example ``v1``.
        metric_details: (Optional) A description of the metric and how it is calculated.
        metric_metadata: (Optional) A dictionary containing metadata for the metric.
    '''

    def __init__(
        self,
        eval_fn,
        name,
        greater_is_better,
        long_name=None,
        version=None,
        metric_details=None,
        metric_metadata=None,
    ):
        self.eval_fn = eval_fn
        self.name = name
        self.greater_is_better = greater_is_better
        self.long_name = long_name or name
        self.version = version
        self.metric_details = metric_details
        self.metric_metadata = metric_metadata

    def __str__(self):
        parts = [f"name={self.name}, greater_is_better={self.greater_is_better}"]

        if self.long_name:
            parts.append(f"long_name={self.long_name}")
        if self.version:
            parts.append(f"version={self.version}")
        if self.metric_details:
            parts.append(f"metric_details={self.metric_details}")
        if self.metric_metadata:
            parts.append(f"metric_metadata={self.metric_metadata}")

        return "EvaluationMetric(" + ", ".join(parts) + ")"


def make_metric(
    *,
    eval_fn,
    greater_is_better,
    name=None,
    long_name=None,
    version=None,
    metric_details=None,
    metric_metadata=None,
):
    '''
    A factory function to create an :py:class:`EvaluationMetric` object.

    Args:
        eval_fn: A function that computes the metric with the following signature:

            .. code-block:: python

                def eval_fn(
                    predictions: pandas.Series,
                    targets: pandas.Series,
                    metrics: Dict[str, MetricValue],
                    **kwargs,
                ) -> Union[float, MetricValue]:
                    """
                    Args:
                        predictions: A pandas Series containing the predictions made by the model.
                        targets: (Optional) A pandas Series containing the corresponding labels
                            for the predictions made on that input.
                        metrics: (Optional) A dictionary containing the metrics calculated by the
                            default evaluator.  The keys are the names of the metrics and the values
                            are the metric values.  To access the MetricValue for the metrics
                            calculated by the system, make sure to specify the type hint for this
                            parameter as Dict[str, MetricValue].  Refer to the DefaultEvaluator
                            behavior section for what metrics will be returned based on the type of
                            model (i.e. classifier or regressor).  kwargs: Includes a list of args
                            that are used to compute the metric. These args could information coming
                            from input data, model outputs or parameters specified in the
                            `evaluator_config` argument of the `mlflow.evaluate` API.
                        kwargs: Includes a list of args that are used to compute the metric. These
                            args could be information coming from input data, model outputs,
                            other metrics, or parameters specified in the `evaluator_config`
                            argument of the `mlflow.evaluate` API.

                    Returns: MetricValue with per-row scores, per-row justifications, and aggregate
                        results.
                    """
                    ...

        greater_is_better: Whether a higher value of the metric is better.
        name: The name of the metric. This argument must be specified if ``eval_fn`` is a lambda
                    function or the ``eval_fn.__name__`` attribute is not available.
        long_name: (Optional) The long name of the metric. For example, ``"mean_squared_error"``
            for ``"mse"``.
        version: (Optional) The metric version. For example ``v1``.
        metric_details: (Optional) A description of the metric and how it is calculated.
        metric_metadata: (Optional) A dictionary containing metadata for the metric.

    .. seealso::

        - :py:class:`mlflow.models.EvaluationMetric`
        - :py:func:`mlflow.evaluate`
    '''
    if name is None:
        if isinstance(eval_fn, FunctionType) and eval_fn.__name__ == "<lambda>":
            raise MlflowException(
                "`name` must be specified if `eval_fn` is a lambda function.",
                INVALID_PARAMETER_VALUE,
            )
        if not hasattr(eval_fn, "__name__"):
            raise MlflowException(
                "`name` must be specified if `eval_fn` does not have a `__name__` attribute.",
                INVALID_PARAMETER_VALUE,
            )
        name = eval_fn.__name__

    if "/" in name:
        raise MlflowException(
            f"Invalid metric name '{name}'. Metric names cannot include forward slashes ('/').",
            INVALID_PARAMETER_VALUE,
        )

    if not name.isidentifier():
        _logger.warning(
            f"The metric name '{name}' provided is not a valid Python identifier, which will "
            "prevent its use as a base metric for derived metrics. Please use a valid identifier "
            "to enable creation of derived metrics that use the given metric."
        )

    if keyword.iskeyword(name):
        _logger.warning(
            f"The metric name '{name}' is a reserved Python keyword, which will "
            "prevent its use as a base metric for derived metrics. Please use a valid identifier "
            "to enable creation of derived metrics that use the given metric."
        )

    if name in ["predictions", "targets", "metrics"]:
        _logger.warning(
            f"The metric name '{name}' is used as a special parameter in MLflow metrics, which "
            "will prevent its use as a base metric for derived metrics. Please use a different "
            "name to enable creation of derived metrics that use the given metric."
        )

    return EvaluationMetric(
        eval_fn,
        name,
        greater_is_better,
        long_name,
        version,
        metric_details,
        metric_metadata,
    )


@developer_stable
class EvaluationArtifact(metaclass=ABCMeta):
    """
    A model evaluation artifact containing an artifact uri and content.
    """

    def __init__(self, uri, content=None):
        self._uri = uri
        self._content = content

    @abstractmethod
    def _load_content_from_file(self, local_artifact_path):
        """
        Abstract interface to load the content from local artifact file path,
        and return the loaded content.
        """

    def _load(self, local_artifact_path=None):
        """
        If ``local_artifact_path`` is ``None``, download artifact from the artifact uri.
        Otherwise, load artifact content from the specified path. Assign the loaded content to
        ``self._content``, and return the loaded content.
        """
        if local_artifact_path is not None:
            self._content = self._load_content_from_file(local_artifact_path)
        else:
            with TempDir() as temp_dir:
                temp_dir_path = temp_dir.path()
                _download_artifact_from_uri(self._uri, temp_dir_path)
                local_artifact_file = temp_dir.path(os.listdir(temp_dir_path)[0])
                self._content = self._load_content_from_file(local_artifact_file)
        return self._content

    @abstractmethod
    def _save(self, output_artifact_path):
        """Save artifact content into specified path."""

    @property
    def content(self):
        """
        The content of the artifact (representation varies)
        """
        if self._content is None:
            self._load()
        return self._content

    @property
    def uri(self) -> str:
        """
        The URI of the artifact
        """
        return self._uri

    def __repr__(self):
        return f"{self.__class__.__name__}(uri='{self.uri}')"


class EvaluationResult:
    """
    Represents the model evaluation outputs of a `mlflow.evaluate()` API call, containing
    both scalar metrics and output artifacts such as performance plots.
    """

    def __init__(self, metrics, artifacts, baseline_model_metrics=None, run_id=None):
        self._metrics = metrics
        self._artifacts = artifacts
        self._baseline_model_metrics = baseline_model_metrics if baseline_model_metrics else {}
        self._run_id = (
            run_id
            if run_id is not None
            else (mlflow.active_run().info.run_id if mlflow.active_run() is not None else None)
        )

    @classmethod
    def load(cls, path):
        """Load the evaluation results from the specified local filesystem path"""
        with open(os.path.join(path, "metrics.json")) as fp:
            metrics = json.load(fp)

        with open(os.path.join(path, "artifacts_metadata.json")) as fp:
            artifacts_metadata = json.load(fp)

        artifacts = {}

        artifacts_dir = os.path.join(path, "artifacts")

        for artifact_name, meta in artifacts_metadata.items():
            uri = meta["uri"]
            ArtifactCls = _get_class_from_string(meta["class_name"])
            artifact = ArtifactCls(uri=uri)
            filename = pathlib.Path(urllib.parse.urlparse(uri).path).name
            artifact._load(os.path.join(artifacts_dir, filename))
            artifacts[artifact_name] = artifact

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    def save(self, path):
        """Write the evaluation results to the specified local filesystem path"""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metrics.json"), "w") as fp:
            json.dump(self.metrics, fp, cls=NumpyEncoder)

        artifacts_metadata = {
            artifact_name: {
                "uri": artifact.uri,
                "class_name": _get_fully_qualified_class_name(artifact),
            }
            for artifact_name, artifact in self.artifacts.items()
        }
        with open(os.path.join(path, "artifacts_metadata.json"), "w") as fp:
            json.dump(artifacts_metadata, fp)

        artifacts_dir = os.path.join(path, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        for artifact in self.artifacts.values():
            filename = pathlib.Path(urllib.parse.urlparse(artifact.uri).path).name
            artifact._save(os.path.join(artifacts_dir, filename))

    @property
    def metrics(self) -> Dict[str, Any]:
        """
        A dictionary mapping scalar metric names to scalar metric values
        """
        return self._metrics

    @property
    def artifacts(self) -> Dict[str, "mlflow.models.EvaluationArtifact"]:
        """
        A dictionary mapping standardized artifact names (e.g. "roc_data") to
        artifact content and location information
        """
        return self._artifacts

    @property
    def baseline_model_metrics(self) -> Dict[str, Any]:
        """
        A dictionary mapping scalar metric names to scalar metric values for the baseline model
        """
        return self._baseline_model_metrics

    @experimental
    @property
    def tables(self) -> Dict[str, "pd.DataFrame"]:
        """
        A dictionary mapping standardized artifact names (e.g. "eval_results_table") to
        corresponding table content as pandas DataFrame.
        """
        eval_tables = {}
        if self._run_id is None:
            _logger.warning("Cannot load eval_results_table because run_id is not specified.")
            return eval_tables

        for table_name, table_path in self._artifacts.items():
            path = urllib.parse.urlparse(table_path.uri).path
            table_fileName = os.path.basename(path)
            try:
                eval_tables[table_name] = mlflow.load_table(table_fileName, run_ids=[self._run_id])
            except Exception:
                pass  # Swallow the exception since we assume its not a table.

        return eval_tables


@developer_stable
class ModelEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def can_evaluate(self, *, model_type, evaluator_config, **kwargs) -> bool:
        """
        Args:
            model_type: A string describing the model type (e.g., "regressor", "classifier", …).
            evaluator_config: A dictionary of additional configurations for
                the evaluator.
            kwargs: For forwards compatibility, a placeholder for additional arguments
                that may be added to the evaluation interface in the future.

        Returns:
            True if the evaluator can evaluate the specified model on the
            specified dataset. False otherwise.
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(
        self,
        *,
        model_type,
        dataset,
        run_id,
        evaluator_config,
        model=None,
        custom_metrics=None,
        extra_metrics=None,
        custom_artifacts=None,
        baseline_model=None,
        predictions=None,
        **kwargs,
    ):
        """
        The abstract API to log metrics and artifacts, and return evaluation results.

        Args:
            model_type: A string describing the model type
                (e.g., ``"regressor"``, ``"classifier"``, …).
            dataset: An instance of `mlflow.models.evaluation.base._EvaluationDataset`
                containing features and labels (optional) for model evaluation.
            run_id: The ID of the MLflow Run to which to log results.
            evaluator_config: A dictionary of additional configurations for
                the evaluator.
            model: A pyfunc model instance, used as the candidate_model
                to be compared with baseline_model (specified by the `baseline_model` param)
                for model validation. If None, the model output is supposed to be found in
                ``dataset.predictions_data``.
            extra_metrics: A list of :py:class:`EvaluationMetric` objects.
            custom_artifacts: A list of callable custom artifact functions.
            kwargs: For forwards compatibility, a placeholder for additional arguments that
                may be added to the evaluation interface in the future.
            baseline_model: (Optional) A string URI referring to a MLflow model with the pyfunc
                flavor as a baseline model to be compared with the
                candidate model (specified by the `model` param) for model
                validation. (pyfunc model instance is not allowed)
            predictions: The column name of the model output column that is used for evaluation.
                This is only used when a model returns a pandas dataframe that contains
                multiple columns.

        Returns:
            A :py:class:`mlflow.models.EvaluationResult` instance containing
            evaluation metrics for candidate model and baseline model and
            artifacts for candidate model.
        """
        raise NotImplementedError()


def list_evaluators():
    """
    Return a name list for all available Evaluators.
    """
    # import _model_evaluation_registry inside function to avoid circuit importing
    from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry

    return list(_model_evaluation_registry._registry.keys())


@contextmanager
def _start_run_or_reuse_active_run():
    """
    A manager context return:
     - If there's an active run, return the active run id.
     - otherwise start a mflow run with the specified run_id,
       if specified run_id is None, start a new run.
    """
    active_run = mlflow.active_run()
    if not active_run:
        # Note `mlflow.start_run` throws if `run_id` is not found.
        with mlflow.start_run() as run:
            yield run.info.run_id
    else:
        yield active_run.info.run_id


def _normalize_evaluators_and_evaluator_config_args(
    evaluators,
    evaluator_config,
):
    from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry

    def check_nesting_config_dict(_evaluator_name_list, _evaluator_name_to_conf_map):
        return isinstance(_evaluator_name_to_conf_map, dict) and all(
            k in _evaluator_name_list and isinstance(v, dict)
            for k, v in _evaluator_name_to_conf_map.items()
        )

    if evaluators is None:
        evaluator_name_list = list(_model_evaluation_registry._registry.keys())
        if len(evaluator_name_list) > 1:
            _logger.debug(
                f"Multiple registered evaluators have been configured: {evaluator_name_list}. "
                "Each evaluator will be used for evaluation if the specified model type is "
                "compatible with the evaluator definition. If you are intending to override "
                "the default evaluator, define your custom evaluator by declaring it via the "
                "`evaluator` argument. If your evaluator requires additional configuration, "
                "ensure that it is provided by specifying the `evaluator_config` argument."
            )
        if evaluator_config is not None:
            conf_dict_value_error = MlflowException(
                message="If `evaluators` argument is None, all available evaluators will be used. "
                "If only the default evaluator is available, the `evaluator_config` argument is "
                "interpreted as the config dictionary for the default evaluator. Otherwise, the "
                "`evaluator_config` argument must be a dictionary mapping each evaluator's name "
                "to its own evaluator config dictionary.",
                error_code=INVALID_PARAMETER_VALUE,
            )
            if evaluator_name_list == ["default"]:
                if not isinstance(evaluator_config, dict):
                    raise conf_dict_value_error
                elif "default" not in evaluator_config:
                    evaluator_name_to_conf_map = {"default": evaluator_config}
                else:
                    evaluator_name_to_conf_map = evaluator_config
            else:
                if not check_nesting_config_dict(evaluator_name_list, evaluator_config):
                    raise conf_dict_value_error
                evaluator_name_to_conf_map = evaluator_config
        else:
            evaluator_name_to_conf_map = {}
    elif isinstance(evaluators, str):
        if not (evaluator_config is None or isinstance(evaluator_config, dict)):
            raise MlflowException(
                message="If `evaluators` argument is the name of an evaluator, evaluator_config"
                " must be None or a dict containing config items for the evaluator.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        evaluator_name_list = [evaluators]
        evaluator_name_to_conf_map = {evaluators: evaluator_config}
    elif isinstance(evaluators, list):
        if evaluator_config is not None:
            if not check_nesting_config_dict(evaluators, evaluator_config):
                raise MlflowException(
                    message="If `evaluators` argument is an evaluator name list, evaluator_config "
                    "must be a dict contains mapping from evaluator name to individual "
                    "evaluator config dict.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        # Use `OrderedDict.fromkeys` to deduplicate elements but keep elements order.
        evaluator_name_list = list(OrderedDict.fromkeys(evaluators))
        evaluator_name_to_conf_map = evaluator_config or {}
    else:
        raise MlflowException(
            message="`evaluators` argument must be None, an evaluator name string, or a list of "
            "evaluator names.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    return evaluator_name_list, evaluator_name_to_conf_map


def _model_validation_contains_model_comparison(validation_thresholds):
    """
    Helper function for determining if validation_thresholds contains
    thresholds for model comparsion: either min_relative_change or min_absolute_change
    """
    if not validation_thresholds:
        return False
    thresholds = validation_thresholds.values()
    return any(
        threshold.min_relative_change or threshold.min_absolute_change for threshold in thresholds
    )


_last_failed_evaluator = None


def _get_last_failed_evaluator():
    """
    Return the evaluator name of the last failed evaluator when calling `evaluate`.
    This can be used to check which evaluator fail when `evaluate` API fail.
    """
    return _last_failed_evaluator


def _validate(validation_thresholds, candidate_metrics, baseline_metrics=None):
    """
    Validate the model based on validation_thresholds by metrics value and
    metrics comparison between candidate model's metrics (candidate_metrics) and
    baseline model's metrics (baseline_metrics).

    Args:
        validation_thresholds: A dictionary from metric_name to MetricThreshold.
        candidate_metrics: The metric evaluation result of the candidate model.
        baseline_metrics: The metric evaluation result of the baseline model.
            If the validation does not pass, raise an MlflowException with detail failure message.
    """
    if not baseline_metrics:
        baseline_metrics = {}

    validation_results = {
        metric_name: _MetricValidationResult(
            metric_name,
            candidate_metrics.get(metric_name, None),
            threshold,
            baseline_metrics.get(metric_name, None),
        )
        for (metric_name, threshold) in validation_thresholds.items()
    }

    for metric_name in validation_thresholds.keys():
        metric_threshold, validation_result = (
            validation_thresholds[metric_name],
            validation_results[metric_name],
        )

        if metric_name not in candidate_metrics:
            validation_result.missing_candidate = True
            continue

        candidate_metric_value, baseline_metric_value = (
            candidate_metrics[metric_name],
            baseline_metrics[metric_name] if baseline_metrics else None,
        )

        # If metric is higher is better, >= is used, otherwise <= is used
        # for thresholding metric value and model comparsion
        comparator_fn = operator.__ge__ if metric_threshold.greater_is_better else operator.__le__
        operator_fn = operator.add if metric_threshold.greater_is_better else operator.sub

        if metric_threshold.threshold is not None:
            # metric threshold fails
            # - if not (metric_value >= threshold) for higher is better
            # - if not (metric_value <= threshold) for lower is better
            validation_result.threshold_failed = not comparator_fn(
                candidate_metric_value, metric_threshold.threshold
            )

        if (
            metric_threshold.min_relative_change or metric_threshold.min_absolute_change
        ) and metric_name not in baseline_metrics:
            validation_result.missing_baseline = True
            continue

        if metric_threshold.min_absolute_change is not None:
            # metric comparsion aboslute change fails
            # - if not (metric_value >= baseline + min_absolute_change) for higher is better
            # - if not (metric_value <= baseline - min_absolute_change) for lower is better
            validation_result.min_absolute_change_failed = not comparator_fn(
                Decimal(candidate_metric_value),
                Decimal(operator_fn(baseline_metric_value, metric_threshold.min_absolute_change)),
            )

        if metric_threshold.min_relative_change is not None:
            # If baseline metric value equals 0, fallback to simple comparison check
            if baseline_metric_value == 0:
                _logger.warning(
                    f"Cannot perform relative model comparison for metric {metric_name} as "
                    "baseline metric value is 0. Falling back to simple comparison: verifying "
                    "that candidate metric value is better than the baseline metric value."
                )
                validation_result.min_relative_change_failed = not comparator_fn(
                    Decimal(candidate_metric_value),
                    Decimal(operator_fn(baseline_metric_value, 1e-10)),
                )
                continue
            # metric comparsion relative change fails
            # - if (metric_value - baseline) / baseline < min_relative_change for higher is better
            # - if (baseline - metric_value) / baseline < min_relative_change for lower is better
            if metric_threshold.greater_is_better:
                relative_change = (
                    candidate_metric_value - baseline_metric_value
                ) / baseline_metric_value
            else:
                relative_change = (
                    baseline_metric_value - candidate_metric_value
                ) / baseline_metric_value
            validation_result.min_relative_change_failed = (
                relative_change < metric_threshold.min_relative_change
            )

    failure_messages = []

    for metric_validation_result in validation_results.values():
        if metric_validation_result.is_success():
            continue
        failure_messages.append(str(metric_validation_result))

    if not failure_messages:
        return

    raise ModelValidationFailedException(message=os.linesep.join(failure_messages))


def _evaluate(
    *,
    model,
    model_type,
    dataset,
    run_id,
    evaluator_name_list,
    evaluator_name_to_conf_map,
    custom_metrics,
    extra_metrics,
    custom_artifacts,
    baseline_model,
    predictions,
):
    """
    The public API "evaluate" will verify argument first, and then pass normalized arguments
    to the _evaluate method.
    """
    # import _model_evaluation_registry and PyFuncModel inside function to avoid circuit importing
    from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry

    global _last_failed_evaluator
    _last_failed_evaluator = None

    client = MlflowClient()

    model_uuid = getattr(model, "metadata", None)

    if model_uuid is not None:
        model_uuid = model_uuid.model_uuid
        dataset._log_dataset_tag(client, run_id, model_uuid)

    eval_results = []
    for evaluator_name in evaluator_name_list:
        config = evaluator_name_to_conf_map.get(evaluator_name) or {}
        try:
            evaluator = _model_evaluation_registry.get_evaluator(evaluator_name)
        except MlflowException:
            _logger.warning(f"Evaluator '{evaluator_name}' is not registered.")
            continue

        _last_failed_evaluator = evaluator_name
        if evaluator.can_evaluate(model_type=model_type, evaluator_config=config):
            _logger.debug(f"Evaluating the model with the {evaluator_name} evaluator.")
            eval_result = evaluator.evaluate(
                model=model,
                model_type=model_type,
                dataset=dataset,
                run_id=run_id,
                evaluator_config=config,
                custom_metrics=custom_metrics,
                extra_metrics=extra_metrics,
                custom_artifacts=custom_artifacts,
                baseline_model=baseline_model,
                predictions=predictions,
            )
            eval_results.append(eval_result)

    _last_failed_evaluator = None

    if len(eval_results) == 0:
        raise MlflowException(
            message="The model could not be evaluated by any of the registered evaluators, please "
            "verify that the model type and other configs are set correctly.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    merged_eval_result = EvaluationResult({}, {}, {}, None)

    for eval_result in eval_results:
        if not eval_result:
            continue
        merged_eval_result.metrics.update(eval_result.metrics)
        merged_eval_result.artifacts.update(eval_result.artifacts)
        if baseline_model and eval_result.baseline_model_metrics:
            merged_eval_result.baseline_model_metrics.update(eval_result.baseline_model_metrics)

    return merged_eval_result


def _get_model_from_function(fn):
    from mlflow.pyfunc.model import _PythonModelPyfuncWrapper

    class ModelFromFunction(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input: pd.DataFrame):
            return fn(model_input)

    python_model = ModelFromFunction()
    return _PythonModelPyfuncWrapper(python_model, None, None)


def _is_model_deployment_endpoint_uri(model: Any) -> bool:
    if not isinstance(model, str):
        return False

    from mlflow.metrics.genai.model_utils import _parse_model_uri

    try:
        schema, path = _parse_model_uri(model)
        return schema == "endpoints"
    except MlflowException:
        return False


def _get_model_from_deployment_endpoint_uri(
    endpoint_uri: str, params: Optional[Dict[str, Any]] = None
):
    from mlflow.metrics.genai.model_utils import _call_deployments_api, _parse_model_uri
    from mlflow.pyfunc.model import _PythonModelPyfuncWrapper

    class ModelFromDeploymentEndpoint(mlflow.pyfunc.PythonModel):
        def __init__(self, endpoint, params):
            self.endpoint = endpoint
            self.params = params

        def predict(self, context, model_input: pd.DataFrame):
            if len(model_input.columns) != 1:
                raise MlflowException(
                    f"The number of input columns must be 1, but got {model_input.columns}. "
                    "Multi-column input is not supported for evaluating an MLflow Deployments "
                    "endpoint. Please include the input text or payload in a single column.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            input_column = model_input.columns[0]

            predictions = []
            for data in model_input[input_column]:
                if isinstance(data, str):
                    # If the input data is a string, we will construct the request payload from it.
                    prediction = _call_deployments_api(self.endpoint, data, self.params)
                elif isinstance(data, dict):
                    # If the input data is a dictionary, we will directly use it as the request
                    # payload, with adding the inference parameters if provided.
                    prediction = _call_deployments_api(
                        self.endpoint, data, self.params, wrap_payload=False
                    )
                else:
                    raise MlflowException(
                        f"Invalid input column type: {type(data)}. The input data must be either "
                        "a string or a dictionary contains the request payload for evaluating an "
                        "MLflow Deployments endpoint.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )

                predictions.append(prediction)

            return pd.Series(predictions)

    prefix, endpoint = _parse_model_uri(endpoint_uri)
    params = params or {}

    python_model = ModelFromDeploymentEndpoint(endpoint, params)
    return _PythonModelPyfuncWrapper(python_model, None, None)


def evaluate(
    model=None,
    data=None,
    *,
    model_type=None,
    targets=None,
    predictions=None,
    dataset_path=None,
    feature_names=None,
    evaluators=None,
    evaluator_config=None,
    custom_metrics=None,
    extra_metrics=None,
    custom_artifacts=None,
    validation_thresholds=None,
    baseline_model=None,
    env_manager="local",
    model_config=None,
    baseline_config=None,
    inference_params=None,
):
    '''
    Evaluate the model performance on given data and selected metrics.

    This function evaluates a PyFunc model or custom callable on the specified dataset using
    specified ``evaluators``, and logs resulting metrics & artifacts to MLflow tracking server.
    Users can also skip setting ``model`` and put the model outputs in ``data`` directly for
    evaluation. For detailed information, please read
    :ref:`the Model Evaluation documentation <model-evaluation>`.

    Default Evaluator behavior:
     - The default evaluator, which can be invoked with ``evaluators="default"`` or
       ``evaluators=None``, supports model types listed below. For each pre-defined model type, the
       default evaluator evaluates your model on a selected set of metrics and generate artifacts
       like plots. Please find more details below.

     - For both the ``"regressor"`` and ``"classifier"`` model types, the default evaluator
       generates model summary plots and feature importance plots using
       `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_.

     - For regressor models, the default evaluator additionally logs:
        - **metrics**: example_count, mean_absolute_error, mean_squared_error,
          root_mean_squared_error, sum_on_target, mean_on_target, r2_score, max_error,
          mean_absolute_percentage_error.

     - For binary classifiers, the default evaluator additionally logs:
        - **metrics**: true_negatives, false_positives, false_negatives, true_positives, recall,
          precision, f1_score, accuracy_score, example_count, log_loss, roc_auc,
          precision_recall_auc.
        - **artifacts**: lift curve plot, precision-recall plot, ROC plot.

     - For multiclass classifiers, the default evaluator additionally logs:
        - **metrics**: accuracy_score, example_count, f1_score_micro, f1_score_macro, log_loss
        - **artifacts**: A CSV file for "per_class_metrics" (per-class metrics includes
          true_negatives/false_positives/false_negatives/true_positives/recall/precision/roc_auc,
          precision_recall_auc), precision-recall merged curves plot, ROC merged curves plot.

     - For question-answering models, the default evaluator logs:
        - **metrics**: ``exact_match``, ``token_count``, `toxicity`_ (requires `evaluate`_,
          `torch`_, `flesch_kincaid_grade_level`_ (requires `textstat`_) and `ari_grade_level`_.
        - **artifacts**: A JSON file containing the inputs, outputs, targets (if the ``targets``
          argument is supplied), and per-row metrics of the model in tabular format.

        .. _toxicity:
            https://huggingface.co/spaces/evaluate-measurement/toxicity

        .. _torch:
            https://pytorch.org/get-started/locally/

        .. _transformers:
            https://huggingface.co/docs/transformers/installation

        .. _ari_grade_level:
            https://en.wikipedia.org/wiki/Automated_readability_index

        .. _flesch_kincaid_grade_level:
            https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level

        .. _evaluate:
            https://pypi.org/project/evaluate

        .. _textstat:
            https://pypi.org/project/textstat

     - For text-summarization models, the default evaluator logs:
        - **metrics**: ``token_count``, `ROUGE`_ (requires `evaluate`_, `nltk`_, and
          `rouge_score`_ to be installed), `toxicity`_ (requires `evaluate`_, `torch`_,
          `transformers`_), `ari_grade_level`_ (requires `textstat`_),
          `flesch_kincaid_grade_level`_ (requires `textstat`_).
        - **artifacts**: A JSON file containing the inputs, outputs, targets (if the ``targets``
          argument is supplied), and per-row metrics of the model in the tabular format.

        .. _ROUGE:
            https://huggingface.co/spaces/evaluate-metric/rouge

        .. _toxicity:
            https://huggingface.co/spaces/evaluate-measurement/toxicity

        .. _torch:
            https://pytorch.org/get-started/locally/

        .. _transformers:
            https://huggingface.co/docs/transformers/installation

        .. _ari_grade_level:
            https://en.wikipedia.org/wiki/Automated_readability_index

        .. _flesch_kincaid_grade_level:
            https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level

        .. _evaluate:
            https://pypi.org/project/evaluate

        .. _nltk:
            https://pypi.org/project/nltk

        .. _rouge_score:
            https://pypi.org/project/rouge-score

        .. _textstat:
            https://pypi.org/project/textstat

     - For text models, the default evaluator logs:
        - **metrics**: ``token_count``, `toxicity`_ (requires `evaluate`_, `torch`_,
          `transformers`_), `ari_grade_level`_ (requires `textstat`_),
          `flesch_kincaid_grade_level`_ (requires `textstat`_).
        - **artifacts**: A JSON file containing the inputs, outputs, targets (if the ``targets``
          argument is supplied), and per-row metrics of the model in tabular format.

        .. _evaluate:
            https://pypi.org/project/evaluate

        .. _toxicity:
            https://huggingface.co/spaces/evaluate-measurement/toxicity

        .. _torch:
            https://pytorch.org/get-started/locally/

        .. _transformers:
            https://huggingface.co/docs/transformers/installation

        .. _ari_grade_level:
            https://en.wikipedia.org/wiki/Automated_readability_index

        .. _flesch_kincaid_grade_level:
            https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level

        .. _textstat:
            https://pypi.org/project/textstat

     - For retriever models, the default evaluator logs:
        - **metrics**: :mod:`precision_at_k(k) <mlflow.metrics.precision_at_k>`,
          :mod:`recall_at_k(k) <mlflow.metrics.recall_at_k>` and
          :mod:`ndcg_at_k(k) <mlflow.metrics.ndcg_at_k>` - all have a default value of
          ``retriever_k`` = 3.
        - **artifacts**: A JSON file containing the inputs, outputs, targets, and per-row metrics
          of the model in tabular format.

     - For sklearn models, the default evaluator additionally logs the model's evaluation criterion
       (e.g. mean accuracy for a classifier) computed by `model.score` method.

     - The metrics/artifacts listed above are logged to the active MLflow run.
       If no active run exists, a new MLflow run is created for logging these metrics and
       artifacts. Note that no metrics/artifacts are logged for the ``baseline_model``.

     - Additionally, information about the specified dataset - hash, name (if specified), path
       (if specified), and the UUID of the model that evaluated it - is logged to the
       ``mlflow.datasets`` tag.

     - The available ``evaluator_config`` options for the default evaluator include:
        - **log_model_explainability**: A boolean value specifying whether or not to log model
          explainability insights, default value is True.
        - **explainability_algorithm**: A string to specify the SHAP Explainer algorithm for model
          explainability. Supported algorithm includes: 'exact', 'permutation', 'partition',
          'kernel'.
          If not set, ``shap.Explainer`` is used with the "auto" algorithm, which chooses the best
          Explainer based on the model.
        - **explainability_nsamples**: The number of sample rows to use for computing model
          explainability insights. Default value is 2000.
        - **explainability_kernel_link**: The kernel link function used by shap kernal explainer.
          Available values are "identity" and "logit". Default value is "identity".
        - **max_classes_for_multiclass_roc_pr**:
          For multiclass classification tasks, the maximum number of classes for which to log
          the per-class ROC curve and Precision-Recall curve. If the number of classes is
          larger than the configured maximum, these curves are not logged.
        - **metric_prefix**: An optional prefix to prepend to the name of each metric and artifact
          produced during evaluation.
        - **log_metrics_with_dataset_info**: A boolean value specifying whether or not to include
          information about the evaluation dataset in the name of each metric logged to MLflow
          Tracking during evaluation, default value is True.
        - **pos_label**: If specified, the positive label to use when computing classification
          metrics such as precision, recall, f1, etc. for binary classification models. For
          multiclass classification and regression models, this parameter will be ignored.
        - **average**: The averaging method to use when computing classification metrics such as
          precision, recall, f1, etc. for multiclass classification models
          (default: ``'weighted'``). For binary classification and regression models, this
          parameter will be ignored.
        - **sample_weights**: Weights for each sample to apply when computing model performance
          metrics.
        - **col_mapping**: A dictionary mapping column names in the input dataset or output
          predictions to column names used when invoking the evaluation functions.
        - **retriever_k**: A parameter used when ``model_type="retriever"`` as the number of
          top-ranked retrieved documents to use when computing the built-in metric
          :mod:`precision_at_k(k) <mlflow.metrics.precision_at_k>`,
          :mod:`recall_at_k(k) <mlflow.metrics.recall_at_k>` and
          :mod:`ndcg_at_k(k) <mlflow.metrics.ndcg_at_k>`. Default value is 3. For all other
          model types, this parameter will be ignored.

     - Limitations of evaluation dataset:
        - For classification tasks, dataset labels are used to infer the total number of classes.
        - For binary classification tasks, the negative label value must be 0 or -1 or False, and
          the positive label value must be 1 or True.

     - Limitations of metrics/artifacts computation:
        - For classification tasks, some metric and artifact computations require the model to
          output class probabilities. Currently, for scikit-learn models, the default evaluator
          calls the ``predict_proba`` method on the underlying model to obtain probabilities. For
          other model types, the default evaluator does not compute metrics/artifacts that require
          probability outputs.

     - Limitations of default evaluator logging model explainability insights:
        - The ``shap.Explainer`` ``auto`` algorithm uses the ``Linear`` explainer for linear models
          and the ``Tree`` explainer for tree models. Because SHAP's ``Linear`` and ``Tree``
          explainers do not support multi-class classification, the default evaluator falls back to
          using the ``Exact`` or ``Permutation`` explainers for multi-class classification tasks.
        - Logging model explainability insights is not currently supported for PySpark models.
        - The evaluation dataset label values must be numeric or boolean, all feature values
          must be numeric, and each feature column must only contain scalar values.

     - Limitations when environment restoration is enabled:
        - When environment restoration is enabled for the evaluated model (i.e. a non-local
          ``env_manager`` is specified), the model is loaded as a client that invokes a MLflow
          Model Scoring Server process in an independent Python environment with the model's
          training time dependencies installed. As such, methods like ``predict_proba`` (for
          probability outputs) or ``score`` (computes the evaluation criterian for sklearn models)
          of the model become inaccessible and the default evaluator does not compute metrics or
          artifacts that require those methods.
        - Because the model is an MLflow Model Server process, SHAP explanations are slower to
          compute. As such, model explainaibility is disabled when a non-local ``env_manager``
          specified, unless the ``evaluator_config`` option **log_model_explainability** is
          explicitly set to ``True``.

    Args:
        model: Optional. If specified, it should be one of the following:

            - A pyfunc model instance
            - A URI referring to a pyfunc model
            - A URI referring to an MLflow Deployments endpoint e.g. ``"endpoints:/my-chat"``
            - A callable function: This function should be able to take in model input and
              return predictions. It should follow the signature of the
              :py:func:`predict <mlflow.pyfunc.PyFuncModel.predict>` method. Here's an example
              of a valid function:

              ..code-block:: python

                  model = mlflow.pyfunc.load_model(model_uri)

                  def fn(model_input):
                      return model.predict(model_input)

            If omitted, it indicates a static dataset will be used for evaluation instead of a
            model.  In this case, the ``data`` argument must be a Pandas DataFrame or an mlflow
            PandasDataset that contains model outputs, and the ``predictions`` argument must be the
            name of the column in ``data`` that contains model outputs.

        data: One of the
            following:

            - A numpy array or list of evaluation features, excluding labels.
            - A Pandas DataFrame containing evaluation features, labels, and optionally model
                outputs. Model outputs are required to be provided when model is unspecified.
                If ``feature_names`` argument not specified, all columns except for the label
                column and model_output column are regarded as feature columns. Otherwise,
                only column names present in ``feature_names`` are regarded as feature columns.
            -  A Spark DataFrame containing evaluation features and labels. If
                ``feature_names`` argument not specified, all columns except for the label
                column are regarded as feature columns. Otherwise, only column names present in
                ``feature_names`` are regarded as feature columns. Only the first 10000 rows in
                the Spark DataFrame will be used as evaluation data.
            - A :py:class:`mlflow.data.dataset.Dataset` instance containing evaluation
                features, labels, and optionally model outputs. Model outputs are only supported
                with a PandasDataset. Model outputs are required when model is unspecified, and
                should be specified via the ``predictions`` prerty of the PandasDataset.

        targets: If ``data`` is a numpy array or list, a numpy array or list of evaluation
            labels. If ``data`` is a DataFrame, the string name of a column from ``data``
            that contains evaluation labels. Required for classifier and regressor models,
            but optional for question-answering, text-summarization, and text models. If
            ``data`` is a :py:class:`mlflow.data.dataset.Dataset` that defines targets,
            then ``targets`` is optional.

        predictions: Optional. The name of the column that contains model outputs.

            - When ``model`` is specified and outputs multiple columns, ``predictions`` can be used
              to specify the name of the column that will be used to store model outputs for
              evaluation.
            - When ``model`` is not specified and ``data`` is a pandas dataframe,
              ``predictions`` can be used to specify the name of the column in ``data`` that
              contains model outputs.

            .. code-block:: python
                :caption: Example usage of predictions

                # Evaluate a model that outputs multiple columns
                data = pd.DataFrame({"question": ["foo"]})


                def model(inputs):
                    return pd.DataFrame({"answer": ["bar"], "source": ["baz"]})


                results = evaluate(model=model, data=data, predictions="answer", ...)

                # Evaluate a static dataset
                data = pd.DataFrame({"question": ["foo"], "answer": ["bar"], "source": ["baz"]})
                results = evaluate(data=data, predictions="answer", ...)

        model_type: (Optional) A string describing the model type. The default evaluator
            supports the following model types:

            - ``'classifier'``
            - ``'regressor'``
            - ``'question-answering'``
            - ``'text-summarization'``
            - ``'text'``
            - ``'retriever'``

            If no ``model_type`` is specified, then you must provide a a list of
            metrics to compute via the ``extra_metrics`` param.

            .. note::
                ``'question-answering'``, ``'text-summarization'``, ``'text'``, and
                ``'retriever'`` are experimental and may be changed or removed in a
                future release.

        inference_params: (Optional) A dictionary of inference parameters to be passed to the model
            when making predictions, such as ``{"max_tokens": 100}``. This is only used when
            the ``model`` is an MLflow Deployments endpoint URI e.g. ``"endpoints:/my-chat"``

        dataset_path: (Optional) The path where the data is stored. Must not contain double
            quotes (``“``). If specified, the path is logged to the ``mlflow.datasets``
            tag for lineage tracking purposes.

        feature_names: (Optional) A list. If the ``data`` argument is a numpy array or list,
            ``feature_names`` is a list of the feature names for each feature. If
            ``feature_names=None``, then the ``feature_names`` are generated using the
            format ``feature_{feature_index}``. If the ``data`` argument is a Pandas
            DataFrame or a Spark DataFrame, ``feature_names`` is a list of the names
            of the feature columns in the DataFrame. If ``feature_names=None``, then
            all columns except the label column and the predictions column are
            regarded as feature columns.

        evaluators: The name of the evaluator to use for model evaluation, or a list of
            evaluator names. If unspecified, all evaluators capable of evaluating the
            specified model on the specified dataset are used. The default evaluator
            can be referred to by the name ``"default"``. To see all available
            evaluators, call :py:func:`mlflow.models.list_evaluators`.

        evaluator_config: A dictionary of additional configurations to supply to the evaluator.
            If multiple evaluators are specified, each configuration should be
            supplied as a nested dictionary whose key is the evaluator name.

        extra_metrics:
            (Optional) A list of :py:class:`EvaluationMetric <mlflow.models.EvaluationMetric>`
            objects.  These metrics are computed in addition to the default metrics associated with
            pre-defined `model_type`, and setting `model_type=None` will only compute the metrics
            specified in `extra_metrics`. See the `mlflow.metrics` module for more information about
            the builtin metrics and how to define extra metrics.

            .. code-block:: python
                :caption: Example usage of extra metrics

                import mlflow
                import numpy as np


                def root_mean_squared_error(eval_df, _builtin_metrics):
                    return np.sqrt((np.abs(eval_df["prediction"] - eval_df["target"]) ** 2).mean)


                rmse_metric = mlflow.models.make_metric(
                    eval_fn=root_mean_squared_error,
                    greater_is_better=False,
                )
                mlflow.evaluate(..., extra_metrics=[rmse_metric])

        custom_artifacts:
            (Optional) A list of custom artifact functions with the following signature:

            .. code-block:: python

                def custom_artifact(
                    eval_df: Union[pandas.Dataframe, pyspark.sql.DataFrame],
                    builtin_metrics: Dict[str, float],
                    artifacts_dir: str,
                ) -> Dict[str, Any]:
                    """
                    Args:
                        eval_df:
                            A Pandas or Spark DataFrame containing ``prediction`` and ``target``
                            column.  The ``prediction`` column contains the predictions made by the
                            model.  The ``target`` column contains the corresponding labels to the
                            predictions made on that row.
                        builtin_metrics:
                            A dictionary containing the metrics calculated by the default evaluator.
                            The keys are the names of the metrics and the values are the scalar
                            values of the metrics. Refer to the DefaultEvaluator behavior section
                            for what metrics will be returned based on the type of model (i.e.
                            classifier or regressor).
                        artifacts_dir:
                            A temporary directory path that can be used by the custom artifacts
                            function to temporarily store produced artifacts. The directory will be
                            deleted after the artifacts are logged.

                    Returns:
                        A dictionary that maps artifact names to artifact objects
                        (e.g. a Matplotlib Figure) or to artifact paths within ``artifacts_dir``.
                    """
                    ...

            Object types that artifacts can be represented as:

                - A string uri representing the file path to the artifact. MLflow will infer the
                  type of the artifact based on the file extension.
                - A string representation of a JSON object. This will be saved as a .json artifact.
                - Pandas DataFrame. This will be resolved as a CSV artifact.
                - Numpy array. This will be saved as a .npy artifact.
                - Matplotlib Figure. This will be saved as an image artifact. Note that
                  ``matplotlib.pyplot.savefig`` is called behind the scene with default
                  configurations.
                  To customize, either save the figure with the desired configurations and return
                  its file path or define customizations through environment variables in
                  ``matplotlib.rcParams``.
                - Other objects will be attempted to be pickled with the default protocol.

            .. code-block:: python
                :caption: Example usage of custom artifacts

                import mlflow
                import matplotlib.pyplot as plt


                def scatter_plot(eval_df, builtin_metrics, artifacts_dir):
                    plt.scatter(eval_df["prediction"], eval_df["target"])
                    plt.xlabel("Targets")
                    plt.ylabel("Predictions")
                    plt.title("Targets vs. Predictions")
                    plt.savefig(os.path.join(artifacts_dir, "example.png"))
                    plt.close()
                    return {"pred_target_scatter": os.path.join(artifacts_dir, "example.png")}


                def pred_sample(eval_df, _builtin_metrics, _artifacts_dir):
                    return {"pred_sample": pred_sample.head(10)}


                mlflow.evaluate(..., custom_artifacts=[scatter_plot, pred_sample])

        validation_thresholds: (Optional) A dictionary of metric name to
            :py:class:`mlflow.models.MetricThreshold` used for model validation. Each metric name
            must either be the name of a builtin metric or the name of a metric defined in the
            ``extra_metrics`` parameter.

            .. code-block:: python
                :caption: Example of Model Validation

                from mlflow.models import MetricThreshold

                thresholds = {
                    "accuracy_score": MetricThreshold(
                        # accuracy should be >=0.8
                        threshold=0.8,
                        # accuracy should be at least 5 percent greater than baseline model accuracy
                        min_absolute_change=0.05,
                        # accuracy should be at least 0.05 greater than baseline model accuracy
                        min_relative_change=0.05,
                        greater_is_better=True,
                    ),
                }

                with mlflow.start_run():
                    mlflow.evaluate(
                        model=your_candidate_model,
                        data,
                        targets,
                        model_type,
                        dataset_name,
                        evaluators,
                        validation_thresholds=thresholds,
                        baseline_model=your_baseline_model,
                    )

            See :ref:`the Model Validation documentation <model-validation>`
            for more details.

        baseline_model: (Optional) A string URI referring to an MLflow model with the pyfunc
            flavor. If specified, the candidate ``model`` is compared to this
            baseline for model validation purposes.

        env_manager: Specify an environment manager to load the candidate ``model`` and
            ``baseline_model`` in isolated Python environments and restore their
            dependencies. Default value is ``local``, and the following values are
            supported:

            - ``virtualenv``: (Recommended) Use virtualenv to restore the python
              environment that was used to train the model.
            - ``conda``:  Use Conda to restore the software environment that was used
              to train the model.
            - ``local``: Use the current Python environment for model inference, which
              may differ from the environment used to train the model and may lead to
              errors or invalid predictions.

        model_config: the model configuration to use for loading the model with pyfunc. Inspect
            the model's pyfunc flavor to know which keys are supported for your
            specific model. If not indicated, the default model configuration
            from the model is used (if any).
        baseline_config: the model configuration to use for loading the baseline
            model. If not indicated, the default model configuration
            from the baseline model is used (if any).

    Returns:
        An :py:class:`mlflow.models.EvaluationResult` instance containing
        metrics of candidate model and baseline model, and artifacts of candidate model.
    '''
    from mlflow.pyfunc import PyFuncModel, _load_model_or_server, _ServedPyFuncModel
    from mlflow.utils import env_manager as _EnvManager

    # Inference params are currently only supported for passing a deployment endpoint as the model.
    # TODO: We should support inference_params for other model types

    if inference_params is not None and not _is_model_deployment_endpoint_uri(model):
        raise MlflowException(
            message="The inference_params argument can only be specified when the model "
            "is an MLflow Deployments endpoint URI like `endpoints:/my-chat`",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if evaluator_config is not None:
        col_mapping = evaluator_config.get("col_mapping", {})

        if isinstance(targets, str):
            targets = col_mapping.get(targets, targets)

        if isinstance(predictions, str):
            predictions = col_mapping.get(predictions, predictions)

    if data is None:
        raise MlflowException(
            message="The data argument cannot be None.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    _EnvManager.validate(env_manager)

    # If Dataset is provided, the targets can only be specified by the Dataset,
    # not the targets parameters of the mlflow.evaluate() API.
    if isinstance(data, Dataset) and targets is not None:
        raise MlflowException(
            message="The top-level targets parameter should not be specified since a Dataset "
            "is used. Please only specify the targets column name in the Dataset. For example: "
            "`data = mlflow.data.from_pandas(df=X.assign(y=y), targets='y')`. "
            "Meanwhile, please specify `mlflow.evaluate(..., targets=None, ...)`.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    # If Dataset is provided and model is None, then the predictions can only be specified by the
    # Dataset, not the predictions parameters of the mlflow.evaluate() API.
    if isinstance(data, Dataset) and model is None and predictions is not None:
        raise MlflowException(
            message="The top-level predictions parameter should not be specified since a Dataset "
            "is used. Please only specify the predictions column name in the Dataset. For example:"
            " `data = mlflow.data.from_pandas(df=X.assign(y=y), predictions='y')`"
            "Meanwhile, please specify `mlflow.evaluate(..., predictions=None, ...)`.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    # If Dataset is provided and model is specified, then the data.predictions cannot be specified.
    if (
        isinstance(data, Dataset)
        and model is not None
        and getattr(data, "predictions", None) is not None
    ):
        raise MlflowException(
            message="The predictions parameter should not be specified in the Dataset since a "
            "model is specified. Please remove the predictions column from the Dataset.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if model_type in [_ModelType.REGRESSOR, _ModelType.CLASSIFIER]:
        if isinstance(data, Dataset):
            if getattr(data, "targets", None) is not None:
                targets = data.targets
            else:
                raise MlflowException(
                    message="The targets column name must be specified in the provided Dataset "
                    f"for {model_type} models. For example: "
                    "`data = mlflow.data.from_pandas(df=X.assign(y=y), targets='y')`",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            if targets is None:
                raise MlflowException(
                    f"The targets argument must be specified for {model_type} models.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
    elif model_type is None:
        if not extra_metrics:
            raise MlflowException(
                message="The extra_metrics argument must be specified model_type is None.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    if isinstance(model, str):
        if _is_model_deployment_endpoint_uri(model):
            model = _get_model_from_deployment_endpoint_uri(model, inference_params)
        else:
            model = _load_model_or_server(model, env_manager, model_config)
    elif env_manager != _EnvManager.LOCAL:
        raise MlflowException(
            message="The model argument must be a string URI referring to an MLflow model when a "
            "non-local env_manager is specified.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    elif isinstance(model, PyFuncModel):
        if model_config:
            raise MlflowException(
                message="Indicating ``model_config`` when passing a `PyFuncModel`` object as "
                "model argument is not allowed. If you need to change the model configuration "
                "for the evaluation model, use "
                "``mlflow.pyfunc.load_model(model_uri, model_config=<value>)`` and indicate "
                "the desired configuration there.",
                error_code=INVALID_PARAMETER_VALUE,
            )
    elif callable(model):
        model = _get_model_from_function(model)
    elif model is not None:
        raise MlflowException(
            message="The model argument must be a string URI referring to an MLflow model, "
            "an MLflow Deployments endpoint URI, an instance of `mlflow.pyfunc.PyFuncModel`, "
            "a function, or None.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if validation_thresholds:
        try:
            assert type(validation_thresholds) is dict
            for key in validation_thresholds.keys():
                assert type(key) is str
            for threshold in validation_thresholds.values():
                assert isinstance(threshold, MetricThreshold)
        except AssertionError:
            raise MlflowException(
                message="The validation thresholds argument must be a dictionary that maps strings "
                "to MetricThreshold objects.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    if isinstance(baseline_model, str):
        baseline_model = _load_model_or_server(
            baseline_model, env_manager, model_config=baseline_config
        )
    elif baseline_model is not None:
        raise MlflowException(
            message="The baseline model argument must be a string URI referring to an "
            "MLflow model.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    elif _model_validation_contains_model_comparison(validation_thresholds):
        raise MlflowException(
            message="The baseline model argument is None. The baseline model must be specified "
            "when model comparison thresholds (min_absolute_change, min_relative_change) "
            "are specified.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    (
        evaluator_name_list,
        evaluator_name_to_conf_map,
    ) = _normalize_evaluators_and_evaluator_config_args(evaluators, evaluator_config)

    with _start_run_or_reuse_active_run() as run_id:
        if not isinstance(data, Dataset):
            # Convert data to `mlflow.data.dataset.Dataset`.
            if model is None:
                data = convert_data_to_mlflow_dataset(
                    data=data, targets=targets, predictions=predictions
                )
            else:
                data = convert_data_to_mlflow_dataset(data=data, targets=targets)

        from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin

        if isinstance(data, Dataset) and issubclass(data.__class__, PyFuncConvertibleDatasetMixin):
            dataset = data.to_evaluation_dataset(dataset_path, feature_names)
            if evaluator_name_to_conf_map and evaluator_name_to_conf_map.get("default", None):
                context = evaluator_name_to_conf_map["default"].get("metric_prefix", None)
            else:
                context = None
            client = MlflowClient()
            tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value=context)] if context else []
            dataset_input = DatasetInput(dataset=data._to_mlflow_entity(), tags=tags)
            client.log_inputs(run_id, [dataset_input])
        else:
            dataset = EvaluationDataset(
                data,
                targets=targets,
                path=dataset_path,
                feature_names=feature_names,
                predictions=predictions,
            )
        predictions_expected_in_model_output = predictions if model is not None else None

        try:
            evaluate_result = _evaluate(
                model=model,
                model_type=model_type,
                dataset=dataset,
                run_id=run_id,
                evaluator_name_list=evaluator_name_list,
                evaluator_name_to_conf_map=evaluator_name_to_conf_map,
                custom_metrics=custom_metrics,
                extra_metrics=extra_metrics,
                custom_artifacts=custom_artifacts,
                baseline_model=baseline_model,
                predictions=predictions_expected_in_model_output,
            )
        finally:
            if isinstance(model, _ServedPyFuncModel):
                os.kill(model.pid, signal.SIGTERM)
            if isinstance(baseline_model, _ServedPyFuncModel):
                os.kill(baseline_model.pid, signal.SIGTERM)

        if not validation_thresholds:
            return evaluate_result

        _logger.info("Validating generated model metrics")
        _validate(
            validation_thresholds,
            evaluate_result.metrics,
            evaluate_result.baseline_model_metrics,
        )
        _logger.info("Model validation passed!")

        return evaluate_result
