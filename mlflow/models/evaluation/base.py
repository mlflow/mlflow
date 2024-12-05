import inspect
import json
import keyword
import logging
import os
import pathlib
import signal
import urllib
import urllib.parse
import warnings
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from inspect import Parameter, Signature
from types import FunctionType
from typing import Any, Optional, Union

import mlflow
from mlflow.data.dataset import Dataset
from mlflow.data.evaluation_dataset import (
    EvaluationDataset,
    convert_data_to_mlflow_dataset,
)
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.utils.trace import configure_autologging_for_evaluation
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
    # This model type is used for Mosaic AI Agent evaluation and only available in Databricks
    # https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html
    DATABRICKS_AGENT = "databricks-agent"

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
        genai_metric_args: (Optional) A dictionary containing arguments specified by users
            when calling make_genai_metric or make_genai_metric_from_prompt. Those args
            are persisted so that we can deserialize the same metric object later.
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
        genai_metric_args=None,
    ):
        self.eval_fn = eval_fn
        self.name = name
        self.greater_is_better = greater_is_better
        self.long_name = long_name or name
        self.version = version
        self.metric_details = metric_details
        self.metric_metadata = metric_metadata
        self.genai_metric_args = genai_metric_args

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


# NB: we need this function because we cannot modify the signature of
# a class's __call__ method after the class has been defined.
# This is also useful to distinguish between the metric signatures with different eval_fn signatures
def _generate_eval_metric_class(eval_fn, require_strict_signature=False):
    """
    Dynamically generate a GenAIEvaluationMetric class that can be used to evaluate the metric
    on the given input data. The generated class is callable with a __call__ method that
    takes the arguments specified in the signature of the eval_fn function.

    Args:
        eval_fn: the evaluation function of the EvaluationMetric.
        require_strict_signature: (Optional) Whether the eval_fn needs to follow a strict signature.
            If True, then the eval_fn must follow below signature:

                .. code-block:: python

                    def eval_fn(
                        predictions: "pd.Series",
                        metrics: Dict[str, MetricValue],
                        inputs: "pd.Series",
                        *args,
                    ) -> MetricValue:
                        pass

            When generating a metric from `make_genai_metric`, this should be set to True.
            Default to False.

    Returns:
        A dynamically generated callable CallableEvaluationMetric class.
    """
    from mlflow.metrics.base import MetricValue

    if require_strict_signature:
        allowed_kwargs_names = [
            param_name
            for param_name in inspect.signature(eval_fn).parameters.keys()
            if param_name not in ["predictions", "metrics", "inputs"]
        ]

        def genai_call_method(
            self,
            *,
            predictions: Union[pd.Series, str, list[str]],
            inputs: Union[pd.Series, str, list[str]],
            metrics: Optional[dict[str, MetricValue]] = None,
            **kwargs,
        ) -> MetricValue:
            if missed_kwargs := set(allowed_kwargs_names) - set(kwargs.keys()):
                raise MlflowException.invalid_parameter_value(
                    f"Missing required arguments: {missed_kwargs}",
                )
            if extra_kwargs := set(kwargs.keys()) - set(allowed_kwargs_names):
                raise MlflowException.invalid_parameter_value(
                    f"Unexpected arguments: {extra_kwargs}",
                )
            return self.eval_fn(
                _convert_val_to_pd_Series(predictions, "predictions"),
                metrics or {},
                _convert_val_to_pd_Series(inputs, "inputs"),
                # Note: based on https://github.com/mlflow/mlflow/blob/4fef77afdbe4d76302cb0b1aad2bd72b5cde64e9/mlflow/metrics/genai/genai_metric.py#L49-L53
                # the extra params passed https://github.com/mlflow/mlflow/blob/4fef77afdbe4d76302cb0b1aad2bd72b5cde64e9/mlflow/metrics/genai/genai_metric.py#L513
                # should always be pandas Series
                *[
                    _convert_val_to_pd_Series(kwargs[arg_name], arg_name)
                    for arg_name in allowed_kwargs_names
                ],
            )

        genai_call_method.__signature__ = Signature(
            parameters=[
                Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter(
                    "predictions",
                    Parameter.KEYWORD_ONLY,
                    annotation=Union[pd.Series, str, list[str]],
                ),
                Parameter(
                    "inputs",
                    Parameter.KEYWORD_ONLY,
                    annotation=Union[pd.Series, str, list[str]],
                ),
                Parameter(
                    "metrics",
                    Parameter.KEYWORD_ONLY,
                    annotation=Optional[dict[str, MetricValue]],
                    default=None,
                ),
                *[
                    Parameter(
                        name, Parameter.KEYWORD_ONLY, annotation=Union[pd.Series, str, list[str]]
                    )
                    for name in allowed_kwargs_names
                ],
            ]
        )
        genai_call_method.__doc__ = f"""
            Evaluate the metric on the given inputs and predictions.
            Note: only keyword arguments are supported.

            Args:
                predictions: predictions made by the model.
                inputs: inputs used to make the predictions.
                metrics: metrics calculated by the default evaluator.
                kwargs: additional arguments used to compute the metric.
                    Required arguments: {allowed_kwargs_names}

            Returns:
                evaluation result as MetricValue object.
            """
        call_method = genai_call_method

    else:

        def _call_method(
            self,
            **kwargs,
        ) -> MetricValue:
            return self.eval_fn(**kwargs)

        allowed_kwargs_params = inspect.signature(eval_fn).parameters
        _call_method.__signature__ = Signature(
            parameters=[
                Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
                *[
                    Parameter(
                        name,
                        Parameter.KEYWORD_ONLY,
                        annotation=allowed_kwargs_params[name].annotation,
                    )
                    for name in allowed_kwargs_params.keys()
                ],
            ]
        )
        _call_method.__doc__ = f"""
            Evaluate the metric on the given inputs and predictions.
            Note: only keyword arguments are supported.

            Args:
                kwargs: additional arguments used to compute the metric.
                    Required arguments: {list(allowed_kwargs_params.keys())}

            Returns:
                evaluation result as MetricValue object.
            """
        call_method = _call_method

    return type(
        "CallableEvaluationMetric",
        (EvaluationMetric,),
        {"__call__": call_method},
    )


def _convert_val_to_pd_Series(val, name):
    if val is not None and not isinstance(val, pd.Series):
        if isinstance(val, str):
            return pd.Series([val])
        elif isinstance(val, list):
            return pd.Series(val)
        else:
            raise TypeError(
                f"Expected {name} to be a string, list, or Pandas Series, got {type(val)}"
            )
    return val


def make_metric(
    *,
    eval_fn,
    greater_is_better,
    name=None,
    long_name=None,
    version=None,
    metric_details=None,
    metric_metadata=None,
    genai_metric_args=None,
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
        genai_metric_args: (Optional) A dictionary containing arguments specified by users
            when calling make_genai_metric or make_genai_metric_from_prompt. Those args
            are persisted so that we can deserialize the same metric object later.

    .. seealso::

        - :py:class:`mlflow.models.EvaluationMetric`
        - :py:func:`mlflow.evaluate`
    '''
    return _make_metric(
        eval_fn=eval_fn,
        greater_is_better=greater_is_better,
        name=name,
        long_name=long_name,
        version=version,
        metric_details=metric_details,
        metric_metadata=metric_metadata,
        genai_metric_args=genai_metric_args,
        require_strict_signature=False,
    )


def _make_metric(
    *,
    eval_fn,
    greater_is_better,
    name=None,
    long_name=None,
    version=None,
    metric_details=None,
    metric_metadata=None,
    genai_metric_args=None,
    require_strict_signature=False,
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
        genai_metric_args: (Optional) A dictionary containing arguments specified by users
            when calling make_genai_metric or make_genai_metric_from_prompt. Those args
            are persisted so that we can deserialize the same metric object later.
        require_strict_signature: (Optional) Whether the eval_fn needs to follow a strict signature.
            If True, then the eval_fn must follow below signature:

                .. code-block:: python

                    def eval_fn(
                        predictions: "pd.Series",
                        metrics: Dict[str, MetricValue],
                        inputs: "pd.Series",
                        *args,
                    ) -> MetricValue:
                        pass

            When generating a metric from `make_genai_metric`, this should be set to True.
            Default to False.

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

    return _generate_eval_metric_class(eval_fn, require_strict_signature=require_strict_signature)(
        eval_fn=eval_fn,
        name=name,
        greater_is_better=greater_is_better,
        long_name=long_name,
        version=version,
        metric_details=metric_details,
        metric_metadata=metric_metadata,
        genai_metric_args=genai_metric_args,
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

    def __init__(self, metrics, artifacts, run_id=None):
        self._metrics = metrics
        self._artifacts = artifacts
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
    def metrics(self) -> dict[str, Any]:
        """
        A dictionary mapping scalar metric names to scalar metric values
        """
        return self._metrics

    @property
    def artifacts(self) -> dict[str, "mlflow.models.EvaluationArtifact"]:
        """
        A dictionary mapping standardized artifact names (e.g. "roc_data") to
        artifact content and location information
        """
        return self._artifacts

    @experimental
    @property
    def tables(self) -> dict[str, "pd.DataFrame"]:
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
    @classmethod
    @abstractmethod
    def can_evaluate(cls, *, model_type, evaluator_config, **kwargs) -> bool:
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
            model: A pyfunc model instance. If None, the model output is supposed to be found in
                ``dataset.predictions_data``.
            custom_metrics: Deprecated. Use ``extra_metrics`` instead.
            extra_metrics: A list of :py:class:`EvaluationMetric` objects.
            custom_artifacts: A list of callable custom artifact functions.
            predictions: The column name of the model output column that is used for evaluation.
                This is only used when a model returns a pandas dataframe that contains
                multiple columns.
            kwargs: For forwards compatibility, a placeholder for additional arguments that
                may be added to the evaluation interface in the future.

        Returns:
            A :py:class:`mlflow.models.EvaluationResult` instance containing
            evaluation metrics and artifacts for the model.
        """


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


# NB: We often pass around evaluator name, config, and its instance together. Ideally, the
# evaluator class should have name and config as class attributes, however, it was not
# designed that way. Adding them while keeping backward compatibility is not trivial.
# So, we use a dataclass to bundle them together.
@dataclass
class EvaluatorBundle:
    name: str
    evaluator: ModelEvaluator
    config: dict[str, Any]


def _resolve_default_evaluator(model_type, default_config) -> list[EvaluatorBundle]:
    """
    Determine which built-in evaluators should be used for the given model type by default.

    Previously, MLflow evaluate API only had a single "default" evaluator used for all models like
    classifier, regressor, etc. We split it into multiple built-in evaluators for different model
    types for maintainability, but in order to maintain backward compatibility, we need to map
    the "default" provided by users to the correct built-in evaluators.

    Args:
        model_type: A string describing the model type (e.g., "regressor", "classifier", …).
        default_config: A dictionary of configurations for the "default" evaluator. If any
            non-default built-in evaluator is applicable, this config will be applied to them.
    """
    from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry

    builtin_evaluators = []
    for name in _model_evaluation_registry._registry:
        evaluator = _model_evaluation_registry.get_evaluator(name)
        if (
            name != "default"
            and _model_evaluation_registry.is_builtin(name)
            and evaluator.can_evaluate(model_type=model_type, evaluator_config=default_config)
        ):
            builtin_evaluators.append(EvaluatorBundle(name, evaluator, default_config))

    # We should use DefaultEvaluator only if there is no other built-in evaluator applicable.
    if not builtin_evaluators:
        default_evaluator = _model_evaluation_registry.get_evaluator("default")
        builtin_evaluators = [EvaluatorBundle("default", default_evaluator, default_config)]

    return builtin_evaluators


def resolve_evaluators_and_configs(
    evaluators: Union[str, list[str], None],
    evaluator_config: Union[dict[str, Any], None],
    model_type: Optional[str] = None,
) -> list[EvaluatorBundle]:
    """
    The `evaluators` and `evaluator_config` arguments of the `evaluate` API can be specified
    in multiple ways. This function normalizes the arguments into a single format for easier
    downstream processing.

    Args:
        evaluators: A string or a list of strings specifying the evaluators to use for model
            evaluation. If None, all available evaluators will be used.
        evaluator_config: A dictionary containing configuration items for the evaluators.
        model_type: A string describing the model type (e.g., "regressor", "classifier", …).

    Returns:
        A list of EvaluatorBundle that contains name, evaluator, config for each evaluator.
    """
    from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry as rg

    # NB: The `databricks-agents` package must be installed to use the 'databricks-agent' model
    # type. Ideally this check should be done in the 'databricks-agent' evaluator implementation,
    # but we need to do it here because the code won't reach the evaluator implementation if the
    # package is not installed.
    if model_type == _ModelType.DATABRICKS_AGENT:
        try:
            import databricks.agents  # noqa: F401
        except ImportError as e:
            raise MlflowException(
                message="Databricks Agents SDK must be installed to use the "
                f"`{_ModelType.DATABRICKS_AGENT}` model type. Run `pip install databricks-agents` "
                "to install the package and try again.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from e

    def check_nesting_config_dict(_evaluator_name_list, _evaluator_name_to_conf_map):
        return isinstance(_evaluator_name_to_conf_map, dict) and all(
            k in _evaluator_name_list and isinstance(v, dict)
            for k, v in _evaluator_name_to_conf_map.items()
        )

    if evaluators is None:
        # If no evaluators are specified, use all available evaluators.
        evaluators = list(rg._registry.keys())

        evaluator_config = evaluator_config or {}
        if evaluator_config is not None and not any(
            name in evaluator_config for name in evaluators
        ):
            # If evaluator config is passed but any of available evaluator key is not
            # in the evaluator config, we assume the evaluator config to be a flat dict,
            # which is globally applied to all evaluators.
            evaluator_config = {ev: evaluator_config for ev in evaluators}

        # Filter out evaluators that cannot evaluate the model type.
        resolved = []
        for name in evaluators:
            evaluator = rg.get_evaluator(name)
            config = evaluator_config.get(name, {})
            if evaluator.can_evaluate(model_type=model_type, evaluator_config=config):
                resolved.append(EvaluatorBundle(name=name, evaluator=evaluator, config=config))

        # If any of built-in evaluator can apply, skip "default" evaluator.
        default = next((ev for ev in resolved if ev.name == "default"), None)
        non_default_builtins = [
            ev for ev in resolved if ev.name != "default" and rg.is_builtin(ev.name)
        ]
        if default and non_default_builtins:
            resolved.remove(default)
            # Apply default config (passed like `evaluator_config={"default": config}`) to
            # non-default built-in evaluators (e.g., ClassifierEvaluator) if they don't have
            # explicitly specified configs. This is for backward compatibility where we only
            # had a single "default" evaluator used for all models.
            # For example, if the user passes this for a classifier model:
            #     evaluator_config = {"default": my_config}
            # it should be equivalent to
            #    evaluator_config = {"classifier": my_config, "shap": my_config}
            for ev in non_default_builtins:
                ev.config = ev.config or default.config

        return resolved

    elif isinstance(evaluators, str):
        # Single evaluator name specified
        if not (evaluator_config is None or isinstance(evaluator_config, dict)):
            raise MlflowException(
                message="If `evaluators` argument is the name of an evaluator, evaluator_config"
                " must be None or a dict containing config items for the evaluator.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        evaluator_config = evaluator_config or {}
        if evaluators == "default":
            # Previously we only had a single "default" evaluator used for all models.
            # We need to map "default" to the new dedicated builtin evaluators.
            return _resolve_default_evaluator(model_type, evaluator_config)
        elif rg.is_registered(evaluators):
            return [EvaluatorBundle(evaluators, rg.get_evaluator(evaluators), evaluator_config)]
        else:
            return []

    elif isinstance(evaluators, list):
        if evaluator_config is not None and not check_nesting_config_dict(
            evaluators, evaluator_config
        ):
            raise MlflowException(
                message="If `evaluators` argument is an evaluator name list, evaluator_config "
                "must be a dict containing mapping from evaluator name to individual "
                "evaluator config dict.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        evaluator_config = evaluator_config or {}

        # Previously we only had a single "default" evaluator used for all models.
        # We need to map "default" to the new dedicated builtin evaluators.
        resolved = []
        for name in evaluators:
            config = evaluator_config.get(name, {})
            if name == "default":
                builtin_evaluators = _resolve_default_evaluator(model_type, config)
                resolved.extend(builtin_evaluators)
            else:
                resolved.append(EvaluatorBundle(name, rg.get_evaluator(name), config))
        return resolved
    else:
        raise MlflowException(
            message="Invalid `evaluators` and `evaluator_config` arguments. "
            "Please refer to the documentation for correct usage.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _model_validation_contains_model_comparison(validation_thresholds):
    """
    Helper function for determining if validation_thresholds contains
    thresholds for model comparison: either min_relative_change or min_absolute_change
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


# DO NOT CHANGE THE ORDER OF THE ARGUMENTS
# The order of the arguments need to be preserved. You can add new arguments at the end
# of the argument list, but do not change the order of the existing arguments.
def _evaluate(
    *,
    model,
    model_type,
    dataset,
    run_id,
    # The `evaluator_name_list` and `evaluator_name_to_conf_map` are not used by MLflow at all,
    # but we need to keep these for backward compatibility.
    evaluator_name_list,
    evaluator_name_to_conf_map,
    custom_metrics,
    extra_metrics,
    custom_artifacts,
    predictions,
    evaluators,
):
    """
    The public API "evaluate" will verify argument first, and then pass normalized arguments
    to the _evaluate method.
    """
    global _last_failed_evaluator
    _last_failed_evaluator = None

    client = MlflowClient()

    model_uuid = getattr(model, "metadata", None)

    if model_uuid is not None:
        model_uuid = model_uuid.model_uuid
        dataset._log_dataset_tag(client, run_id, model_uuid)

    eval_results = []
    should_enable_tracing = model is not None  # Do not enable tracing if static dataset is provided
    for eval_ in evaluators:
        _logger.debug(f"Evaluating the model with the {eval_.name} evaluator.")
        _last_failed_evaluator = eval_.name
        if eval_.evaluator.can_evaluate(model_type=model_type, evaluator_config=eval_.config):
            with configure_autologging_for_evaluation(enable_tracing=should_enable_tracing):
                eval_result = eval_.evaluator.evaluate(
                    model=model,
                    model_type=model_type,
                    dataset=dataset,
                    run_id=run_id,
                    evaluator_config=eval_.config,
                    custom_metrics=custom_metrics,
                    extra_metrics=extra_metrics,
                    custom_artifacts=custom_artifacts,
                    predictions=predictions,
                )

            if eval_result is not None:
                eval_results.append(eval_result)

    _last_failed_evaluator = None

    if len(eval_results) == 0:
        raise MlflowException(
            message="The model could not be evaluated by any of the registered evaluators, please "
            "verify that the model type and other configs are set correctly.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    merged_eval_result = EvaluationResult({}, {}, None)

    for eval_result in eval_results:
        merged_eval_result.metrics.update(eval_result.metrics)
        merged_eval_result.artifacts.update(eval_result.artifacts)

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
    endpoint_uri: str, params: Optional[dict[str, Any]] = None
):
    from mlflow.metrics.genai.model_utils import _parse_model_uri
    from mlflow.pyfunc.model import ModelFromDeploymentEndpoint, _PythonModelPyfuncWrapper

    _, endpoint = _parse_model_uri(endpoint_uri)
    params = params or {}

    python_model = ModelFromDeploymentEndpoint(endpoint, params)
    return _PythonModelPyfuncWrapper(python_model, None, None)


def evaluate(  # noqa: D417
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
       artifacts.

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
        - **explainability_kernel_link**: The kernel link function used by shap kernel explainer.
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

              .. code-block:: python

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


                results = evaluate(
                    model=model,
                    data=data,
                    predictions="answer",
                    # other arguments if needed
                )

                # Evaluate a static dataset
                data = pd.DataFrame({"question": ["foo"], "answer": ["bar"], "source": ["baz"]})
                results = evaluate(
                    data=data,
                    predictions="answer",
                    # other arguments if needed
                )
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

        custom_metrics: Deprecated. Use ``extra_metrics`` instead.
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
                    return np.sqrt((np.abs(eval_df["prediction"] - eval_df["target"]) ** 2).mean())


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

        validation_thresholds: DEPRECATED. Please use :py:func:`mlflow.validate_evaluation_results`
            API instead for running model validation against baseline.

        baseline_model: DEPRECATED. Please use :py:func:`mlflow.validate_evaluation_results`
            API instead for running model validation against baseline.

        env_manager: Specify an environment manager to load the candidate ``model`` in
            isolated Python environments and restore their
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

        baseline_config: DEPRECATED. Please use :py:func:`mlflow.validate_evaluation_results`
            API instead for running model validation against baseline.

        inference_params: (Optional) A dictionary of inference parameters to be passed to the model
            when making predictions, such as ``{"max_tokens": 100}``. This is only used when
            the ``model`` is an MLflow Deployments endpoint URI e.g. ``"endpoints:/my-chat"``

    Returns:
        An :py:class:`mlflow.models.EvaluationResult` instance containing
        metrics of evaluating the model with the given dataset.
    '''
    from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry
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

    evaluators: list[EvaluatorBundle] = resolve_evaluators_and_configs(
        evaluators, evaluator_config, model_type
    )

    # NB: MLflow do not use either of these two variables. However, we need to pass these to
    # _evaluate() function for backward compatibility.
    evaluator_name_list = [evaluator.name for evaluator in evaluators]
    evaluator_name_to_conf_map = {evaluator.name: evaluator.config for evaluator in evaluators}

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

            # Use metrix_prefix configured for builtin evaluators as a dataset tag
            context = None
            for e in evaluators:
                if _model_evaluation_registry.is_builtin(e.name) and e.config.get("metric_prefix"):
                    context = e.config.get("metric_prefix")
                    break

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
                predictions=predictions_expected_in_model_output,
                evaluators=evaluators,
            )
        finally:
            if isinstance(model, _ServedPyFuncModel):
                os.kill(model.pid, signal.SIGTERM)

    # TODO: Remove this block in a future release when we
    # remove the deprecated arguments.
    if baseline_model is not None and validation_thresholds is not None:
        from mlflow.models.evaluation.validation import validate_evaluation_results

        warnings.warn(
            "Model validation functionality is moved from `mlflow.evaluate` to the "
            "`mlflow.validate_evaluation_results()` API. The "
            "`baseline_model` argument will be removed in a future release.",
            category=FutureWarning,
            stacklevel=2,
        )

        if isinstance(baseline_model, str):
            baseline_model = _load_model_or_server(
                baseline_model, env_manager, model_config=baseline_config
            )

        baseline_result = _evaluate(
            model=baseline_model,
            model_type=model_type,
            dataset=dataset,
            run_id=run_id,
            evaluator_name_list=evaluator_name_list,
            evaluator_name_to_conf_map=evaluator_name_to_conf_map,
            custom_metrics=custom_metrics,
            extra_metrics=extra_metrics,
            custom_artifacts=custom_artifacts,
            predictions=predictions_expected_in_model_output,
            evaluators=evaluators,
        )
        return validate_evaluation_results(
            validation_thresholds=validation_thresholds,
            candidate_result=evaluate_result,
            baseline_result=baseline_result,
        )

    return evaluate_result
