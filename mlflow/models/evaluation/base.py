import json
import logging
import math
import operator
import os
import pathlib
import signal
import struct
import sys
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
from mlflow.entities import RunTag
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
from mlflow.utils import _get_fully_qualified_class_name, _insecure_md5
from mlflow.utils.annotations import developer_stable, experimental
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.string_utils import generate_feature_name_if_not_string

try:
    # `numpy` and `pandas` are not required for `mlflow-skinny`.
    import numpy as np
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

    :param eval_fn:
        A function that computes the metric with the following signature:

        .. code-block:: python

            def eval_fn(
                predictions: pandas.Series,
                targets: pandas.Series,
                metrics: Dict[str, MetricValue],
                **kwargs,
            ) -> Union[float, MetricValue]:
                """
                :param predictions: A pandas Series containing the predictions made by the model.
                :param targets: (Optional) A pandas Series containing the corresponding labels
                    for the predictions made on that input.
                :param metrics: (Optional) A dictionary containing the metrics calculated by the
                    default evaluator.
                    The keys are the names of the metrics and the values are the metric values.
                    To access the MetricValue for the metrics calculated by the system, make sure
                    to specify the type hint for this parameter as Dict[str, MetricValue].
                    Refer to the DefaultEvaluator behavior section for what metrics
                    will be returned based on the type of model (i.e. classifier or regressor).
                :param kwargs: Includes a list of args that are used to compute the metric. These
                    args could information coming from input data, model outputs or parameters
                    specified in the `evaluator_config` argument of the `mlflow.evaluate` API.
                :return:
                    MetricValue with per-row scores, per-row justifications, and aggregate results.
                """
                ...

    :param name: The name of the metric.
    :param greater_is_better: Whether a higher value of the metric is better.
    :param long_name: (Optional) The long name of the metric. For example,
        ``"root_mean_squared_error"`` for ``"mse"``.
    :param version: (Optional) The metric version. For example ``v1``.
    '''

    def __init__(self, eval_fn, name, greater_is_better, long_name=None, version=None):
        self.eval_fn = eval_fn
        self.name = name
        self.greater_is_better = greater_is_better
        self.long_name = long_name or name
        self.version = version

    def __str__(self):
        if self.long_name:
            return (
                f"EvaluationMetric(name={self.name}, long_name={self.long_name}, "
                f"greater_is_better={self.greater_is_better})"
            )
        else:
            return f"EvaluationMetric(name={self.name}, greater_is_better={self.greater_is_better})"


def make_metric(
    *,
    eval_fn,
    greater_is_better,
    name=None,
    long_name=None,
    version=None,
):
    '''
    A factory function to create an :py:class:`EvaluationMetric` object.

    :param eval_fn:
        A function that computes the metric with the following signature:

        .. code-block:: python

            def eval_fn(
                predictions: pandas.Series,
                targets: pandas.Series,
                metrics: Dict[str, MetricValue],
                **kwargs,
            ) -> Union[float, MetricValue]:
                """
                :param predictions: A pandas Series containing the predictions made by the model.
                :param targets: (Optional) A pandas Series containing the corresponding labels
                    for the predictions made on that input.
                :param metrics: (Optional) A dictionary containing the metrics calculated by the
                    default evaluator.
                    The keys are the names of the metrics and the values are the metric values.
                    To access the MetricValue for the metrics calculated by the system, make sure
                    to specify the type hint for this parameter as Dict[str, MetricValue].
                    Refer to the DefaultEvaluator behavior section for what metrics
                    will be returned based on the type of model (i.e. classifier or regressor).
                :param kwargs: Includes a list of args that are used to compute the metric. These
                    args could information coming from input data, model outputs or parameters
                    specified in the `evaluator_config` argument of the `mlflow.evaluate` API.
                :return:
                    MetricValue with per-row scores, per-row justifications, and aggregate results.
                """
                ...

    :param greater_is_better: Whether a higher value of the metric is better.
    :param name: The name of the metric. This argument must be specified if ``eval_fn`` is a lambda
                 function or the ``eval_fn.__name__`` attribute is not available.
    :param long_name: (Optional) The long name of the metric. For example, ``"mean_squared_error"``
        for ``"mse"``.
    :param version: (Optional) The metric version. For example ``v1``.

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

    return EvaluationMetric(eval_fn, name, greater_is_better, long_name, version)


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
        pass

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
        pass

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


_cached_mlflow_client = None


def _hash_uint64_ndarray_as_bytes(array):
    assert len(array.shape) == 1
    # see struct pack format string https://docs.python.org/3/library/struct.html#format-strings
    return struct.pack(f">{array.size}Q", *array)


def _hash_ndarray_as_bytes(nd_array):
    return _hash_uint64_ndarray_as_bytes(
        pd.util.hash_array(nd_array.flatten(order="C"))
    ) + _hash_uint64_ndarray_as_bytes(np.array(nd_array.shape, dtype="uint64"))


def _hash_array_like_obj_as_bytes(data):
    """
    Helper method to convert pandas dataframe/numpy array/list into bytes for
    MD5 calculation purpose.
    """
    if isinstance(data, pd.DataFrame):
        # add checking `'pyspark' in sys.modules` to avoid importing pyspark when user
        # run code not related to pyspark.
        if "pyspark" in sys.modules:
            from pyspark.ml.linalg import Vector as spark_vector_type
        else:
            spark_vector_type = None

        def _hash_array_like_element_as_bytes(v):
            if spark_vector_type is not None:
                if isinstance(v, spark_vector_type):
                    return _hash_ndarray_as_bytes(v.toArray())
            if isinstance(v, np.ndarray):
                return _hash_ndarray_as_bytes(v)
            if isinstance(v, list):
                return _hash_ndarray_as_bytes(np.array(v))
            return v

        data = data.applymap(_hash_array_like_element_as_bytes)
        return _hash_uint64_ndarray_as_bytes(pd.util.hash_pandas_object(data))
    elif isinstance(data, np.ndarray):
        return _hash_ndarray_as_bytes(data)
    elif isinstance(data, list):
        return _hash_ndarray_as_bytes(np.array(data))
    else:
        raise ValueError("Unsupported data type.")


def _gen_md5_for_arraylike_obj(md5_gen, data):
    """
    Helper method to generate MD5 hash array-like object, the MD5 will calculate over:
     - array length
     - first NUM_SAMPLE_ROWS_FOR_HASH rows content
     - last NUM_SAMPLE_ROWS_FOR_HASH rows content
    """
    len_bytes = _hash_uint64_ndarray_as_bytes(np.array([len(data)], dtype="uint64"))
    md5_gen.update(len_bytes)
    if len(data) < EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH * 2:
        md5_gen.update(_hash_array_like_obj_as_bytes(data))
    else:
        head_rows = data[: EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH]
        tail_rows = data[-EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH :]
        md5_gen.update(_hash_array_like_obj_as_bytes(head_rows))
        md5_gen.update(_hash_array_like_obj_as_bytes(tail_rows))


class EvaluationDataset:
    """
    An input dataset for model evaluation. This is intended for use with the
    :py:func:`mlflow.models.evaluate()`
    API.
    """

    NUM_SAMPLE_ROWS_FOR_HASH = 5
    SPARK_DATAFRAME_LIMIT = 10000

    def __init__(
        self, data, *, targets=None, name=None, path=None, feature_names=None, predictions=None
    ):
        """
        The values of the constructor arguments comes from the `evaluate` call.
        """
        if name is not None and '"' in name:
            raise MlflowException(
                message=f'Dataset name cannot include a double quote (") but got {name}',
                error_code=INVALID_PARAMETER_VALUE,
            )
        if path is not None and '"' in path:
            raise MlflowException(
                message=f'Dataset path cannot include a double quote (") but got {path}',
                error_code=INVALID_PARAMETER_VALUE,
            )

        self._user_specified_name = name
        self._path = path
        self._hash = None
        self._supported_dataframe_types = (pd.DataFrame,)
        self._spark_df_type = None
        self._labels_data = None
        self._targets_name = None
        self._has_targets = False
        self._predictions_data = None
        self._predictions_name = None
        self._has_predictions = False

        try:
            # add checking `'pyspark' in sys.modules` to avoid importing pyspark when user
            # run code not related to pyspark.
            if "pyspark" in sys.modules:
                from pyspark.sql import DataFrame as SparkDataFrame

                self._supported_dataframe_types = (pd.DataFrame, SparkDataFrame)
                self._spark_df_type = SparkDataFrame
        except ImportError:
            pass

        if feature_names is not None and len(set(feature_names)) < len(list(feature_names)):
            raise MlflowException(
                message="`feature_names` argument must be a list containing unique feature names.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        has_targets = targets is not None
        if has_targets:
            self._has_targets = True
        if isinstance(data, (np.ndarray, list)):
            if has_targets and not isinstance(targets, (np.ndarray, list)):
                raise MlflowException(
                    message="If data is a numpy array or list of evaluation features, "
                    "`targets` argument must be a numpy array or list of evaluation labels.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            shape_message = (
                "If the `data` argument is a numpy array, it must be a 2-dimensional "
                "array, with the second dimension representing the number of features. If the "
                "`data` argument is a list, each of its elements must be a feature array of "
                "the numpy array or list, and all elements must have the same length."
            )

            if isinstance(data, list):
                try:
                    data = np.array(data)
                except ValueError as e:
                    raise MlflowException(
                        message=shape_message, error_code=INVALID_PARAMETER_VALUE
                    ) from e

            if len(data.shape) != 2:
                raise MlflowException(
                    message=shape_message,
                    error_code=INVALID_PARAMETER_VALUE,
                )

            self._features_data = data
            if has_targets:
                self._labels_data = (
                    targets if isinstance(targets, np.ndarray) else np.array(targets)
                )

                if len(self._features_data) != len(self._labels_data):
                    raise MlflowException(
                        message="The input features example rows must be the same length "
                        "with labels array.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )

            num_features = data.shape[1]

            if feature_names is not None:
                feature_names = list(feature_names)
                if num_features != len(feature_names):
                    raise MlflowException(
                        message="feature name list must be the same length with feature data.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                self._feature_names = feature_names
            else:
                self._feature_names = [
                    f"feature_{str(i + 1).zfill(math.ceil(math.log10(num_features + 1)))}"
                    for i in range(num_features)
                ]
        elif isinstance(data, self._supported_dataframe_types):
            if has_targets and not isinstance(targets, str):
                raise MlflowException(
                    message="If data is a Pandas DataFrame or Spark DataFrame, `targets` argument "
                    "must be the name of the column which contains evaluation labels in the `data` "
                    "dataframe.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if self._spark_df_type and isinstance(data, self._spark_df_type):
                if data.count() > EvaluationDataset.SPARK_DATAFRAME_LIMIT:
                    _logger.warning(
                        "Specified Spark DataFrame is too large for model evaluation. Only "
                        f"the first {EvaluationDataset.SPARK_DATAFRAME_LIMIT} rows will be used. "
                        "If you want evaluate on the whole spark dataframe, please manually call "
                        "`spark_dataframe.toPandas()`."
                    )
                data = data.limit(EvaluationDataset.SPARK_DATAFRAME_LIMIT).toPandas()

            if has_targets:
                self._labels_data = data[targets].to_numpy()
                self._targets_name = targets

            self._has_predictions = predictions is not None
            if self._has_predictions:
                self._predictions_data = data[predictions].to_numpy()
                self._predictions_name = predictions

            if feature_names is not None:
                self._features_data = data[list(feature_names)]
                self._feature_names = feature_names
            else:
                features_data = data

                if has_targets:
                    features_data = features_data.drop(targets, axis=1, inplace=False)

                if self._has_predictions:
                    features_data = features_data.drop(predictions, axis=1, inplace=False)

                self._features_data = features_data
                self._feature_names = [
                    generate_feature_name_if_not_string(c) for c in self._features_data.columns
                ]
        else:
            raise MlflowException(
                message="The data argument must be a numpy array, a list or a Pandas DataFrame, or "
                "spark DataFrame if pyspark package installed.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # generate dataset hash
        md5_gen = _insecure_md5()
        _gen_md5_for_arraylike_obj(md5_gen, self._features_data)
        if self._labels_data is not None:
            _gen_md5_for_arraylike_obj(md5_gen, self._labels_data)
        if self._predictions_data is not None:
            _gen_md5_for_arraylike_obj(md5_gen, self._predictions_data)
        md5_gen.update(",".join(list(map(str, self._feature_names))).encode("UTF-8"))

        self._hash = md5_gen.hexdigest()

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def features_data(self):
        """
        return features data as a numpy array or a pandas DataFrame.
        """
        return self._features_data

    @property
    def labels_data(self):
        """
        return labels data as a numpy array
        """
        return self._labels_data

    @property
    def has_targets(self):
        """
        Returns True if the dataset has targets, False otherwise.
        """
        return self._has_targets

    @property
    def targets_name(self):
        """
        return targets name
        """
        return self._targets_name

    @property
    def predictions_data(self):
        """
        return labels data as a numpy array
        """
        return self._predictions_data

    @property
    def has_predictions(self):
        """
        Returns True if the dataset has targets, False otherwise.
        """
        return self._has_predictions

    @property
    def predictions_name(self):
        """
        return targets name
        """
        return self._predictions_name

    @property
    def name(self):
        """
        Dataset name, which is specified dataset name or the dataset hash if user don't specify
        name.
        """
        return self._user_specified_name if self._user_specified_name is not None else self.hash

    @property
    def path(self):
        """
        Dataset path
        """
        return self._path

    @property
    def hash(self):
        """
        Dataset hash, includes hash on first 20 rows and last 20 rows.
        """
        return self._hash

    @property
    def _metadata(self):
        """
        Return dataset metadata containing name, hash, and optional path.
        """
        metadata = {
            "name": self.name,
            "hash": self.hash,
        }
        if self.path is not None:
            metadata["path"] = self.path
        return metadata

    def _log_dataset_tag(self, client, run_id, model_uuid):
        """
        Log dataset metadata as a tag "mlflow.datasets", if the tag already exists, it will
        append current dataset metadata into existing tag content.
        """
        existing_dataset_metadata_str = client.get_run(run_id).data.tags.get(
            "mlflow.datasets", "[]"
        )
        dataset_metadata_list = json.loads(existing_dataset_metadata_str)

        for metadata in dataset_metadata_list:
            if (
                metadata["hash"] == self.hash
                and metadata["name"] == self.name
                and metadata["model"] == model_uuid
            ):
                break
        else:
            dataset_metadata_list.append({**self._metadata, "model": model_uuid})

        dataset_metadata_str = json.dumps(dataset_metadata_list, separators=(",", ":"))
        client.log_batch(
            run_id,
            tags=[RunTag("mlflow.datasets", dataset_metadata_str)],
        )

    def __hash__(self):
        return hash(self.hash)

    def __eq__(self, other):
        if not isinstance(other, EvaluationDataset):
            return False

        if isinstance(self._features_data, np.ndarray):
            is_features_data_equal = np.array_equal(self._features_data, other._features_data)
        else:
            is_features_data_equal = self._features_data.equals(other._features_data)

        return (
            is_features_data_equal
            and np.array_equal(self._labels_data, other._labels_data)
            and self.name == other.name
            and self.path == other.path
            and self._feature_names == other._feature_names
        )


@developer_stable
class ModelEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def can_evaluate(self, *, model_type, evaluator_config, **kwargs) -> bool:
        """
        :param model_type: A string describing the model type (e.g., "regressor", "classifier", …).
        :param evaluator_config: A dictionary of additional configurations for
                                 the evaluator.
        :param kwargs: For forwards compatibility, a placeholder for additional arguments
                       that may be added to the evaluation interface in the future.
        :return: True if the evaluator can evaluate the specified model on the
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
        **kwargs,
    ):
        """
        The abstract API to log metrics and artifacts, and return evaluation results.

        :param model_type: A string describing the model type
                           (e.g., ``"regressor"``, ``"classifier"``, …).
        :param dataset: An instance of `mlflow.models.evaluation.base._EvaluationDataset`
                        containing features and labels (optional) for model evaluation.
        :param run_id: The ID of the MLflow Run to which to log results.
        :param evaluator_config: A dictionary of additional configurations for
                                 the evaluator.
        :param model: A pyfunc model instance, used as the candidate_model
                      to be compared with baseline_model (specified by the `baseline_model` param)
                      for model validation. If None, the model output is supposed to be found in
                      ``dataset.predictions_data``.
        :param extra_metrics: A list of :py:class:`EvaluationMetric` objects.
        :param custom_artifacts: A list of callable custom artifact functions.
        :param kwargs: For forwards compatibility, a placeholder for additional arguments that
                       may be added to the evaluation interface in the future.
        :param baseline_model: (Optional) A string URI referring to a MLflow model with the pyfunc
                                          flavor as a baseline model to be compared with the
                                          candidate model (specified by the `model` param) for model
                                          validation. (pyfunc model instance is not allowed)
        :return: A :py:class:`mlflow.models.EvaluationResult` instance containing
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
            _logger.warning(
                f"Multiple registered evaluators are found {evaluator_name_list} and "
                "they will all be used in evaluation if they support the specified model type. "
                "If you want to evaluate with one evaluator, specify the `evaluator` argument "
                "and optionally specify the `evaluator_config` argument."
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
    :param validation_thresholds: A dictionary from metric_name to MetricThreshold.
    :param candidate_metrics: The metric evaluation result of the candidate model.
    :param baseline_metrics: The metric evaluation result of the baseline model.
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


def _convert_data_to_mlflow_dataset(data, targets=None, predictions=None):
    """Convert input data to mlflow dataset."""
    if "pyspark" in sys.modules:
        from pyspark.sql import DataFrame as SparkDataFrame

    if isinstance(data, list):
        return mlflow.data.from_numpy(np.array(data), targets=np.array(targets))
    elif isinstance(data, np.ndarray):
        return mlflow.data.from_numpy(data, targets=targets)
    elif isinstance(data, pd.DataFrame):
        return mlflow.data.from_pandas(df=data, targets=targets, predictions=predictions)
    elif "pyspark" in sys.modules and isinstance(data, SparkDataFrame):
        return mlflow.data.from_spark(df=data, targets=targets)
    else:
        # Cannot convert to mlflow dataset, return original data.
        _logger.info(
            "Cannot convert input data to `evaluate()` to an mlflow dataset, input must be a list, "
            f"a numpy array, a panda Dataframe or a spark Dataframe, but received {type(data)}."
        )
        return data


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
            _logger.info(f"Evaluating the model with the {evaluator_name} evaluator.")
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


def evaluate(
    model: Optional[str] = None,
    data=None,
    *,
    model_type: Optional[str] = None,
    targets=None,
    predictions=None,
    dataset_path=None,
    feature_names: Optional[list] = None,
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
):
    '''
    Evaluate a PyFunc model on the specified dataset using one or more specified ``evaluators``, and
    log resulting metrics & artifacts to MLflow Tracking. Set thresholds on the generated metrics to
    validate model quality. For additional overview information, see
    :ref:`the Model Evaluation documentation <model-evaluation>`.

    Default Evaluator behavior:
     - The default evaluator, which can be invoked with ``evaluators="default"`` or
       ``evaluators=None``, supports the ``"regressor"`` and ``"classifier"`` model types.
       It generates a variety of model performance metrics, model performance plots, and
       model explanations.

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
        - **metrics**: ``exact_match``, ``token_count``, `mean_perplexity`_ (requires `evaluate`_,
          `pytorch`_, `transformers`_), `toxicity_ratio`_ (requires `evaluate`_, `pytorch`_,
          `mean_flesch_kincaid_grade_level`_ (requires `textstat`_).
        - **artifacts**: A JSON file containing the inputs, outputs, targets (if the ``targets``
          argument is supplied), and per-row metrics of the model in tabular format.

        .. _mean_perplexity:
            https://huggingface.co/spaces/evaluate-metric/perplexity

        .. _toxicity_ratio:
            https://huggingface.co/spaces/evaluate-measurement/toxicity

        .. _pytorch:
            https://pytorch.org/get-started/locally/

        .. _transformers:
            https://huggingface.co/docs/transformers/installation

        .. _mean_ari_grade_level:
            https://en.wikipedia.org/wiki/Automated_readability_index

        .. _mean_flesch_kincaid_grade_level:
            https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level

        .. _evaluate:
            https://pypi.org/project/evaluate

        .. _textstat:
            https://pypi.org/project/textstat

     - For text-summarization models, the default evaluator logs:
        - **metrics**: ``token_count``, `ROUGE`_ (requires `evaluate`_, `nltk`_, and
          `rouge_score`_ to be installed), `mean_perplexity`_ (requires `evaluate`_, `pytorch`_,
          `transformers`_), `toxicity_ratio`_ (requires `evaluate`_, `pytorch`_, `transformers`_),
          `mean_ari_grade_level`_ (requires `textstat`_), `mean_flesch_kincaid_grade_level`_
          (requires `textstat`_).
        - **artifacts**: A JSON file containing the inputs, outputs, targets (if the ``targets``
          argument is supplied), and per-row metrics of the model in the tabular format.

        .. _ROUGE:
            https://huggingface.co/spaces/evaluate-metric/rouge

        .. _mean_perplexity:
            https://huggingface.co/spaces/evaluate-metric/perplexity

        .. _toxicity_ratio:
            https://huggingface.co/spaces/evaluate-measurement/toxicity

        .. _pytorch:
            https://pytorch.org/get-started/locally/

        .. _transformers:
            https://huggingface.co/docs/transformers/installation

        .. _mean_ari_grade_level:
            https://en.wikipedia.org/wiki/Automated_readability_index

        .. _mean_flesch_kincaid_grade_level:
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
        - **metrics**: ``token_count``, `mean_perplexity`_ (requires `evaluate`_, `pytorch`_,
          `transformers`_), `toxicity_ratio`_ (requires `evaluate`_, `pytorch`_, `transformers`_),
          `mean_ari_grade_level`_ (requires `textstat`_), `mean_flesch_kincaid_grade_level`_
          (requires `textstat`_).
        - **artifacts**: A JSON file containing the inputs, outputs, targets (if the ``targets``
          argument is supplied), and per-row metrics of the model in tabular format.

        .. _evaluate:
            https://pypi.org/project/evaluate

        .. _mean_perplexity:
            https://huggingface.co/spaces/evaluate-metric/perplexity

        .. _toxicity_ratio:
            https://huggingface.co/spaces/evaluate-measurement/toxicity

        .. _pytorch:
            https://pytorch.org/get-started/locally/

        .. _transformers:
            https://huggingface.co/docs/transformers/installation

        .. _mean_ari_grade_level:
            https://en.wikipedia.org/wiki/Automated_readability_index

        .. _mean_flesch_kincaid_grade_level:
            https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level

        .. _textstat:
            https://pypi.org/project/textstat

    - For retriever models, the default evaluator logs:
        - **metrics**: ``precision_at_k``: precision at k, where k is specified by the
            ``evaluator_config`` argument. The default value of k is 3.
        - **artifacts**: A JSON file containing the inputs, outputs, targets (if the ``targets``
          argument is supplied), and per-row metrics of the model in tabular format.

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
        - **k**: The number of top predictions to use when computing the built-in metric
          precision_at_k.

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

    :param model: Optional. If specified, it should be one of the following:

                  - A pyfunc model instance

                  - A URI referring to a pyfunc model

                  - A callable function: This function should be able to take in model input and
                    return predictions. It should follow the signature of the
                    :py:func:`predict <mlflow.pyfunc.PyFuncModel.predict>` method. Here's an example
                    of a valid function:

                        model = mlflow.pyfunc.load_model(model_uri)

                        def fn(model_input):
                            return model.predict(model_input)

        If omitted, it indicates a static dataset will be used for evaluation instead of a model.
        In this case, the ``data`` argument must be a Pandas DataFrame or an mlflow PandasDataset
        that contains model outputs, and the ``predictions`` argument must be the name of the
        column in ``data`` that contains model outputs.

    :param data: One of the following:

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
                   should be specified via the ``predictions`` property of the PandasDataset.

    :param targets: If ``data`` is a numpy array or list, a numpy array or list of evaluation
                    labels. If ``data`` is a DataFrame, the string name of a column from ``data``
                    that contains evaluation labels. Required for classifier and regressor models,
                    but optional for question-answering, text-summarization, and text models. If
                    ``data`` is a :py:class:`mlflow.data.dataset.Dataset` that defines targets,
                    then ``targets`` is optional.

    :param predictions: Optional. Only used when ``model`` is not specified and ``data`` is a pandas
                        dataframe. The name of the column in ``data`` that contains model outputs.

    :param model_type: (Optional) A string describing the model type. The default evaluator
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

    :param dataset_path: (Optional) The path where the data is stored. Must not contain double
                         quotes (``“``). If specified, the path is logged to the ``mlflow.datasets``
                         tag for lineage tracking purposes.

    :param feature_names: (Optional) If the ``data`` argument is a feature data numpy array or list,
                          ``feature_names`` is a list of the feature names for each feature. If
                          ``None``, then the ``feature_names`` are generated using the format
                          ``feature_{feature_index}``. If the ``data`` argument is a Pandas
                          DataFrame or a Spark DataFrame, ``feature_names`` is a list of the names
                          of the feature columns in the DataFrame. If ``None``, then all columns
                          except the label column and the predictions column are regarded as
                          feature columns.

    :param evaluators: The name of the evaluator to use for model evaluation, or a list of
                       evaluator names. If unspecified, all evaluators capable of evaluating the
                       specified model on the specified dataset are used. The default evaluator
                       can be referred to by the name ``"default"``. To see all available
                       evaluators, call :py:func:`mlflow.models.list_evaluators`.

    :param evaluator_config: A dictionary of additional configurations to supply to the evaluator.
                             If multiple evaluators are specified, each configuration should be
                             supplied as a nested dictionary whose key is the evaluator name.

    :param extra_metrics:
        (Optional) A list of :py:class:`EvaluationMetric <mlflow.models.EvaluationMetric>` objects.
        See the `mlflow.metrics` module for more information about the
        builtin metrics and how to define custom metrics

        .. code-block:: python
            :caption: Example usage of custom metrics

            import mlflow
            import numpy as np


            def root_mean_squared_error(eval_df, _builtin_metrics):
                return np.sqrt((np.abs(eval_df["prediction"] - eval_df["target"]) ** 2).mean)


            rmse_metric = mlflow.models.make_metric(
                eval_fn=root_mean_squared_error,
                greater_is_better=False,
            )
            mlflow.evaluate(..., extra_metrics=[rmse_metric])

    :param custom_artifacts:
        (Optional) A list of custom artifact functions with the following signature:

        .. code-block:: python

            def custom_artifact(
                eval_df: Union[pandas.Dataframe, pyspark.sql.DataFrame],
                builtin_metrics: Dict[str, float],
                artifacts_dir: str,
            ) -> Dict[str, Any]:
                """
                :param eval_df:
                    A Pandas or Spark DataFrame containing ``prediction`` and ``target`` column.
                    The ``prediction`` column contains the predictions made by the model.
                    The ``target`` column contains the corresponding labels to the predictions made
                    on that row.
                :param builtin_metrics:
                    A dictionary containing the metrics calculated by the default evaluator.
                    The keys are the names of the metrics and the values are the scalar values of
                    the metrics. Refer to the DefaultEvaluator behavior section for what metrics
                    will be returned based on the type of model (i.e. classifier or regressor).
                :param artifacts_dir:
                    A temporary directory path that can be used by the custom artifacts function to
                    temporarily store produced artifacts. The directory will be deleted after the
                    artifacts are logged.
                :return:
                    A dictionary that maps artifact names to artifact objects
                    (e.g. a Matplotlib Figure) or to artifact paths within ``artifacts_dir``.
                """
                ...

        Object types that artifacts can be represented as:

            - A string uri representing the file path to the artifact. MLflow will infer the type of
              the artifact based on the file extension.
            - A string representation of a JSON object. This will be saved as a .json artifact.
            - Pandas DataFrame. This will be resolved as a CSV artifact.
            - Numpy array. This will be saved as a .npy artifact.
            - Matplotlib Figure. This will be saved as an image artifact. Note that
              ``matplotlib.pyplot.savefig`` is called behind the scene with default configurations.
              To customize, either save the figure with the desired configurations and return its
              file path or define customizations through environment variables in
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

    :param validation_thresholds: (Optional) A dictionary of metric name to
        :py:class:`mlflow.models.MetricThreshold` used for model validation. Each metric name must
        either be the name of a builtin metric or the name of a metric defined in the
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

    :param baseline_model: (Optional) A string URI referring to an MLflow model with the pyfunc
                           flavor. If specified, the candidate ``model`` is compared to this
                           baseline for model validation purposes.

    :param env_manager: Specify an environment manager to load the candidate ``model`` and
                        ``baseline_model`` in isolated Python evironments and restore their
                        dependencies. Default value is ``local``, and the following values are
                        supported:

                         - ``virtualenv``: (Recommended) Use virtualenv to restore the python
                           environment that was used to train the model.
                         - ``conda``:  Use Conda to restore the software environment that was used
                           to train the model.
                         - ``local``: Use the current Python environment for model inference, which
                           may differ from the environment used to train the model and may lead to
                           errors or invalid predictions.

    :param model_config: the model configuration to use for loading the model with pyfunc. Inspect
                         the model's pyfunc flavor to know which keys are supported for your
                         specific model. If not indicated, the default model configuration
                         from the model is used (if any).
    :param baseline_config: the model configuration to use for loading the baseline
                            model. If not indicated, the default model configuration
                            from the baseline model is used (if any).

    :return: An :py:class:`mlflow.models.EvaluationResult` instance containing
             metrics of candidate model and baseline model, and artifacts of candidate model.
    '''
    from mlflow.pyfunc import PyFuncModel, _load_model_or_server, _ServedPyFuncModel
    from mlflow.utils import env_manager as _EnvManager

    if data is None:
        raise MlflowException(
            message="The data argument cannot be None.", error_code=INVALID_PARAMETER_VALUE
        )

    if predictions is not None and model is not None:
        raise MlflowException(
            message="The predictions argument cannot be specified when model is specified.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    _EnvManager.validate(env_manager)

    # If Dataset is provided, the targets and predictions can only be specified by the Dataset,
    # not the targets and predictions parameters of the mlflow.evaluate() API.
    if isinstance(data, Dataset) and targets is not None:
        raise MlflowException(
            message="The top-level targets parameter should not be specified since a Dataset "
            "is used. Please only specify the targets column name in the Dataset. For example: "
            "`data = mlflow.data.from_pandas(df=X.assign(y=y), targets='y')`. "
            "Meanwhile, please specify `mlflow.evaluate(..., targets=None, ...)`.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if isinstance(data, Dataset) and predictions is not None:
        raise MlflowException(
            message="The top-level predictions parameter should not be specified since a Dataset "
            "is used. Please only specify the predictions column name in the Dataset. For example:"
            " `data = mlflow.data.from_pandas(df=X.assign(y=y), predictions='y')`"
            "Meanwhile, please specify `mlflow.evaluate(..., predictions=None, ...)`.",
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
        pass
    elif model is None:
        # Evaluating a static dataset
        if isinstance(data, pd.DataFrame):
            # If data is a pandas dataframe, predictions must be specified
            if predictions is None:
                raise MlflowException(
                    message="The model output must be specified in the predicitons "
                    "parameter when model=None.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        elif isinstance(data, mlflow.data.pandas_dataset.PandasDataset):
            # If data is a mlflow PandasDataset, data.predictions must be specified
            if data.predictions is None:
                raise MlflowException(
                    message="The predictions parameter must be specified with the provided "
                    "PandasDataset when model=None. For example: "
                    "`data = mlflow.data.from_pandas(df=X.assign(y=y), predictions='y')`",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            # Other data formats are not supported
            raise MlflowException(
                message="The data must be a pandas dataframe or mlflow.data.pandas_dataset."
                "PandasDataset when model=None.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    elif callable(model):
        model = _get_model_from_function(model)
    else:
        raise MlflowException(
            message="The model argument must be a string URI referring to an MLflow model, "
            "an instance of `mlflow.pyfunc.PyFuncModel`, a function, or None.",
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
            data = _convert_data_to_mlflow_dataset(
                data=data, targets=targets, predictions=predictions
            )

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
            )

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
