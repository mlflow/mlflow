from typing import Dict, Union
import mlflow
import hashlib
import json
import os
from contextlib import contextmanager
from mlflow.exceptions import MlflowException
from mlflow.utils.file_utils import TempDir
from mlflow.entities import RunTag
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.annotations import experimental
import logging
import struct
import sys
import math
from collections import OrderedDict
from abc import ABCMeta, abstractmethod


_logger = logging.getLogger(__name__)


class EvaluationMetrics(dict):
    """
    A dictionary of model evaluation metrics.
    """

    pass


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

    def load(self, local_artifact_path=None):
        """
        If `local_artifact_path` is None, download artifact from the artifact uri,
        otherwise load artifact content from specified path.
        then assign the loaded content to `self._content`, and return the loaded content.
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
    def save(self, output_artifact_path):
        """Save artifact content into specified path."""
        pass

    @property
    def content(self):
        """
        The content of the artifact (representation varies)
        """
        if self._content is None:
            self.load()
        return self._content

    @property
    def uri(self) -> str:
        """
        The URI of the artifact
        """
        return self._uri


class EvaluationResult:
    """
    Represent an return value of `mlflow.evaluate()` API. Contains metrics dict and
    artifact dict.
    """

    def __init__(self, metrics, artifacts):
        self._metrics = metrics
        self._artifacts = artifacts

    @classmethod
    def load(cls, path):
        """Load the evaluation results from the specified local filesystem path"""
        with open(os.path.join(path, "metrics.json"), "r") as fp:
            metrics = EvaluationMetrics(json.load(fp))

        with open(os.path.join(path, "artifacts_metadata.json"), "r") as fp:
            artifacts_metadata = json.load(fp)

        artifacts = {}

        artifacts_dir = os.path.join(path, "artifacts")

        for artifact_name, meta in artifacts_metadata.items():
            uri = meta["uri"]
            ArtifactCls = _get_class_from_string(meta["class_name"])
            artifact = ArtifactCls(uri=uri)
            artifact.load(os.path.join(artifacts_dir, artifact_name))
            artifacts[artifact_name] = artifact

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    def save(self, path):
        """Write the evaluation results to the specified local filesystem path"""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metrics.json"), "w") as fp:
            json.dump(self.metrics, fp)

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
        os.mkdir(artifacts_dir)

        for artifact_name, artifact in self.artifacts.items():
            artifact.save(os.path.join(artifacts_dir, artifact_name))

    @property
    def metrics(self) -> "mlflow.models.evaluation.EvaluationMetrics":
        """
        A dictionary mapping scalar metric names to scalar metric values
        """
        return self._metrics

    @property
    def artifacts(self) -> Dict[str, "mlflow.models.evaluation.EvaluationArtifact"]:
        """
        A dictionary mapping standardized artifact names (e.g. "roc_data") to
        artifact content and location information
        """
        return self._artifacts


_cached_mlflow_client = None


def _hash_uint64_ndarray_as_bytes(array):
    assert len(array.shape) == 1
    # see struct pack format string https://docs.python.org/3/library/struct.html#format-strings
    return struct.pack(f">{array.size}Q", *array)


def _hash_ndarray_as_bytes(nd_array):
    from pandas.util import hash_array
    import numpy as np

    return _hash_uint64_ndarray_as_bytes(
        hash_array(nd_array.flatten(order="C"))
    ) + _hash_uint64_ndarray_as_bytes(np.array(nd_array.shape, dtype="uint64"))


def _hash_array_like_obj_as_bytes(data):
    """
    Helper method to convert pandas dataframe/numpy array/list into bytes for
    MD5 calculation purpose.
    """
    from pandas.util import hash_pandas_object
    import numpy as np
    import pandas as pd

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
        return _hash_uint64_ndarray_as_bytes(hash_pandas_object(data))
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
    import numpy as np

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
    :py:func:`mlflow.models.evaluation.evaluate()`
    API.
    """

    NUM_SAMPLE_ROWS_FOR_HASH = 5
    SPARK_DATAFRAME_LIMIT = 10000

    def __init__(self, data, labels, name=None, path=None, feature_names=None):
        """
        :param data: One of the following:
         - A numpy array or list of evaluation features, excluding labels.
         - A Pandas DataFrame, or a spark DataFrame,
           containing evaluation features and labels. All columns will be regarded as feature
           columns except the "labels" column.
         Note: If the mlflow model to be evaluated is a pyspark ML model, then the input data must
           be a spark DataFrame contains a feature column of "Vector" type, and a label column.

        :param labels: One of the following:
         - A numpy array or list of evaluation labels, if `data` is also a numpy array or list.
         - The string name of a column from `data` that contains evaluation labels, if `data`
           is a DataFrame.

        :param name: (Optional) The name of the dataset (must not contain ").

        :param path: (Optional) the path to a serialized DataFrame (must not contain ").
          (e.g. a delta table, parquet file)

        :param feature_names: (Optional) A list of the feature names attached to the numpy array
          input data. The argument is only useful in the case the input data is numpy array.
          For pandas DataFrame input case, the pandas column name will be used as feature name.
          The feature names will be shown in model explainability plots.
        """
        import numpy as np
        import pandas as pd

        if name is not None and '"' in name:
            raise ValueError(f'Dataset name cannot include a double quote (") but got {name}')
        if path is not None and '"' in path:
            raise ValueError(f'Dataset path cannot include a double quote (") but got {path}')

        self._user_specified_name = name
        self._path = path
        self._hash = None

        try:
            # add checking `'pyspark' in sys.modules` to avoid importing pyspark when user
            # run code not related to pyspark.
            if "pyspark" in sys.modules:
                from pyspark.sql import DataFrame as SparkDataFrame

                supported_dataframe_types = (pd.DataFrame, SparkDataFrame)
                spark_df_type = SparkDataFrame
            else:
                supported_dataframe_types = (pd.DataFrame,)
                spark_df_type = None
        except ImportError:
            supported_dataframe_types = (pd.DataFrame,)

        if isinstance(data, (np.ndarray, list)):
            if not isinstance(labels, (np.ndarray, list)):
                raise ValueError(
                    "If data is a numpy array or list of evaluation features, "
                    "labels must be a numpy array or list of evaluation labels"
                )
            if isinstance(data, list):
                data = np.array(data)

            if len(data.shape) != 2:
                raise ValueError(
                    "If the `data` argument is a numpy array, it must be a 2 dimension array "
                    "and second dimension represent the number of features. If the `data` "
                    "argument is a list, each of its element must be a feature array of "
                    "numpy array or list and all element must has the same length."
                )

            self._features_data = data
            self._labels_data = labels if isinstance(labels, np.ndarray) else np.array(labels)

            if len(self._features_data) != len(self._labels_data):
                raise ValueError(
                    "The input features example rows must be the same length with labels array."
                )

            num_features = data.shape[1]

            if feature_names is not None:
                feature_names = list(feature_names)
                if num_features != len(feature_names):
                    raise ValueError("feature name list must be the same length with feature data.")
                self._feature_names = feature_names
            else:
                self._feature_names = [
                    f"feature_{str(i + 1).zfill(math.ceil((math.log10(num_features + 1))))}"
                    for i in range(num_features)
                ]
        elif isinstance(data, supported_dataframe_types):
            if not isinstance(labels, str):
                raise ValueError(
                    "If data is a Pandas DataFrame or Spark DataFrame, labels must be the "
                    "string name of a column from `data` that contains evaluation labels"
                )
            if isinstance(data, spark_df_type):
                _logger.warning(
                    f"Specified Spark DataFrame is too large for model evaluation. Only "
                    f"the first {EvaluationDataset.SPARK_DATAFRAME_LIMIT} rows will be used."
                    "If you want evaluate on the whole spark dataframe, please manually call "
                    "`spark_dataframe.toPandas()`."
                )
                data = data.limit(EvaluationDataset.SPARK_DATAFRAME_LIMIT).toPandas()

            self._features_data = data.drop(labels, axis=1, inplace=False)
            self._labels_data = data[labels].to_numpy()

            if feature_names is not None:
                raise ValueError(
                    "If `data` argument is pandas/spark dataframe, you cannot specify the "
                    "`feature_names` argument, instead, the column names of the input "
                    "dataframe will be used as the feature names."
                )
            self._feature_names = list(self._features_data.columns)
        else:
            raise ValueError(
                "The data argument must be a numpy array, a list or a Pandas DataFrame, or "
                "spark DataFrame if pyspark package installed."
            )

        # generate dataset hash
        md5_gen = hashlib.md5()
        _gen_md5_for_arraylike_obj(md5_gen, self._features_data)
        _gen_md5_for_arraylike_obj(md5_gen, self._labels_data)
        md5_gen.update(",".join(self._feature_names).encode("UTF-8"))

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


class ModelEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def can_evaluate(self, *, model_type, evaluator_config, **kwargs) -> bool:
        """
        :param model_type: A string describing the model type (e.g., "regressor",
                           "classifier", …).
        :param evaluator_config: A dictionary of additional configurations for
                                 the evaluator.
        :param kwargs: For forwards compatibility, a placeholder for additional
                         arguments that may be added to the evaluation interface
                         in the future.
        :return: True if the evaluator can evaluate the specified model on the
                 specified dataset. False otherwise.
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, *, model, model_type, dataset, run_id, evaluator_config, **kwargs):
        """
        The abstract API to log metrics and artifacts, and return evaluation results.

        :param model: A pyfunc model instance.
        :param model_type: A string describing the model type (e.g., "regressor",
                   "classifier", …).
        :param dataset: An instance of :py:class:`mlflow.models.evaluation.EvaluationDataset`
                        containing features and labels (optional) for model evaluation.
        :param run_id: The ID of the MLflow Run to which to log results.
        :param evaluator_config: A dictionary of additional configurations for
                                 the evaluator.
        :param kwargs: For forwards compatibility, a placeholder for additional
                         arguments that may be added to the evaluation interface
                         in the future.
        :return: An :py:class:`mlflow.models.evaluation.EvaluationResult` instance containing
                 evaluation results.
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
def _start_run_or_reuse_active_run(run_id):
    """
    A manager context return:
     - If there's an active run, return the active run id.
     - otherwise start a mflow run with the specified run_id,
       if specified run_id is None, start a new run.
    """
    active_run = mlflow.active_run()
    if not active_run:
        # Note `mlflow.start_run` throws if `run_id` is not found.
        with mlflow.start_run(run_id=run_id) as run:
            yield run.info.run_id
    else:
        if run_id and active_run.info.run_id != run_id:
            raise ValueError(
                "An active run exists, you cannot specify another run_id when evaluating."
            )
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
            conf_dict_value_error = ValueError(
                "If `evaluators` argument is None, all available evaluators will be used. "
                "If only the default evaluator is available, the `evaluator_config` argument is "
                "interpreted as the config dictionary for the default evaluator. Otherwise, the "
                "`evaluator_config` argument must be a dictionary mapping each evaluator's name "
                "to its own evaluator config dictionary."
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
            raise ValueError(
                "If `evaluators` argument is the name of an evaluator, evaluator_config must be "
                "None or a dict containing config items for the evaluator."
            )
        evaluator_name_list = [evaluators]
        evaluator_name_to_conf_map = {evaluators: evaluator_config}
    elif isinstance(evaluators, list):
        if evaluator_config is not None:
            if not check_nesting_config_dict(evaluators, evaluator_config):
                raise ValueError(
                    "If `evaluators` argument is an evaluator name list, evaluator_config "
                    "must be a dict contains mapping from evaluator name to individual "
                    "evaluator config dict."
                )
        # Use `OrderedDict.fromkeys` to deduplicate elements but keep elements order.
        evaluator_name_list = list(OrderedDict.fromkeys(evaluators))
        evaluator_name_to_conf_map = evaluator_config or {}
    else:
        raise ValueError(
            "`evaluators` argument must be None, an evaluator name string, or a list of "
            "evaluator names."
        )

    return evaluator_name_list, evaluator_name_to_conf_map


_last_failed_evaluator = None


def _get_last_failed_evaluator():
    """
    Return the evaluator name of the last failed evaluator when calling `evalaute`.
    This can be used to check which evaluator fail when `evaluate` API fail.
    """
    return _last_failed_evaluator


def _evaluate(
    *, model, model_type, dataset, actual_run_id, evaluator_name_list, evaluator_name_to_conf_map
):
    """
    The public API "evaluate" will verify argument first, and then pass normalized arguments
    to the _evaluate method.
    """
    # import _model_evaluation_registry and PyFuncModel inside function to avoid circuit importing
    from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry

    global _last_failed_evaluator
    _last_failed_evaluator = None

    client = mlflow.tracking.MlflowClient()
    model_uuid = model.metadata.model_uuid
    dataset._log_dataset_tag(client, actual_run_id, model_uuid)

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
            result = evaluator.evaluate(
                model=model,
                model_type=model_type,
                dataset=dataset,
                run_id=actual_run_id,
                evaluator_config=config,
            )
            eval_results.append(result)

    _last_failed_evaluator = None

    if len(eval_results) == 0:
        raise ValueError(
            "The model could not be evaluated by any of the registered evaluators, please "
            "verify that the model type and other configs are set correctly."
        )

    merged_eval_result = EvaluationResult(EvaluationMetrics(), dict())
    for eval_result in eval_results:
        merged_eval_result.metrics.update(eval_result.metrics)
        merged_eval_result.artifacts.update(eval_result.artifacts)

    return merged_eval_result


@experimental
def evaluate(
    *,
    model: Union[str, "mlflow.pyfunc.PyFuncModel"],
    model_type: str,
    dataset: "mlflow.models.evaluation.EvaluationDataset",
    run_id=None,
    evaluators=None,
    evaluator_config=None,
) -> "mlflow.models.evaluation.EvaluationResult":
    """
    Evaluate a PyFunc model on the specified dataset using one or more specified evaluators, and
    log resulting metrics & artifacts to MLflow Tracking.

    :param model: A pyfunc model instance, or a URI referring to such a model.

    :param model_type: A string describing the model type. The default evaluator
                       supports "regressor" and "classifier" as model types.
    :param dataset: An instance of :py:class:`mlflow.models.evaluation.EvaluationDataset`
                    containing features labels (optional) for model evaluation.
    :param run_id: The ID of the MLflow Run to which to log results. If
                   unspecified, behavior depends on the specified `evaluator`.
                   When `run_id` is unspecified, the default evaluator logs
                   results to the current active run, creating a new active run if
                   one does not exist.
    :param evaluators: The name of the evaluator to use for model evaluations, or
                       a list of evaluator names. If unspecified, all evaluators
                       capable of evaluating the specified model on the specified
                       dataset are used. The default evaluator can be referred to
                       by the name 'default'. If this argument is unspecified, then
                       fetch all evaluators from the registry.
    :param evaluator_config: A dictionary of additional configurations to supply
                             to the evaluator. If multiple evaluators are
                             specified, each configuration should be supplied as
                             a nested dictionary whose key is the evaluator name.
    :return: An :py:class:`mlflow.models.evaluation.EvaluationDataset` instance containing
             evaluation results.

    The default evaluator supports the 'regressor' and 'classifer' model types.

    For both the 'regressor' and 'classifer' types, the default evaluator will generate model
    summary plots and feature importance plots generated by shap explainer.

    For regressor model, the default evaluator will additionally log:

     - **metrics**: example_count, mean_absolute_error, mean_squared_error, root_mean_squared_error,
       sum_on_label, mean_on_label, r2_score, max_error, mean_absolute_percentage_error.

    For binary classifier, the default evaluator will additionally log:

     - **metrics**: true_negatives, false_positives, false_negatives, true_positives, recall,
       precision, f1_score, accuracy, example_count, log_loss, roc_auc, precision_recall_auc.
     - **artifacts**: lift curve plot, precision-recall plot, ROC plot.

    For multiclass classifier, the default evaluator will additionally log:

     - **metrics**: accuracy, example_count, f1_score_micro, f1_score_macro, log_loss
     - **artifacts**: A CSV file for "per_class_metrics" (per-class metrics includes true_negatives/
       false_positives/false_negatives/true_positives/recall/precision/roc_auc,
       precision_recall_auc), precision-recall merged curves plot, ROC merged curves plot.

    The available `evaluator_config` options for the default evaluator include:

     - **log_model_explainability**: A boolean value specifying whether or not to log model
       explainability insights, default value is True.
     - **explainability_algorithm**: A string to specify the SHAP Explainer algorithm for model
       explainability. Supported algorithm includes: 'exact', 'permutation', 'partition'.
       If not set, `shap.Explainer` is used with the "auto" algorithm, which chooses the best
       Explainer based on the model.
     - **explainability_nsamples**: The number of sample rows to use for computing model
       explainability insights. Default value is 2000.
     - **max_num_classes_threshold_logging_roc_pr_curve_for_multiclass_classifier**:
       For multiclass classifier, specify the max number of classes which allow logging per-class
       ROC curve and Precision-Recall curve.

    Limitations of evaluation dataset:
     - If the input dataset is pandas dataframe, the feature columns in pandas dataframe must be
       scalar value columns, other object types (nd.array/list/etc.) are not supported yet.
     - If the mlflow model to be evaluated is a pyspark ML model, then the input data must
       be a spark DataFrame or pandas DataFrame contains a feature column with values of type
       "pyspark.ml.linalg.Vector", and a label column.
     - For classifier, evaluation dataset labels must contains all distinct values, the dataset
       labels data will be used to infer the number of classes. For binary classifier, the
       negative label value must be 0 or -1 or False, and the positive label value must be
       1 or True.
       For multiclass classifier, if logging explainability insights enabled, the label values
       must be number type.

    Limitations of metrics/artifacts computation:
     - For classifier, some metrics and plot computation require model provides
       "predict probability" function. Currently, for sklearn model, we will extract "predict_proba"
       method from the raw model to achieve this, for other model, it will skip logging
       metrics/artifacts which require probability prediction.

    Limitations of default evaluator logging model explainability insights:
     - The `shap.Explainer` "auto" algorithm will choose Linear explainer for linear model,
       and choose Tree explainer for tree model. But the shap Linear/Tree explainer does not
       support multi-class classifier, in this case, default evaluator will fallback to use
       shap Exact or Permutation explainer.
     - Logging model explainability insights is not currently supported for PySpark models.
    """
    from mlflow.pyfunc import PyFuncModel

    if isinstance(model, str):
        model = mlflow.pyfunc.load_model(model)
    elif isinstance(model, PyFuncModel):
        pass
    else:
        raise ValueError(
            "The model argument must be a string URI referring to an MLflow model or "
            "an instance of `mlflow.pyfunc.PyFuncModel`."
        )

    (
        evaluator_name_list,
        evaluator_name_to_conf_map,
    ) = _normalize_evaluators_and_evaluator_config_args(evaluators, evaluator_config)

    with _start_run_or_reuse_active_run(run_id) as actual_run_id:
        return _evaluate(
            model=model,
            model_type=model_type,
            dataset=dataset,
            actual_run_id=actual_run_id,
            evaluator_name_list=evaluator_name_list,
            evaluator_name_to_conf_map=evaluator_name_to_conf_map,
        )
