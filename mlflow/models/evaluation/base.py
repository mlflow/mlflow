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
import logging
import struct


_logger = logging.getLogger(__name__)


class EvaluationMetrics(dict):
    """
    A dictionary of model evaluation metrics.
    """

    pass


class EvaluationArtifact:
    """
    A model evaluation artifact containing an artifact uri and content.
    """

    def __init__(self, uri, content=None):
        self._uri = uri
        self._content = content

    def _load_content_from_file(self, local_artifact_path):
        """
        Abstract interface to load the content from local artifact file path,
        and return the loaded content.
        """
        raise NotImplementedError()

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

    def save(self, output_artifact_path):
        """Save artifact content into specified path."""
        raise NotImplementedError()

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


class EvaluationDataset:
    """
    An input dataset for model evaluation. This is intended for use with the
    :py:func:`mlflow.models.evaluation.evaluate()`
    API.
    """

    NUM_SAMPLE_ROWS_FOR_HASH = 5
    SPARK_DATAFRAME_LIMIT = 10000

    def __init__(self, data, labels, name=None, path=None):
        """
        :param data: One of the following:
         - A numpy array or list of evaluation features, excluding labels.
         - A Pandas DataFrame, or a spark DataFrame,
           containing evaluation features and labels. All columns will be regarded as feature
           columns except the "labels" column.

        :param labels: One of the following:
         - A numpy array or list of evaluation labels, if `data` is also a numpy array or list.
         - The string name of a column from `data` that contains evaluation labels, if `data`
           is a DataFrame.

        :param name: (Optional) The name of the dataset (must not contain ").

        :param path: (Optional) the path to a serialized DataFrame (must not contain ").
          (e.g. a delta table, parquet file)
        """
        import numpy as np
        import pandas as pd

        try:
            from pyspark.sql import DataFrame as SparkDataFrame

            supported_dataframe_types = (pd.DataFrame, SparkDataFrame)
        except ImportError:
            supported_dataframe_types = (pd.DataFrame,)

        if name is not None and '"' in name:
            raise ValueError(f'Dataset name cannot include a double quote (") but got {name}')
        if path is not None and '"' in path:
            raise ValueError(f'Dataset path cannot include a double quote (") but got {path}')

        if isinstance(data, (np.ndarray, list)):
            if not isinstance(labels, (np.ndarray, list)):
                raise ValueError(
                    "If data is a numpy array or list of evaluation features, "
                    "labels must be a numpy array or list of evaluation labels"
                )
        elif isinstance(data, supported_dataframe_types):
            if not isinstance(labels, str):
                raise ValueError(
                    "If data is a Pandas DataFrame or Spark DataFrame, labels must be the "
                    "string name of a column from `data` that contains evaluation labels"
                )
        else:
            raise ValueError(
                "The data argument must be a numpy array, a list or a Pandas DataFrame, or "
                "spark DataFrame if pyspark package installed."
            )

        self._user_specified_name = name
        self._original_data = data
        self._data = None
        self.labels = labels
        self.path = path
        self._hash = None

    @property
    def data(self):
        """
        Return original data if data is numpy array or pandas dataframe,
        For spark dataframe, will only return the first SPARK_DATAFRAME_LIMIT rows as pandas
        dataframe and emit warning.
        """
        if self._data is not None:
            return self._data

        try:
            from pyspark.sql import DataFrame as SparkDataFrame

            spark_df_type = SparkDataFrame
        except ImportError:
            spark_df_type = None

        if spark_df_type and isinstance(self._original_data, spark_df_type):
            self._data = self._original_data.limit(
                EvaluationDataset.SPARK_DATAFRAME_LIMIT
            ).toPandas()
            _logger.warning(
                f"Specified Spark DataFrame is too large for model evaluation. Only "
                f"the first {EvaluationDataset.SPARK_DATAFRAME_LIMIT} rows will be used."
            )
        else:
            self._data = self._original_data

        return self._data

    def _extract_features_and_labels(self):
        """
        Extract features data and labels data.
        For spark dataframe, will only extract the first SPARK_DATAFRAME_LIMIT rows data
        and emit warning.
        """
        import numpy as np

        if isinstance(self.data, np.ndarray):
            return self.data, self.labels
        else:
            return (
                self.data.drop(self.labels, axis=1, inplace=False),
                self.data[self.labels].to_numpy(),
            )

    @staticmethod
    def _convert_uint64_ndarray_to_bytes(array):
        assert len(array.shape) == 1
        # see struct pack format string https://docs.python.org/3/library/struct.html#format-strings
        return struct.pack(f">{array.size}Q", *array)

    @staticmethod
    def _array_like_obj_to_bytes(data):
        """
        Helper method to convert pandas dataframe/numpy array/list into bytes for
        MD5 calculation purpose.
        """
        from pandas.util import hash_pandas_object, hash_array
        import numpy as np
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return EvaluationDataset._convert_uint64_ndarray_to_bytes(hash_pandas_object(data))
        elif isinstance(data, np.ndarray):
            return EvaluationDataset._convert_uint64_ndarray_to_bytes(
                hash_array(data.flatten(order="C"))
            )
        elif isinstance(data, list):
            return EvaluationDataset._convert_uint64_ndarray_to_bytes(hash_array(np.array(data)))
        else:
            raise ValueError("Unsupported data type.")

    @staticmethod
    def _gen_md5_for_arraylike_obj(md5_gen, data):
        """
        Helper method to generate MD5 hash array-like object, the MD5 will calculate over:
         - array length
         - first NUM_SAMPLE_ROWS_FOR_HASH rows content
         - last NUM_SAMPLE_ROWS_FOR_HASH rows content
        """
        import numpy as np

        len_bytes = EvaluationDataset._convert_uint64_ndarray_to_bytes(
            np.array([len(data)], dtype="uint64")
        )
        md5_gen.update(len_bytes)
        if len(data) < EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH * 2:
            md5_gen.update(EvaluationDataset._array_like_obj_to_bytes(data))
        else:
            md5_gen.update(
                EvaluationDataset._array_like_obj_to_bytes(
                    data[: EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH]
                )
            )
            md5_gen.update(
                EvaluationDataset._array_like_obj_to_bytes(
                    data[-EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH :]
                )
            )

    @property
    def name(self):
        """
        Dataset name, which is specified dataset name or the dataset hash if user don't specify
        name.
        """
        return self._user_specified_name if self._user_specified_name is not None else self.hash

    @property
    def hash(self):
        """
        Compute a hash from the specified dataset by selecting the first 5 records, last 5 records,
        dataset size and feeding them through a cheap, low-collision hash function
        """
        import numpy as np
        import pandas as pd

        if self._hash is None:
            md5_gen = hashlib.md5()
            if isinstance(self.data, np.ndarray):
                EvaluationDataset._gen_md5_for_arraylike_obj(md5_gen, self.data)
                EvaluationDataset._gen_md5_for_arraylike_obj(md5_gen, self.labels)
            elif isinstance(self.data, pd.DataFrame):
                column_names = ",".join(self.data.columns)
                meta_str = f"columns={column_names}\nlabels={self.labels}"
                md5_gen.update(meta_str.encode("UTF-8"))
                EvaluationDataset._gen_md5_for_arraylike_obj(md5_gen, self.data)
            self._hash = md5_gen.hexdigest()
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

    def _log_dataset_tag(self, client, run_id):
        """
        Log dataset metadata as a tag "mlflow.datasets", if the tag already exists, it will
        append current dataset metadata into existing tag content.
        """
        existing_dataset_metadata_str = client.get_run(run_id).data.tags.get(
            "mlflow.datasets", "[]"
        )
        dataset_metadata_list = json.loads(existing_dataset_metadata_str)

        for metadata in dataset_metadata_list:
            if metadata["hash"] == self.hash and metadata["name"] == self._user_specified_name:
                break
        else:
            dataset_metadata_list.append(self._metadata)

        dataset_metadata_str = json.dumps(dataset_metadata_list, separators=(",", ":"))
        client.log_batch(
            run_id,
            tags=[RunTag("mlflow.datasets", dataset_metadata_str)],
        )


class ModelEvaluator:
    def can_evaluate(self, model_type, evaluator_config=None, **kwargs) -> bool:
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

    def evaluate(
        self,
        model,
        model_type,
        dataset,
        run_id,
        evaluator_config,
        **kwargs,
    ):
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
                "An active run exists, you cannot specify another run_id when " "evaluating."
            )
        yield active_run.info.run_id


def evaluate(
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
    """
    # import _model_evaluation_registry and PyFuncModel inside function to avoid circuit importing
    from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry
    from mlflow.pyfunc import PyFuncModel

    if not evaluators:
        evaluators = list(_model_evaluation_registry._registry.keys())

    if isinstance(evaluators, str):
        evaluators = [evaluators]
        if not (evaluator_config is None or isinstance(evaluator_config, dict)):
            raise ValueError(
                "If `evaluators` argument is a str, evaluator_config must be None or a dict."
            )
        evaluator_config = {evaluators[0]: evaluator_config}
    elif isinstance(evaluators, list):
        evaluators = set(evaluators)
        if not (
            isinstance(evaluator_config, dict)
            and all(k in evaluators and isinstance(v, dict) for k, v in evaluator_config.items())
        ):
            raise ValueError(
                "If `evaluators` argument is a evaluator name list, evaluator_config"
                "must be a dict contains mapping from evaluator name to individual "
                "evaluator config dict."
            )

    if isinstance(model, str):
        model = mlflow.pyfunc.load_model(model)
    elif isinstance(model, PyFuncModel):
        pass
    else:
        raise ValueError(
            "The model argument must be a string URI referring to an MLflow model or "
            "an instance of `mlflow.pyfunc.PyFuncModel`."
        )

    with _start_run_or_reuse_active_run(run_id) as actual_run_id:
        client = mlflow.tracking.MlflowClient()
        dataset._log_dataset_tag(client, actual_run_id)

        eval_results = []
        for evaluator_name in evaluators:
            config = evaluator_config.get(evaluator_name) or {}
            try:
                evaluator = _model_evaluation_registry.get_evaluator(evaluator_name)
            except MlflowException:
                _logger.warning(f"Evaluator '{evaluator_name}' is not registered.")
                continue

            if evaluator.can_evaluate(model_type, config):
                _logger.info(f"Evaluating the model with the {evaluator_name} evaluator.")
                result = evaluator.evaluate(model, model_type, dataset, actual_run_id, config)
                eval_results.append(result)

        merged_eval_result = EvaluationResult(EvaluationMetrics(), dict())
        for eval_result in eval_results:
            merged_eval_result.metrics.update(eval_result.metrics)
            merged_eval_result.artifacts.update(eval_result.artifacts)

        return merged_eval_result
