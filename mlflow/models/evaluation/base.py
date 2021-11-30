from typing import Dict, Union
import mlflow
import hashlib
import time
import numpy as np
import pandas as pd
import json
import os
from mlflow.exceptions import MlflowException
from mlflow.utils.file_utils import TempDir
from mlflow.entities import Metric, RunTag
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.class_utils import _get_class_from_string
import logging


_logger = logging.getLogger(__name__)


class EvaluationMetrics(dict):
    """
    Represent a dict of metrics.
    """

    pass


class EvaluationArtifact:
    """
    Represent a artifact. Contains artifact uri and content.
    """

    def __init__(self, uri, content=None):
        self._uri = uri
        self._content = content

    def _load_content_from_file(self, local_artifact_path):
        raise NotImplementedError()

    def load(self, local_artifact_path=None):
        """
        If `local_artifact_path` is None, download artifact from the artifact uri and load it.
        otherwise load artifact content from specified path.
        """
        if local_artifact_path is None:
            return self._load_content_from_file(local_artifact_path)
        else:
            with TempDir() as temp_dir:
                temp_dir_path = temp_dir.path()
                _download_artifact_from_uri(self._uri, temp_dir_path)
                local_artifact_file = temp_dir.path(os.listdir(temp_dir_path)[0])
                return self._load_content_from_file(local_artifact_file)

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

        for artifact_name, meta in artifacts_metadata:
            location = meta["location"]
            ArtifactCls = _get_class_from_string(meta["class_name"])
            content = ArtifactCls.load_content_from_file(os.path.join(artifacts_dir, artifact_name))
            artifacts[artifact_name] = ArtifactCls(location=location, content=content)

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    def save(self, path):
        """Write the evaluation results to the specified local filesystem path"""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metrics.json"), "w") as fp:
            json.dump(self.metrics, fp)

        artifacts_metadata = {
            artifact_name: {
                "location": artifact.location,
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
    def metrics(self) -> EvaluationMetrics:
        """
        A dictionary mapping scalar metric names to scalar metric values
        """
        return self._metrics

    @property
    def artifacts(self) -> Dict[str, EvaluationArtifact]:
        """
        A dictionary mapping standardized artifact names (e.g. "roc_data") to
        artifact content and location information
        """
        return self._artifacts


_cached_mlflow_client = None


class EvaluationDataset:
    """
    Represents an input dataset for model evaluation. This is intended for
    use with the `mlflow.evaluate()`API.
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
        try:
            from pyspark.sql import DataFrame as SparkDataFrame

            supported_dataframe_types = (pd.DataFrame, SparkDataFrame)
        except ImportError:
            supported_dataframe_types = (pd.DataFrame,)

        if name is not None and '"' in name:
            raise ValueError(f'Dataset name cannot include a double quote (") but got name {name}')
        if path is not None and '"' in path:
            raise ValueError(f'Dataset path cannot include a double quote (") but got name {path}')

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
                "The data argument must be a numpy array, a list or a " "Pandas DataFrame."
            )

        self._user_specified_name = name
        self.data = data
        self.labels = labels
        self.path = path
        self._hash = None

    def _extract_features_and_labels(self):
        """
        Extract features data and labels data.
        For spark dataframe, will only extract the first SPARK_DATAFRAME_LIMIT rows data
        and emit warning.
        """
        if isinstance(self.data, np.ndarray):
            return self.data, self.labels
        else:
            if not isinstance(self.data, pd.DataFrame):
                data = self.data.limit(EvaluationDataset.SPARK_DATAFRAME_LIMIT).toPandas()
                _logger.warning(
                    f"Only the first {EvaluationDataset.SPARK_DATAFRAME_LIMIT} rows in the "
                    f"spark dataframe are examined."
                )
            else:
                data = self.data
            feature_cols = [x for x in data.columns if x != self.labels]
            return data[feature_cols], data[self.labels]

    @staticmethod
    def _array_like_obj_to_bytes(data):
        """
        Helper method to convert pandas dataframe/numpy array/list into bytes for
        MD5 calculation purpose.
        """
        if isinstance(data, pd.DataFrame):
            return data.to_numpy().tobytes() + ",".join(list(data.columns)).encode("UTF-8")
        elif isinstance(data, np.ndarray):
            return data.tobytes()
        elif isinstance(data, list):
            return np.array(data).tobytes()
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
        md5_gen.update(np.int64(len(data)).tobytes())
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
        if self._hash is None:
            md5_gen = hashlib.md5()
            if isinstance(self.data, np.ndarray):
                EvaluationDataset._gen_md5_for_arraylike_obj(md5_gen, self.data)
                EvaluationDataset._gen_md5_for_arraylike_obj(md5_gen, self.labels)
            elif isinstance(self.data, pd.DataFrame):
                EvaluationDataset._gen_md5_for_arraylike_obj(md5_gen, self.data)
                md5_gen.update(self.labels.encode("UTF-8"))
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
        existing_dataset_metadata_str = client.get_run(run_id).data.tags.get("mlflow.datasets")
        if existing_dataset_metadata_str is not None:
            dataset_metadata_list = json.loads(existing_dataset_metadata_str)
        else:
            dataset_metadata_list = []

        metadata_exists = False
        for metadata in dataset_metadata_list:
            if metadata["hash"] == self.hash and metadata["name"] == self._user_specified_name:
                metadata_exists = True
                break

        if not metadata_exists:
            dataset_metadata_list.append(self._metadata)

        dataset_metadata_str = json.dumps(dataset_metadata_list)
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
        :param **kwargs: For forwards compatibility, a placeholder for additional
                         arguments that may be added to the evaluation interface
                         in the future.
        :return: True if the evaluator can evaluate the specified model on the
                 specified dataset. False otherwise.
        """
        raise NotImplementedError()

    def _log_metrics(self, run_id, metrics, dataset_name):
        """
        Helper method to log metrics into specified run.
        """
        client = mlflow.tracking.MlflowClient()
        timestamp = int(time.time() * 1000)
        client.log_batch(
            run_id,
            metrics=[
                Metric(key=f"{key}_on_{dataset_name}", value=value, timestamp=timestamp, step=0)
                for key, value in metrics.items()
            ],
        )

    def evaluate(
        self,
        model: "mlflow.pyfunc.PyFuncModel",
        model_type,
        dataset,
        run_id,
        evaluator_config,
        **kwargs,
    ) -> "mlflow.models.evaluation.EvaluationResult":
        """
        The abstract API to log metrics and artifacts, and return evaluation results.

        :param model: A pyfunc model instance.
        :param model_type: A string describing the model type (e.g., "regressor",
                   "classifier", …).
        :param dataset: An instance of `EvaluationDataset` containing features
                        and labels (optional) for model evaluation.
        :param run_id: The ID of the MLflow Run to which to log results.
        :param evaluator_config: A dictionary of additional configurations for
                                 the evaluator.
        :param **kwargs: For forwards compatibility, a placeholder for additional
                         arguments that may be added to the evaluation interface
                         in the future.
        :return: An `EvaluationResult` instance containing evaluation results.
        """
        raise NotImplementedError()


def list_evaluators():
    """
    Return a name list for all available Evaluators.
    """
    # import _model_evaluation_registry inside function to avoid circuit importing
    from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry

    return list(_model_evaluation_registry._registry.keys())


class StartRunOrReuseActiveRun:
    """
    A manager context return:
     - If there's an active run, return the active run id.
     - otherwise start a mflow run with the specified run_id.
    """

    def __init__(self, run_id):
        self.user_specified_run_id = run_id
        self.managed_run = None

    def __enter__(self):
        if mlflow.active_run() is not None:
            active_run_id = mlflow.active_run().info.run_id
            if (
                self.user_specified_run_id is not None
                and self.user_specified_run_id != active_run_id
            ):
                raise ValueError(
                    "An active run exists, you cannot specify another run_id when " "evaluating."
                )
            return active_run_id
        else:
            if self.user_specified_run_id is None:
                raise ValueError(
                    "Active run does not exist, you need specify a run_id when " "evaluating."
                )
            self.managed_run = mlflow.start_run(run_id=self.user_specified_run_id).__enter__()
            return self.user_specified_run_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.managed_run is not None:
            return self.managed_run.__exit__(exc_type, exc_val, exc_tb)


def evaluate(
    model: Union[str, "mlflow.pyfunc.PyFuncModel"],
    model_type,
    dataset,
    run_id=None,
    evaluators=None,
    evaluator_config=None,
) -> "mlflow.models.evaluation.EvaluationResult":
    """
    Evaluate a pyfunc model on specified dataset, log evaluation results (metrics and
    artifacts) into active run or specified mlflow run), and return evaluation results
    containing metrics and artifacts.

    :param model: A pyfunc model instance, or a URI referring to such a model.

    :param model_type: A string describing the model type. The default evaluator
                       supports "regressor" and "classifier" as model types.
    :param dataset: An instance of `EvaluationDataset` containing features
                    labels (optional) for model evaluation.
    :param run_id: The ID of the MLflow Run to which to log results. If
                   unspecified, behavior depends on the specified `evaluator`.
                   When `run_id` is unspecified, the default evaluator logs
                   results to the current active run, creating a new active run if
                   one does not exist.
    :param evaluators: The name of the evaluator to use for model evaluations, or
                       a list of evaluator names. If unspecified, all evaluators
                       capable  of evaluating the specified model on the specified
                       dataset are used. The default evaluator can be referred to
                       by the name 'default'.
    :param evaluator_config: A dictionary of additional configurations to supply
                             to the evaluator. If multiple evaluators are
                             specified, each configuration should be supplied as
                             a nested dictionary whose key is the evaluator name.
    :return: An `EvaluationResult` instance containing evaluation results.
    """
    # import _model_evaluation_registry and PyFuncModel inside function to avoid circuit importing
    from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry
    from mlflow.pyfunc import PyFuncModel

    if not evaluators:
        evaluators = "default"

    if isinstance(evaluators, str):
        evaluators = [evaluators]
        if not (evaluator_config is None or isinstance(evaluator_config, dict)):
            raise ValueError(
                "If `evaluators` argument is a str, evaluator_config must be None " "or a dict."
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
            "The model argument must be a URI str referring to mlflow model or "
            "an instance of `mlflow.pyfunc.PyFuncModel`."
        )

    with StartRunOrReuseActiveRun(run_id) as actual_run_id:
        client = mlflow.tracking.MlflowClient()
        dataset._log_dataset_tag(client, actual_run_id)

        eval_results = []
        for evaluator_name in evaluators:
            config = evaluator_config.get(evaluator_name)
            if config is None:
                config = {}
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
