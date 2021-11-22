from typing import Dict, Union
import mlflow
import hashlib
import time
import numpy as np
import pandas as pd
import pickle
import json
import os
from mlflow.exceptions import MlflowException
from mlflow.utils.file_utils import TempDir
from mlflow.entities import Metric, RunTag
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import _get_fully_qualified_class_name, load_class
from mlflow.pyfunc import PyFuncModel
from mlflow.models.evaluation.evaluator_registry import ModelEvaluatorRegistry


class EvaluationMetrics(dict):
    pass


class EvaluationArtifact:
    def __init__(self, location, content=None):
        self._content = content
        self._location = location

    @classmethod
    def load_content_from_file(cls, local_artifact_path):
        raise NotImplementedError()

    @classmethod
    def save_content_to_file(cls, content, output_artifact_path):
        raise NotImplementedError()

    @property
    def content(self):
        """
        The content of the artifact (representation varies)
        """
        if self._content is None:
            with TempDir() as temp_dir:
                local_artifact_file = temp_dir.path("local_artifact")
                _download_artifact_from_uri(self._location, local_artifact_file)
                self._content = self.load_content_from_file(local_artifact_file)

        return self._content

    @property
    def location(self) -> str:
        """
        The location of the artifact
        """
        return self._location


class EvaluationResult:
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

        for artifact_name, meta in artifacts_metadata:
            location = meta["location"]
            ArtifactCls = load_class(meta["class_name"])
            content = ArtifactCls.load_content_from_file(os.path.join(path, artifact_name))
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

        for artifact_name, artifact in self.artifacts.items():
            artifact.save_content_to_file(artifact.content, os.path.join(path, artifact_name))

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


class EvaluationDataset:
    """
    Represents an input dataset for model evaluation. This is intended for
    use with the `mlflow.evaluate()`API.
    """

    NUM_SAMPLE_ROWS_FOR_HASH = 5

    def __init__(self, data, labels, name=None, path=None):
        """
        :param data: One of the following:
         - A numpy array or list of evaluation features, excluding labels.
         - A Pandas DataFrame, or the path to a serialized DataFrame,
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
        if name is not None and '"' in name:
            raise ValueError(f'Dataset name cannot include " but get name {name}')
        if path is not None and '"' in path:
            raise ValueError(f'Dataset path cannot include " but get name {path}')

        if isinstance(data, (np.ndarray, list)):
            if not isinstance(labels, (np.ndarray, list)):
                raise ValueError(
                    'If data is a numpy array or list of evaluation features, '
                    'labels must be a numpy array or list of evaluation labels'
                )
        elif isinstance(data, pd.DataFrame):
            if not isinstance(labels, str):
                raise ValueError(
                    'If data is a Pandas DataFrame, labels must be the string name of a column '
                    'from `data` that contains evaluation labels'
                )
        else:
            raise ValueError('The data argument must be a numpy array, a list or a '
                             'Pandas DataFrame.')

        self._user_specified_name = name
        self.data = data
        self.labels = labels
        self.path = path
        self._hash = None

    def _extract_features_and_labels(self):
        if isinstance(self.data, np.ndarray):
            return self.data, self.labels
        elif isinstance(self.data, pd.DataFrame):
            feature_cols = [x for x in self.data.columns if x != self.labels]
            return self.data[feature_cols], self.data[self.labels]
        else:
            raise ValueError(f'Unsupported data type: {type(self.data)}')

    @staticmethod
    def _gen_md5_for_arraylike_obj(md5_gen, data):
        md5_gen.update(pickle.dumps(len(data)))
        if len(data) < EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH * 2:
            md5_gen.update(pickle.dumps(data))
        else:
            md5_gen.update(pickle.dumps(data[: EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH]))
            md5_gen.update(pickle.dumps(data[-EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH :]))

    @property
    def name(self):
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
        metadata = {
            "name": self.name,
            "hash": self.hash,
        }
        if self.path is not None:
            metadata["path"] = self.path
        return metadata


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

    def compute_metrics_and_compute_and_log_artifacts(
        self, model, model_type, dataset, evaluator_config, run_id
    ):
        """
        return an tuple of:
         - an instance of EvaluationMetrics
         - a dict of artifact_name -> instance_of_EvaluationArtifact
        and log artifacts into run specified by run_id
        """
        raise NotImplementedError()

    def evaluate(
            self,
            model: PyFuncModel,
            model_type,
            dataset,
            run_id=None,
            evaluator_config=None,
            **kwargs
    ) -> EvaluationResult:
        """
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
        client = mlflow.tracking.MlflowClient()
        self.mlflow_client = client

        def do_evaluate(_run_id):
            timestamp = int(time.time() * 1000)
            existing_dataset_metadata_str = client.get_run(_run_id).data.tags.get("mlflow.datasets")
            if existing_dataset_metadata_str is not None:
                dataset_metadata_list = json.loads(existing_dataset_metadata_str)
            else:
                dataset_metadata_list = []

            metadata_exists = False
            for metadata in dataset_metadata_list:
                if (
                    metadata["hash"] == dataset.hash
                    and metadata["name"] == dataset._user_specified_name
                ):
                    metadata_exists = True
                    break

            if not metadata_exists:
                dataset_metadata_list.append(dataset._metadata)

            dataset_metadata_str = json.dumps(dataset_metadata_list)

            metrics_dict, artifacts_dict = self.compute_metrics_and_compute_and_log_artifacts(
                model, model_type, dataset, evaluator_config, _run_id
            )
            client.log_batch(
                _run_id,
                metrics=[
                    Metric(key=f"{key}_on_{dataset.name}", value=value, timestamp=timestamp, step=0)
                    for key, value in metrics_dict.items()
                ],
                tags=[RunTag("mlflow.datasets", dataset_metadata_str)],
            )

            return EvaluationResult(metrics_dict, artifacts_dict)

        if mlflow.active_run() is not None:
            return do_evaluate(mlflow.active_run().info.run_id)
        else:
            with mlflow.start_run(run_id=run_id) as run:
                return do_evaluate(run.info.run_id)


_model_evaluation_registry = ModelEvaluatorRegistry()
_model_evaluation_registry.register_entrypoints()


def evaluate(
        model: Union[str, PyFuncModel],
        model_type, dataset,
        run_id=None,
        evaluators=None,
        evaluator_config=None
) -> EvaluationResult:
    """
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
    if evaluators is None:
        evaluators = "default"

    if not isinstance(evaluators, list):
        evaluators = [evaluators]
        evaluator_config = {evaluators[0]: evaluator_config}

    if isinstance(model, str):
        model = mlflow.pyfunc.load_model(model)

    eval_results = []
    for evaluator_name in evaluators:
        config = evaluator_config[evaluator_name]
        try:
            evaluator = _model_evaluation_registry.get_evaluator(evaluator_name)
        except MlflowException:
            continue

        if evaluator.can_evaluate(model_type, config):
            result = evaluator.evaluate(model, model_type, dataset, run_id, config)
            eval_results.append(result)

    merged_eval_result = EvaluationResult(EvaluationMetrics(), dict())
    for eval_result in eval_results:
        merged_eval_result.metrics.update(eval_result.metrics)
        merged_eval_result.artifacts.update(eval_result.artifacts)

    return merged_eval_result
