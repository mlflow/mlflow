from typing import Dict, Union
import entrypoints
import warnings
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


class EvaluationMetrics(dict):
    pass


class EvaluationArtifact:

    def __init__(self, location, content=None):
        self._content = content
        self._location = location

    @classmethod
    def load_content_from_file(self, local_artifact_path):
        raise NotImplementedError()

    def save_content_to_file(self, content, output_artifact_path):
        raise NotImplementedError()

    @property
    def content(self):
        """
        The content of the artifact (representation varies)
        """
        if self._content is None:
            with TempDir() as temp_dir:
                local_artifact_file = temp_dir.path('local_artifact')
                _download_artifact_from_uri(self._location, local_artifact_file)
                self._content = self.load_content_from_file(local_artifact_file)

        return self._content

    @property
    def location(self) -> str:
        """
        The location of the artifact
        """
        return self._location

    def __getstate__(self, state):
        state = state.__dict__.copy()
        # skip pickling artifact content
        del state['_content']
        return state


class EvaluationResult:

    def __init__(self, metrics, artifacts):
        self._metrics = metrics
        self._artifacts = artifacts

    @classmethod
    def load(cls, path):
        """Load the evaluation results from the specified local filesystem path"""
        with open(os.path.join(path, 'metrics.json'), 'r') as fp:
            metrics = EvaluationMetrics(json.load(fp))

        with open(os.path.join(path, 'artifacts_metadata.json'), 'r') as fp:
            artifacts_metadata = json.load(fp)

        artifacts = {}

        for artifact_name, meta in artifacts_metadata:
            location = meta['location']
            ArtifactCls = load_class(meta['class_name'])
            content = ArtifactCls.load_content_from_file(os.path.join(path, artifact_name))
            artifacts[artifact_name] = ArtifactCls(location=location, content=content)

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    def save(self, path):
        """Write the evaluation results to the specified local filesystem path"""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'metrics.json'), 'w') as fp:
            json.dump(self.metrics, fp)

        artifacts_metadata = {
            artifact_name: {
                'location': artifact.location,
                'class_name': _get_fully_qualified_class_name(artifact)
            }
            for artifact_name, artifact in self.artifacts.items()
        }
        with open(os.path.join(path, 'artifacts_metadata.json'), 'w') as fp:
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

    def __init__(self, data, labels=None, name=None, path=None):
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

        :param path: (Optional) the path to a serialized DataFrame
          (e.g. a delta table, parquet file)
        """
        self.user_specified_name = name
        self.data = data
        self.labels = labels
        self.path = path
        self._hash = None

    @staticmethod
    def _gen_md5_for_arraylike_obj(md5_gen, data):
        md5_gen.update(pickle.dumps(len(data)))
        if len(data) < EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH * 2:
            md5_gen.update(pickle.dumps(data))
        else:
            md5_gen.update(pickle.dumps(data[:EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH]))
            md5_gen.update(pickle.dumps(data[-EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH:]))

    @property
    def name(self):
        return self.user_specified_name if self.user_specified_name is not None else self.hash

    @property
    def hash(self):
        """
        Compute a hash from the specified dataset by selecting the first 5 records, last 5 records,
        dataset size and feeding them through a cheap, low-collision hash function
        """
        if self._hash is not None:
            return self._hash
        else:
            md5_gen = hashlib.md5()
            if isinstance(self.data, np.ndarray):
                EvaluationDataset._gen_md5_for_arraylike_obj(md5_gen, self.data)
                EvaluationDataset._gen_md5_for_arraylike_obj(md5_gen, self.labels)
            elif isinstance(self.data, pd.DataFrame):
                EvaluationDataset._gen_md5_for_arraylike_obj(md5_gen, self.data)
                md5_gen.update(self.labels.encode('UTF-8'))
            return md5_gen.hexdigest()

    @property
    def metadata(self):
        metadata = {
            'hash': self.hash,
            'path': self.path,
        }
        if self.user_specified_name is not None:
            metadata['name'] = self.user_specified_name
        return metadata


class GetOrCreateRunId:
    """
    Get or create a run, return a run_id
    if user specified a run_id, use it.
    otherwise if there's an active run, use it
    otherwise create a managed run.
    """
    def __init__(self, run_id):
        self.managed_run_context = None
        if run_id is not None:
            self.run_id = run_id
        elif mlflow.active_run() is not None:
            self.run_id = mlflow.active_run().info.run_id
        else:
            self.run_id = None

    def __enter__(self):
        if self.run_id is not None:
            return self.run_id
        else:
            self.managed_run_context = mlflow.start_run()
            return self.managed_run_context.__enter__().info.run_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.managed_run_context is not None:
            return self.managed_run_context.__exit__(exc_type, exc_val, exc_tb)


class ModelEvaluator:

    def can_evaluate(
        self, model_type, evaluator_config=None, **kwargs
    ) -> bool:
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

    def compute_metrics(self, predict, dataset):
        """
        return an instance of EvaluationMetrics
        """
        raise NotImplementedError()

    def compute_and_log_artifacts(self, predict, dataset, run_id, mlflow_client):
        """
        compute and log artifact, and return a dict of
        artifact_name -> instance_of_EvaluationArtifact
        """
        raise NotImplementedError()

    def evaluate(
        self, predict, dataset, run_id=None, evaluator_config=None, **kwargs
    ) -> EvaluationResult:
        """
        :param predict: A function used to compute model predictions. Predict
                        accepts features from the specified `dataset` and
                        feeds them to the model, producing output predictions.
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
        with GetOrCreateRunId(run_id) as run_id:
            metrics_dict = self.compute_metrics(predict, dataset)
            timestamp = int(time.time() * 1000)
            existing_dataset_metadata_str = client.get_run(run_id).data.tags.get('mlflow.datasets')
            if existing_dataset_metadata_str is not None:
                dataset_metadata_list = json.loads(existing_dataset_metadata_str)
            else:
                dataset_metadata_list = []
            dataset_metadata_list.append(dataset.metadata)

            dataset_metadata_str = json.dumps(dataset_metadata_list)

            client.log_batch(
                run_id,
                metrics=[
                    Metric(key=f'{key}_on_{dataset.name}', value=value, timestamp=timestamp, step=0)
                    for key, value in metrics_dict
                ],
                tags=[RunTag('mlflow.datasets', dataset_metadata_str)]
            )
            artifact_dict = self.compute_and_log_artifact(predict, dataset, run_id, client)
            return EvaluationResult(metrics_dict, artifact_dict)


class ModelEvaluatorRegistry:
    """
    Scheme-based registry for model evaluator implementations
    """

    def __init__(self):
        self._registry = {}

    def register(self, scheme, evaluator):
        """Register model evaluator provided by other packages"""
        self._registry[scheme] = evaluator

    def register_entrypoints(self):
        # Register artifact repositories provided by other packages
        for entrypoint in entrypoints.get_group_all("mlflow.model_evaluator"):
            try:
                self.register(entrypoint.name, entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    'Failure attempting to register model evaluator for scheme "{}": {}'.format(
                        entrypoint.name, str(exc)
                    ),
                    stacklevel=2,
                )

    def get_evaluator(self, evaluator_name):
        """
        Get an evaluator instance from the registry based on the name of evaluator
        """
        evaluator_cls = self._registry.get(evaluator_name)
        if evaluator_cls is None:
            raise MlflowException(
                "Could not find a registered model evaluator for: {}. "
                "Currently registered evaluator names are: {}".format(
                    evaluator_name, list(self._registry.keys())
                )
            )
        return evaluator_cls()


_model_evaluation_registry = ModelEvaluatorRegistry()
_model_evaluation_registry.register_entrypoints()


def evaluate(
   model, model_type, dataset, run_id=None, evaluators=None, evaluator_config=None
) -> Union[EvaluationResult, Dict[str, EvaluationResult]]:
    """
    :param model: A model supported by the specified `evaluator`, or a URI
                  referring to such a model. The default evaluator supports the
                  following:

                  - A pyfunc model instance (an instance of class `PyFuncModel`)

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
                       dataset are used.
    :param evaluator_config: A dictionary of additional configurations to supply
                             to the evaluator. If multiple evaluators are
                             specified, each configuration should be supplied as
                             a nested dictionary whose key is the evaluator name.
    :return: An `EvaluationResult` instance containing evaluation results.
    """
    if evaluators is None:
        evaluators = 'default_evaluator'

    if not isinstance(evaluators, list):
        evaluators = [evaluators]
        evaluator_config = {evaluators[0]: evaluator_config}

    if isinstance(model, str):
        model = mlflow.pyfunc.load_model(model)

    predict = model.predict

    eval_results = []
    for evaluator_name in evaluators:
        config = evaluator_config[evaluator_name]
        try:
            evaluator = _model_evaluation_registry.get_evaluator(evaluator_name)
        except MlflowException:
            continue

        if evaluator.can_evaluate(model_type, config):
            result = evaluator.evaluate(predict, dataset, run_id, config)
            eval_results.append(result)

    merged_eval_result = EvaluationResult(EvaluationMetrics(), dict())
    for eval_result in eval_results:
        merged_eval_result.metrics.update(eval_result.metrics)
        merged_eval_result.artifacts.update(eval_result.artifacts)

    return merged_eval_result
