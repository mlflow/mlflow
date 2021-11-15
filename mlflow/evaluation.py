from collections.abc import Mapping
from typing import List
import entrypoints
import warnings
import mlflow
from mlflow.exceptions import MlflowException


class EvaluationMetrics(dict):
    pass


class EvaluationArtifact:

    @property
    def content(self):
        """
        The content of the artifact (representation varies)
        """
        raise NotImplementedError()

    @property
    def location(self) -> str:
        """
        The location of the artifact
        """
        raise NotImplementedError()


class EvaluationResult:

    @classmethod
    def load(cls, path):
        """Load the evaluation results from the specified local filesystem path"""
        raise NotImplementedError()

    def save(self, path):
        """Write the evaluation results to the specified local filesystem path"""
        # We will likely avoid serializing artifacts themselves, just locations.
        # Deserialization will resolve locations to artifact contents.
        raise NotImplementedError()

    @property
    def metrics(self) -> EvaluationMetrics:
        """
        A dictionary mapping scalar metric names to scalar metric values
        """
        raise NotImplementedError()

    @property
    def artifacts(self) -> Mapping[str, EvaluationArtifact]:
        """
        A dictionary mapping standardized artifact names (e.g. "roc_data") to
        artifact content and location information
        """
        raise NotImplementedError()


class EvaluationDataset:
    """
    Represents an input dataset for model evaluation. This is intended for
    use with the `mlflow.evaluate()`API.
    """

    def __init__(self, data, labels=None, name=None):
        """
        :param data: One of the following:
         - A numpy array or list of evaluation features, excluding labels.
         - A Pandas DataFrame, or the path to a serialized DataFrame,
           containing evaluation features and labels.

        :param labels: One of the following:
         - A numpy array or list of evaluation labels, if `data` is also a numpy array or list.
         - The string name of a column from `data` that contains evaluation labels, if `data`
           is a DataFrame.

        :param name: (Optional) The name of the dataset (must not contain ").
        """
        self.name = name
        self.data = data
        self.labels = labels


class ModelEvaluator:

    def can_evaluate(
        model_type, evaluator_config=None, **kwargs
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

    def evaluate(
        predict, dataset, run_id, evaluator_config=None, **kwargs
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
        raise NotImplementedError()


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


def evaluate(
   model, model_type, dataset, run_id=None, evaluators=None, evaluator_config=None
) -> EvaluationResult | Mapping[str, EvaluationResult]:
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

    if isinstance(model, str):
        model = mlflow.pyfunc.load_model(model)

    predict = model.predict

    eval_results = {}
    for evaluator_name in evaluators:
        try:
            evaluator = _model_evaluation_registry.get_evaluator(evaluator_name)
        except MlflowException:
            eval_results[evaluator_name] = None
            continue

        if evaluator.can_evaluate(model_type, evaluator_config):
            result = evaluator.evaluate(predict, dataset, run_id, evaluator_config)
            eval_results[evaluator_name] = result
        else:
            eval_results[evaluator_name] = None

    if len(evaluators) > 1:
        return eval_results
    else:
        return eval_results[evaluators[0]]





