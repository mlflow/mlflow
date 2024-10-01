import warnings

from mlflow.exceptions import MlflowException
from mlflow.utils.import_hooks import register_post_import_hook
from mlflow.utils.plugins import get_entry_points


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
        # Register ModelEvaluator implementation provided by other packages
        for entrypoint in get_entry_points("mlflow.model_evaluator"):
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
                f"Could not find a registered model evaluator for: {evaluator_name}. "
                f"Currently registered evaluator names are: {list(self._registry.keys())}"
            )
        return evaluator_cls()


_model_evaluation_registry = ModelEvaluatorRegistry()


def register_evaluators(module):
    from mlflow.models.evaluation.default_evaluator import DefaultEvaluator

    module._model_evaluation_registry.register("default", DefaultEvaluator)
    module._model_evaluation_registry.register_entrypoints()


# Put it in post-importing hook to avoid circuit importing
register_post_import_hook(register_evaluators, __name__, overwrite=True)
