import functools
from typing import Any, Callable, Optional

from mlflow.store.artifact.utils.models import _parse_model_id_if_present


class _ModelTracker:
    def __init__(self):
        self.models: dict[int, int] = {}

    def set_id(self, model: Any, model_id: str) -> None:
        """
        Associate the given model instance with the given model ID.
        """
        self.models[id(model)] = model_id

    def __call__(self, load_model_fn: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorates `load_model` to track models loaded by `load_model`.
        """

        @functools.wraps(load_model_fn)
        def new_load_model(model_uri: str, *args, **kwargs) -> Any:
            model = load_model_fn(model_uri, *args, **kwargs)
            if model_id := _parse_model_id_if_present(model_uri):
                self.set_id(model, model_id)
            return model

        return new_load_model

    def get_id(self, model: Any) -> Optional[str]:
        """
        Retrieve the model ID associated with the given model instance.
        """
        return self.models.get(id(model))

    def reset(self) -> None:
        self.models.clear()


_MODEL_TRACKER = _ModelTracker()
