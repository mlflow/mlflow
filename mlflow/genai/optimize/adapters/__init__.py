from mlflow.genai.optimize.adapters.base import BasePromptAdapter

__all__ = ["BasePromptAdapter"]


def get_default_adapter() -> BasePromptAdapter:
    # TODO: Implement default adapter
    raise NotImplementedError("Default adapter not implemented")
