from functools import lru_cache

from mlflow.assistant.config import AssistantConfig


@lru_cache(maxsize=100)
def get_project_path(experiment_id: str) -> str | None:
    """Get the project path for a given experiment ID.

    Args:
        experiment_id: The experiment ID to look up.

    Returns:
        The project path if found, None otherwise.
    """
    config = AssistantConfig.load()
    return config.get_project_path(experiment_id)


__all__ = ["get_project_path", "AssistantConfig"]
