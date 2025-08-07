"""
Registered scorer functionality for MLflow GenAI.

This module provides functions to manage registered scorers that automatically
evaluate traces in MLflow experiments.
"""

from mlflow.genai.scheduled_scorers import ScorerScheduleConfig
from mlflow.genai.scorers.base import Scorer, ScorerSamplingConfig
from mlflow.utils.annotations import experimental

_ERROR_MSG = (
    "The `databricks-agents` package is required to register scorers. "
    "Please install it with `pip install databricks-agents`."
)


def _scheduled_scorer_to_scorer(scheduled_scorer: ScorerScheduleConfig) -> Scorer:
    scorer = scheduled_scorer.scorer
    scorer._sampling_config = ScorerSamplingConfig(
        sample_rate=scheduled_scorer.sample_rate,
        filter_string=scheduled_scorer.filter_string,
    )
    return scorer


@experimental(version="3.2.0")
def list_scorers(*, experiment_id: str | None = None) -> list[Scorer]:
    """
    List all registered scorers for an experiment.

    This function returns all registered scorers configured for the specified experiment,
    or for the current active experiment if no experiment ID is provided.

    Args:
        experiment_id: The ID of the MLflow experiment to list scorers for.
            If None, uses the currently active experiment.

    Returns:
        A list of Scorer objects representing all registered scorers configured
        for the specified experiment.

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.scorers import list_scorers

            # List scorers for a specific experiment
            scorers = list_scorers(experiment_id="12345")
            for scorer in scorers:
                print(f"Scorer: {scorer.name}")
                print(f"Sample rate: {scorer.sample_rate}")
                print(f"Filter: {scorer.filter_string}")

            # List scorers for the current active experiment
            mlflow.set_experiment("my_genai_app_monitoring")
            current_scorers = list_scorers()
            print(f"Found {len(current_scorers)} registered scorers")
    """
    try:
        from databricks.agents.scorers import list_scheduled_scorers
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e

    # Get scheduled scorers from the server
    scheduled_scorers = list_scheduled_scorers(experiment_id=experiment_id)

    # Convert to Scorer instances with registration info
    scorers = []
    for scheduled_scorer in scheduled_scorers:
        scorers.append(_scheduled_scorer_to_scorer(scheduled_scorer))

    return scorers


@experimental(version="3.2.0")
def get_scorer(*, name: str, experiment_id: str | None = None) -> Scorer:
    """
    Retrieve a specific registered scorer by name.

    This function returns a Scorer instance with its current registration
    configuration, including sampling rate and filter criteria.

    Args:
        name: The name of the registered scorer to retrieve.
        experiment_id: The ID of the MLflow experiment containing the scorer.
            If None, uses the currently active experiment.

    Returns:
        A Scorer object with its current registration configuration.

    Example:
        .. code-block:: python

            from mlflow.genai.scorers import get_scorer

            # Get a specific scorer
            my_scorer = get_scorer(name="my_safety_scorer")

            print(f"Sample rate: {my_scorer.sample_rate}")
            print(f"Filter: {my_scorer.filter_string}")

            # Update the scorer
            my_scorer = my_scorer.update(sample_rate=0.5)
    """
    try:
        from databricks.agents.scorers import get_scheduled_scorer
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e

    # Get the scheduled scorer from the server
    scheduled_scorer = get_scheduled_scorer(
        scheduled_scorer_name=name,
        experiment_id=experiment_id,
    )

    # Extract the scorer and set registration fields
    return _scheduled_scorer_to_scorer(scheduled_scorer)


@experimental(version="3.2.0")
def delete_scorer(
    *,
    name: str,
    experiment_id: str | None = None,
) -> None:
    """
    Delete scorer with given name from the server.

    This method permanently removes the scorer registration from the MLflow server.
    After deletion, the scorer will no longer evaluate traces automatically and
    must be registered again if needed.

    Args:
        name: Name of the scorer to delete.
        experiment_id: The ID of the MLflow experiment containing the scorer.
            If None, uses the currently active experiment.

    Returns:
        None

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.scorers import RelevanceToQuery, list_scorers, delete_scorer

            # Register and start a scorer
            mlflow.set_experiment("my_genai_app")
            scorer = RelevanceToQuery().register(name="relevance_checker")

            # List current scorers
            scorers = list_scorers()
            print(f"Active scorers: {[s.name for s in scorers]}")

            # Delete the scorer
            delete_scorer(name="relevance_checker")

            # Verify deletion
            scorers_after = list_scorers()
            print(f"Active scorers after deletion: {[s.name for s in scorers_after]}")

            # To use the scorer again, it must be re-registered
            new_scorer = RelevanceToQuery().register(name="relevance_checker_v2")
    """
    try:
        from databricks.agents.scorers import delete_scheduled_scorer
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e

    delete_scheduled_scorer(
        experiment_id=experiment_id,
        scheduled_scorer_name=name,
    )


# Private functions for internal use by Scorer methods
def add_registered_scorer(
    *,
    name: str,
    scorer: Scorer,
    sample_rate: float,
    filter_string: str | None = None,
    experiment_id: str | None = None,
) -> Scorer:
    """Internal function to add a registered scorer."""
    try:
        from databricks.agents.scorers import add_scheduled_scorer
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e

    scheduled_scorer = add_scheduled_scorer(
        experiment_id=experiment_id,
        scheduled_scorer_name=name,
        scorer=scorer,
        sample_rate=sample_rate,
        filter_string=filter_string,
    )
    return _scheduled_scorer_to_scorer(scheduled_scorer)


def update_registered_scorer(
    *,
    name: str,
    scorer: Scorer | None = None,
    sample_rate: float | None = None,
    filter_string: str | None = None,
    experiment_id: str | None = None,
) -> Scorer:
    """Internal function to update a registered scorer."""
    try:
        from databricks.agents.scorers import update_scheduled_scorer
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e

    scheduled_scorer = update_scheduled_scorer(
        experiment_id=experiment_id,
        scheduled_scorer_name=name,
        scorer=scorer,
        sample_rate=sample_rate,
        filter_string=filter_string,
    )
    return _scheduled_scorer_to_scorer(scheduled_scorer)
