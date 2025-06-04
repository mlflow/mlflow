from typing import Any, Optional, Union
from mlflow.genai.scorers.base import Scorer
from mlflow.utils.annotations import experimental

_ERROR_MSG = (
    "The `databricks-agents` package is required to use `mlflow.genai.scheduled_scorers`. "
    "Please install it with `pip install databricks-agents`."
)


@experimental
class ScheduledScorer:
    """
    A scheduled scorer configuration for automated monitoring of generative AI applications.

    Scheduled scorers are used to automatically evaluate traces logged to MLflow experiments
    by production applications. They are part of Databricks' Lakehouse Monitoring for GenAI,
    which helps track quality metrics like groundedness, safety, and guideline adherence
    alongside operational metrics like volume, latency, and cost.

    When configured, scheduled scorers run automatically in the background to evaluate
    a sample of traces based on the specified sampling rate and filter criteria. The
    evaluation results are displayed in the Traces tab of the MLflow experiment and can be used to
    identify quality issues in production.

    Args:
        scorer_fn: The scorer function to run on sampled traces. Must be either a built-in
            scorer (e.g., Safety, Correctness) or a function decorated with @scorer.
            Subclasses of Scorer are not supported.
        scorer_name: A unique name for this scheduled scorer configuration.
        sample_rate: The fraction of traces to evaluate, between 0.0 and 1.0. For example,
            0.1 means 10% of traces will be randomly selected for evaluation.
        filter_string: An optional MLflow search_traces compatible filter string to apply
            before sampling traces. Only traces matching this filter will be considered
            for evaluation. Uses the same syntax as mlflow.search_traces().

    Example:
        .. code-block:: python

            from mlflow.genai.scorers import Safety, scorer
            from mlflow.genai.scheduled_scorers import ScheduledScorer

            # Using a built-in scorer
            safety_config = ScheduledScorer(
                scorer_fn=Safety(),
                scorer_name="production_safety",
                sample_rate=0.2,  # Evaluate 20% of traces
                filter_string="trace.status = 'OK'",
            )


            # Using a custom scorer
            @scorer
            def response_length(outputs):
                return len(str(outputs)) > 100


            length_config = ScheduledScorer(
                scorer_fn=response_length,
                scorer_name="adequate_length",
                sample_rate=0.1,  # Evaluate 10% of traces
                filter_string="trace.status = 'OK'",
            )

    Note:
        Scheduled scorers are executed automatically by Databricks and do not need to be
        manually triggered. The evaluation results appear in the MLflow experiment's
        monitoring dashboard.
    """

    def __init__(
        self,
        *,
        scorer_fn: Scorer,
        scorer_name: str,
        sample_rate: float,
        filter_string: Optional[str] = None,
    ):
        self.scorer_fn = scorer_fn
        self.scorer_name = scorer_name
        self.sample_rate = sample_rate
        self.filter_string = filter_string


# Scheduled Scorer CRUD operations
@experimental
def add_scheduled_scorer(
    *,
    experiment_id: Optional[str] = None,
    scorer_name: str,
    scorer_fn: Scorer,
    sample_rate: float,
    filter_string: Optional[str] = None,
) -> ScheduledScorer:
    """
    Add a scheduled scorer to automatically monitor traces in an MLflow experiment.

    This function configures a scorer to run automatically on traces logged to the specified
    experiment. The scorer will evaluate a sample of traces based on the sampling rate
    and any filter criteria. Results are displayed in the experiment's monitoring dashboard.

    Args:
        experiment_id: The ID of the MLflow experiment to monitor. If None, uses the
            currently active experiment.
        scorer_name: A unique name for this scheduled scorer within the experiment.
            We recommend using the scorer's name (e.g., scorer_fn.name) for consistency.
        scorer_fn: The scorer function to execute on sampled traces. Must be either a
            built-in scorer or a function decorated with @scorer. Subclasses of Scorer
            are not supported.
        sample_rate: The fraction of traces to evaluate, between 0.0 and 1.0. For example,
            0.3 means 30% of traces will be randomly selected for evaluation.
        filter_string: An optional MLflow search_traces compatible filter string. Only
            traces matching this filter will be considered for evaluation. If None,
            all traces in the experiment are eligible for sampling.

    Returns:
        A ScheduledScorer object representing the configured scheduled scorer.

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.scorers import Safety, Correctness
            from mlflow.genai.scheduled_scorers import add_scheduled_scorer

            # Set up your experiment
            experiment = mlflow.set_experiment("my_genai_app_monitoring")

            # Add a safety scorer to monitor 50% of traces
            safety_scorer = add_scheduled_scorer(
                scorer_name="safety_monitor",
                scorer_fn=Safety(),
                sample_rate=0.5,
                filter_string="trace.status = 'OK'",
            )

            # Add a correctness scorer with different sampling
            correctness_scorer = add_scheduled_scorer(
                experiment_id=experiment.experiment_id,  # Explicitly specify experiment
                scorer_name="correctness_monitor",
                scorer_fn=Correctness(),
                sample_rate=0.2,  # More expensive, so lower sample rate
            )

    Note:
        Once added, the scheduled scorer will begin evaluating new traces automatically.
        There may be a delay between when traces are logged and when they are evaluated.
    """
    try:
        from databricks.agents.scorers import add_scheduled_scorer
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return add_scheduled_scorer(experiment_id, scorer_name, scorer_fn, sample_rate, filter_string)


@experimental
def update_scheduled_scorer(
    *,
    experiment_id: Optional[str] = None,
    scorer_name: str,
    scorer_fn: Scorer,
    sample_rate: float,
    filter_string: Optional[str] = None,
) -> ScheduledScorer:
    """
    Update an existing scheduled scorer configuration.

    This function modifies the configuration of an existing scheduled scorer, allowing you
    to change the scorer function, sampling rate, or filter criteria. The scorer will
    continue to run automatically with the new configuration.

    Args:
        experiment_id: The ID of the MLflow experiment containing the scheduled scorer.
            If None, uses the currently active experiment.
        scorer_name: The name of the existing scheduled scorer to update. Must match
            the name used when the scorer was originally added. We recommend using the
            scorer's name (e.g., scorer_fn.name) for consistency.
        scorer_fn: The new scorer function to execute on sampled traces. Must be either
            a built-in scorer or a function decorated with @scorer.
        sample_rate: The new fraction of traces to evaluate, between 0.0 and 1.0.
        filter_string: The new MLflow search_traces compatible filter string. If None,
            all traces in the experiment are eligible for sampling.

    Returns:
        A ScheduledScorer object representing the updated scheduled scorer configuration.

    Example:
        .. code-block:: python

            from mlflow.genai.scorers import Safety
            from mlflow.genai.scheduled_scorers import update_scheduled_scorer

            # Update an existing safety scorer to increase sampling rate
            updated_scorer = update_scheduled_scorer(
                scorer_name="safety_monitor",
                scorer_fn=Safety(),
                sample_rate=0.8,  # Increased from 0.5 to 0.8
                filter_string="trace.status = 'OK'",
            )
    """
    try:
        from databricks.agents.scorers import update_scheduled_scorer
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return update_scheduled_scorer(
        experiment_id, scorer_name, scorer_fn, sample_rate, filter_string
    )


@experimental
def delete_scheduled_scorer(*, experiment_id: Optional[str] = None, scorer_name: str) -> None:
    """
    Delete a scheduled scorer from an MLflow experiment.

    This function removes a scheduled scorer configuration, stopping automatic evaluation
    of traces. Existing evaluation results will remain in the monitoring dashboard, but
    no new evaluations will be performed.

    Args:
        experiment_id: The ID of the MLflow experiment containing the scheduled scorer.
            If None, uses the currently active experiment.
        scorer_name: The name of the scheduled scorer to delete. Must match the name
            used when the scorer was originally added.

    Example:
        .. code-block:: python

            from mlflow.genai.scheduled_scorers import delete_scheduled_scorer

            # Remove a scheduled scorer that's no longer needed
            delete_scheduled_scorer(scorer_name="safety_monitor")

            # To delete all scheduled scorers at once, use set_scheduled_scorers
            # with an empty list instead:
            from mlflow.genai.scheduled_scorers import set_scheduled_scorers

            set_scheduled_scorers(
                scheduled_scorers=[]  # Empty list removes all scorers
            )

    Note:
        Deletion is immediate and cannot be undone. If you need the same scorer
        configuration later, you will need to add it again using add_scheduled_scorer.
    """
    try:
        from databricks.agents.scorers import delete_scheduled_scorer
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return delete_scheduled_scorer(experiment_id, scorer_name)


@experimental
def get_scheduled_scorer(
    *, experiment_id: Optional[str] = None, scorer_name: str
) -> ScheduledScorer:
    """
    Retrieve the configuration of a specific scheduled scorer.

    This function returns the current configuration of a scheduled scorer, including
    its scorer function, sampling rate, and filter criteria.

    Args:
        experiment_id: The ID of the MLflow experiment containing the scheduled scorer.
            If None, uses the currently active experiment.
        scorer_name: The name of the scheduled scorer to retrieve.

    Returns:
        A ScheduledScorer object containing the current configuration of the specified
        scheduled scorer.

    Example:
        .. code-block:: python

            from mlflow.genai.scheduled_scorers import get_scheduled_scorer

            # Get the current configuration of a scheduled scorer
            scorer_config = get_scheduled_scorer(scorer_name="safety_monitor")

            print(f"Sample rate: {scorer_config.sample_rate}")
            print(f"Filter: {scorer_config.filter_string}")
            print(f"Scorer: {scorer_config.scorer_fn.name}")
    """
    try:
        from databricks.agents.scorers import get_scheduled_scorer
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return get_scheduled_scorer(experiment_id, scorer_name)


@experimental
def list_scheduled_scorers(*, experiment_id: Optional[str] = None) -> list[ScheduledScorer]:
    """
    List all scheduled scorers for an experiment.

    This function returns all scheduled scorers configured for the specified experiment,
    or for the current active experiment if no experiment ID is provided.

    Args:
        experiment_id: The ID of the MLflow experiment to list scheduled scorers for.
            If None, uses the currently active experiment.

    Returns:
        A list of ScheduledScorer objects representing all scheduled scorers configured
        for the specified experiment.

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.scheduled_scorers import list_scheduled_scorers

            # List scorers for a specific experiment
            scorers = list_scheduled_scorers(experiment_id="12345")
            for scorer in scorers:
                print(f"Scorer: {scorer.scorer_name}")
                print(f"Sample rate: {scorer.sample_rate}")
                print(f"Filter: {scorer.filter_string}")

            # List scorers for the current active experiment
            mlflow.set_experiment("my_genai_app_monitoring")
            current_scorers = list_scheduled_scorers()
            print(f"Found {len(current_scorers)} scheduled scorers")
    """
    try:
        from databricks.agents.scorers import list_scheduled_scorers
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return list_scheduled_scorers(experiment_id)


@experimental
def set_scheduled_scorers(
    *, experiment_id: Optional[str] = None, scheduled_scorers: list[ScheduledScorer]
) -> None:
    """
    Replace all scheduled scorers for an experiment with the provided list.

    This function removes all existing scheduled scorers for the specified experiment
    and replaces them with the new list. This is useful for batch configuration updates
    or when you want to ensure only specific scorers are active.

    Args:
        experiment_id: The ID of the MLflow experiment to configure. If None, uses the
            currently active experiment.
        scheduled_scorers: A list of ScheduledScorer objects to set as the complete
            set of scheduled scorers for the experiment. Any existing scheduled scorers
            not in this list will be removed.

    Example:
        .. code-block:: python

            from mlflow.genai.scorers import Safety, Correctness, RelevanceToQuery
            from mlflow.genai.scheduled_scorers import ScheduledScorer, set_scheduled_scorers

            # Define a complete monitoring configuration
            monitoring_config = [
                ScheduledScorer(
                    scorer_fn=Safety(),
                    scorer_name="safety_check",
                    sample_rate=1.0,  # Check all traces for safety
                ),
                ScheduledScorer(
                    scorer_fn=Correctness(),
                    scorer_name="correctness_check",
                    sample_rate=0.2,  # Sample 20% for correctness (more expensive)
                    filter_string="trace.status = 'OK'",
                ),
                ScheduledScorer(
                    scorer_fn=RelevanceToQuery(),
                    scorer_name="relevance_check",
                    sample_rate=0.5,  # Sample 50% for relevance
                ),
            ]

            # Apply this configuration, replacing any existing scorers
            set_scheduled_scorers(scheduled_scorers=monitoring_config)

    Warning:
        This function will remove all existing scheduled scorers for the experiment
        that are not included in the provided list. Use add_scheduled_scorer() if you
        want to add scorers without affecting existing ones.

    Note:
        Changes may take a few minutes to take effect in the monitoring system.
        Existing evaluation results will remain in the monitoring dashboard.
    """
    try:
        from databricks.agents.scorers import set_scheduled_scorers
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return set_scheduled_scorers(experiment_id, scheduled_scorers)
