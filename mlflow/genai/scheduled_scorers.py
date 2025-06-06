from dataclasses import dataclass
from typing import Optional

from mlflow.genai.scorers.base import Scorer
from mlflow.utils.annotations import experimental

_ERROR_MSG = (
    "The `databricks-agents` package is required to use `mlflow.genai.scheduled_scorers`. "
    "Please install it with `pip install databricks-agents`."
)


@experimental
@dataclass()
class ScorerScheduleConfig:
    """
    A scheduled scorer configuration for automated monitoring of generative AI applications.

    Scheduled scorers are used to automatically evaluate traces logged to MLflow experiments
    by production applications. They are part of [Databricks Lakehouse Monitoring for GenAI](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/monitoring),
    which helps track quality metrics like groundedness, safety, and guideline adherence
    alongside operational metrics like volume, latency, and cost.

    When configured, scheduled scorers run automatically in the background to evaluate
    a sample of traces based on the specified sampling rate and filter criteria. The
    Assessments are displayed in the Traces tab of the MLflow experiment and can be used to
    identify quality issues in production.

    Args:
        scorer: The scorer function to run on sampled traces. Must be either a built-in
            scorer (e.g., Safety, Correctness) or a function decorated with @scorer.
            Subclasses of Scorer are not supported.
        scheduled_scorer_name: The name for this scheduled scorer configuration
            within the experiment. This name must be unique among all scheduled scorers
            in the same experiment.
            We recommend using the scorer's name (e.g., scorer.name) for consistency.
        sample_rate: The fraction of traces to evaluate, between 0.0 and 1.0. For example,
            0.1 means 10% of traces will be randomly selected for evaluation.
        filter_string: An optional MLflow search_traces compatible filter string to apply
            before sampling traces. Only traces matching this filter will be considered
            for evaluation. Uses the same syntax as mlflow.search_traces().

    Example:
        .. code-block:: python

            from mlflow.genai.scorers import Safety, scorer
            from mlflow.genai.scheduled_scorers import ScorerScheduleConfig

            # Using a built-in scorer
            safety_config = ScorerScheduleConfig(
                scorer=Safety(),
                scheduled_scorer_name="production_safety",
                sample_rate=0.2,  # Evaluate 20% of traces
                filter_string="trace.status = 'OK'",
            )


            # Using a custom scorer
            @scorer
            def response_length(outputs):
                return len(str(outputs)) > 100


            length_config = ScorerScheduleConfig(
                scorer=response_length,
                scheduled_scorer_name="adequate_length",
                sample_rate=0.1,  # Evaluate 10% of traces
                filter_string="trace.status = 'OK'",
            )

    Note:
        Scheduled scorers are executed automatically by Databricks and do not need to be
        manually triggered. The Assessments appear in the Traces tab of the MLflow
        experiment. Only traces logged directly to the experiment are monitored; traces
        logged to individual runs within the experiment are not evaluated.

    .. warning::
        This API is in Beta and may change or be removed in a future release without warning.
    """

    scorer: Scorer
    scheduled_scorer_name: str
    sample_rate: float
    filter_string: Optional[str] = None


# Scheduled Scorer CRUD operations
@experimental
def add_scheduled_scorer(  # clint: disable=missing-docstring-param  # noqa: D417
    *,
    scheduled_scorer_name: str,
    scorer: Scorer,
    sample_rate: float,
    filter_string: Optional[str] = None,
    experiment_id: Optional[str] = None,
    **kwargs,
) -> ScorerScheduleConfig:
    """
    Add a scheduled scorer to automatically monitor traces in an MLflow experiment.

    This function configures a scorer function to run automatically on traces logged to the
    specified experiment. The scorer will evaluate a sample of traces based on the sampling rate
    and any filter criteria. Assessments are displayed in the Traces tab of the MLflow experiment.

    Args:
        scheduled_scorer_name: The name for this scheduled scorer within the experiment.
            We recommend using the scorer's name (e.g., scorer.name) for consistency.
        scorer: The scorer function to execute on sampled traces. Must be either a
            built-in scorer or a function decorated with @scorer. Subclasses of Scorer
            are not supported.
        sample_rate: The fraction of traces to evaluate, between 0.0 and 1.0. For example,
            0.3 means 30% of traces will be randomly selected for evaluation.
        filter_string: An optional MLflow search_traces compatible filter string. Only
            traces matching this filter will be considered for evaluation. If None,
            all traces in the experiment are eligible for sampling.
        experiment_id: The ID of the MLflow experiment to monitor. If None, uses the
            currently active experiment.

    Returns:
        A ScorerScheduleConfig object representing the configured scheduled scorer.

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.scorers import Safety, Correctness
            from mlflow.genai.scheduled_scorers import add_scheduled_scorer

            # Set up your experiment
            experiment = mlflow.set_experiment("my_genai_app_monitoring")

            # Add a safety scorer to monitor 50% of traces
            safety_scorer = add_scheduled_scorer(
                scheduled_scorer_name="safety_monitor",
                scorer=Safety(),
                sample_rate=0.5,
                filter_string="trace.status = 'OK'",
            )

            # Add a correctness scorer with different sampling
            correctness_scorer = add_scheduled_scorer(
                scheduled_scorer_name="correctness_monitor",
                scorer=Correctness(),
                sample_rate=0.2,  # More expensive, so lower sample rate
                experiment_id=experiment.experiment_id,  # Explicitly specify experiment
            )

    Note:
        Once added, the scheduled scorer will begin evaluating new traces automatically.
        There may be a delay between when traces are logged and when they are evaluated.
        Only traces logged directly to the experiment are monitored; traces logged to
        individual runs within the experiment are not evaluated.

    .. warning::
        This API is in Beta and may change or be removed in a future release without warning.
    """
    try:
        from databricks.agents.scorers import add_scheduled_scorer
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return add_scheduled_scorer(
        experiment_id, scheduled_scorer_name, scorer, sample_rate, filter_string, **kwargs
    )


@experimental
def update_scheduled_scorer(  # clint: disable=missing-docstring-param  # noqa: D417
    *,
    scheduled_scorer_name: str,
    scorer: Optional[Scorer] = None,
    sample_rate: Optional[float] = None,
    filter_string: Optional[str] = None,
    experiment_id: Optional[str] = None,
    **kwargs,
) -> ScorerScheduleConfig:
    """
    Update an existing scheduled scorer configuration.

    This function modifies the configuration of an existing scheduled scorer, allowing you
    to change the scorer function, sampling rate, or filter criteria. Only the provided
    parameters will be updated; omitted parameters will retain their current values.
    The scorer will continue to run automatically with the new configuration.

    Args:
        scheduled_scorer_name: The name of the existing scheduled scorer to update. Must match
            the name used when the scorer was originally added. We recommend using the
            scorer's name (e.g., scorer.name) for consistency.
        scorer: The new scorer function to execute on sampled traces. Must be either
            a built-in scorer or a function decorated with @scorer. If None, the
            current scorer function will be retained.
        sample_rate: The new fraction of traces to evaluate, between 0.0 and 1.0.
            If None, the current sample rate will be retained.
        filter_string: The new MLflow search_traces compatible filter string. If None,
            the current filter string will be retained. Pass an empty string to remove
            the filter entirely.
        experiment_id: The ID of the MLflow experiment containing the scheduled scorer.
            If None, uses the currently active experiment.

    Returns:
        A ScorerScheduleConfig object representing the updated scheduled scorer configuration.

    Example:
        .. code-block:: python

            from mlflow.genai.scorers import Safety
            from mlflow.genai.scheduled_scorers import update_scheduled_scorer

            # Update an existing safety scorer to increase sampling rate
            updated_scorer = update_scheduled_scorer(
                scheduled_scorer_name="safety_monitor",
                sample_rate=0.8,  # Increased from 0.5 to 0.8
            )

    .. warning::
        This API is in Beta and may change or be removed in a future release without warning.
    """
    try:
        from databricks.agents.scorers import update_scheduled_scorer
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return update_scheduled_scorer(
        experiment_id, scheduled_scorer_name, scorer, sample_rate, filter_string, **kwargs
    )


@experimental
def delete_scheduled_scorer(  # clint: disable=missing-docstring-param  # noqa: D417
    *,
    scheduled_scorer_name: str,
    experiment_id: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Delete a scheduled scorer from an MLflow experiment.

    This function removes a scheduled scorer configuration, stopping automatic evaluation
    of traces. Existing Assessments will remain in the Traces tab of the MLflow
    experiment, but no new evaluations will be performed.

    Args:
        scheduled_scorer_name: The name of the scheduled scorer to delete. Must match the name
            used when the scorer was originally added.
        experiment_id: The ID of the MLflow experiment containing the scheduled scorer.
            If None, uses the currently active experiment.

    Example:
        .. code-block:: python

            from mlflow.genai.scheduled_scorers import delete_scheduled_scorer

            # Remove a scheduled scorer that's no longer needed
            delete_scheduled_scorer(scheduled_scorer_name="safety_monitor")

            # To delete all scheduled scorers at once, use set_scheduled_scorers
            # with an empty list instead:
            from mlflow.genai.scheduled_scorers import set_scheduled_scorers

            set_scheduled_scorers(
                scheduled_scorers=[]  # Empty list removes all scorers
            )

    Note:
        Deletion is immediate and cannot be undone. If you need the same scorer
        configuration later, you will need to add it again using add_scheduled_scorer.

    .. warning::
        This API is in Beta and may change or be removed in a future release without warning.
    """
    try:
        from databricks.agents.scorers import delete_scheduled_scorer
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return delete_scheduled_scorer(experiment_id, scheduled_scorer_name, **kwargs)


@experimental
def get_scheduled_scorer(  # clint: disable=missing-docstring-param  # noqa: D417
    *,
    scheduled_scorer_name: str,
    experiment_id: Optional[str] = None,
    **kwargs,
) -> ScorerScheduleConfig:
    """
    Retrieve the configuration of a specific scheduled scorer.

    This function returns the current configuration of a scheduled scorer, including
    its scorer function, sampling rate, and filter criteria.

    Args:
        scheduled_scorer_name: The name of the scheduled scorer to retrieve.
        experiment_id: The ID of the MLflow experiment containing the scheduled scorer.
            If None, uses the currently active experiment.

    Returns:
        A ScorerScheduleConfig object containing the current configuration of the specified
        scheduled scorer.

    Example:
        .. code-block:: python

            from mlflow.genai.scheduled_scorers import get_scheduled_scorer

            # Get the current configuration of a scheduled scorer
            scorer_config = get_scheduled_scorer(scheduled_scorer_name="safety_monitor")

            print(f"Sample rate: {scorer_config.sample_rate}")
            print(f"Filter: {scorer_config.filter_string}")
            print(f"Scorer: {scorer_config.scorer.name}")

    .. warning::
        This API is in Beta and may change or be removed in a future release without warning.
    """
    try:
        from databricks.agents.scorers import get_scheduled_scorer
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return get_scheduled_scorer(experiment_id, scheduled_scorer_name, **kwargs)


@experimental
def list_scheduled_scorers(  # clint: disable=missing-docstring-param  # noqa: D417
    *, experiment_id: Optional[str] = None, **kwargs
) -> list[ScorerScheduleConfig]:
    """
    List all scheduled scorers for an experiment.

    This function returns all scheduled scorers configured for the specified experiment,
    or for the current active experiment if no experiment ID is provided.

    Args:
        experiment_id: The ID of the MLflow experiment to list scheduled scorers for.
            If None, uses the currently active experiment.

    Returns:
        A list of ScheduledScorerConfig objects representing all scheduled scorers configured
        for the specified experiment.

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.scheduled_scorers import list_scheduled_scorers

            # List scorers for a specific experiment
            scorers = list_scheduled_scorers(experiment_id="12345")
            for scorer in scorers:
                print(f"Scorer: {scorer.scheduled_scorer_name}")
                print(f"Sample rate: {scorer.sample_rate}")
                print(f"Filter: {scorer.filter_string}")

            # List scorers for the current active experiment
            mlflow.set_experiment("my_genai_app_monitoring")
            current_scorers = list_scheduled_scorers()
            print(f"Found {len(current_scorers)} scheduled scorers")

    .. warning::
        This API is in Beta and may change or be removed in a future release without warning.
    """
    try:
        from databricks.agents.scorers import list_scheduled_scorers
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return list_scheduled_scorers(experiment_id, **kwargs)


@experimental
def set_scheduled_scorers(  # clint: disable=missing-docstring-param  # noqa: D417
    *,
    scheduled_scorers: list[ScorerScheduleConfig],
    experiment_id: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Replace all scheduled scorers for an experiment with the provided list.

    This function removes all existing scheduled scorers for the specified experiment
    and replaces them with the new list. This is useful for batch configuration updates
    or when you want to ensure only specific scorers are active.

    Args:
        scheduled_scorers: A list of ScheduledScorerConfig objects to set as the complete
            set of scheduled scorers for the experiment. Any existing scheduled scorers
            not in this list will be removed.
        experiment_id: The ID of the MLflow experiment to configure. If None, uses the
            currently active experiment.

    Example:
        .. code-block:: python

            from mlflow.genai.scorers import Safety, Correctness, RelevanceToQuery
            from mlflow.genai.scheduled_scorers import ScorerScheduleConfig, set_scheduled_scorers

            # Define a complete monitoring configuration
            monitoring_config = [
                ScorerScheduleConfig(
                    scorer=Safety(),
                    scheduled_scorer_name="safety_check",
                    sample_rate=1.0,  # Check all traces for safety
                ),
                ScorerScheduleConfig(
                    scorer=Correctness(),
                    scheduled_scorer_name="correctness_check",
                    sample_rate=0.2,  # Sample 20% for correctness (more expensive)
                    filter_string="trace.status = 'OK'",
                ),
                ScorerScheduleConfig(
                    scorer=RelevanceToQuery(),
                    scheduled_scorer_name="relevance_check",
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
        Existing Assessments will remain in the Traces tab of the MLflow experiment.

    .. warning::
        This API is in Beta and may change or be removed in a future release without warning.
    """
    try:
        from databricks.agents.scorers import set_scheduled_scorers
    except ImportError as e:
        raise ImportError(_ERROR_MSG) from e
    return set_scheduled_scorers(experiment_id, scheduled_scorers, **kwargs)
