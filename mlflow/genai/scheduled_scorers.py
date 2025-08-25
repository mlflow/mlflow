from dataclasses import dataclass

from mlflow.genai.scorers.base import Scorer
from mlflow.utils.annotations import experimental

_ERROR_MSG = (
    "The `databricks-agents` package is required to use `mlflow.genai.scheduled_scorers`. "
    "Please install it with `pip install databricks-agents`."
)


@experimental(version="3.0.0")
@dataclass()
class ScorerScheduleConfig:
    """
    A scheduled scorer configuration for automated monitoring of generative AI applications.

    Scheduled scorers are used to automatically evaluate traces logged to MLflow experiments
    by production applications. They are part of `Databricks Lakehouse Monitoring for GenAI
    <https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/monitoring>`_,
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
    filter_string: str | None = None
