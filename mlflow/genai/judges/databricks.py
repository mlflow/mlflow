from functools import wraps
from typing import Optional, Union

from mlflow.entities.assessment import Feedback


def requires_databricks_agents(func):
    """Decorator to check if the `databricks-agents` package is installed."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import databricks.agents.evals.judges  # noqa: F401

        except ImportError:
            raise ImportError(
                f"The `databricks-agents` package is required to use "
                f"`mlflow.genai.judges.{func.__name__}`. "
                "Please install it with `pip install databricks-agents`."
            )

        return func(*args, **kwargs)

    return wrapper


@requires_databricks_agents
def is_context_relevant(
    *, request: str, context: dict[str, str], name: Optional[str] = None
) -> Feedback:
    """
    LLM judge determines whether the given context is relevant to the input request.

    Args:
        request: Input to the application to evaluate, user's question or query.
        context: A single dictionary containing a "content" key with a string value.
            E.g. ``{"content": "The capital of France is Paris."}``
        name: Optional name for overriding the default name of the returned feedback.

    Returns:
        A :py:class:`mlflow.entities.assessment.Feedback~` object with a "yes" or "no" value
        indicating whether the context is relevant to the request.
    """
    from databricks.agents.evals.judges import chunk_relevance

    # NB: The `chunk_relevance` takes a list of context chunks and returns a list of feedbacks.
    return chunk_relevance(
        request=request,
        retrieved_context=[context],
        assessment_name=name,
    )[0]


@requires_databricks_agents
def is_context_sufficient(
    *,
    request: str,
    context: list[dict[str, str]],
    expected_facts: list[str],
    expected_response: Optional[str] = None,
    name: Optional[str] = None,
) -> Feedback:
    """
    LLM judge determines whether the given context is sufficient to answer the input request.

    Args:
        request: Input to the application to evaluate, user's question or query.
        context: Context to evaluate the response against. It must be a list of dictionaries,
            each containing a "content" key with a string value.
        expected_facts: A list of expected facts that should be present in the context.
        expected_response: The expected response from the application. Optional.
        name: Optional name for overriding the default name of the returned feedback.

    Returns:
        A :py:class:`mlflow.entities.assessment.Feedback~` object with a "yes" or "no"
        value indicating whether the context is sufficient to answer the request.
    """
    from databricks.agents.evals.judges import context_sufficiency

    return context_sufficiency(
        request=request,
        retrieved_context=context,
        expected_facts=expected_facts,
        expected_response=expected_response,
        assessment_name=name,
    )


@requires_databricks_agents
def is_correct(
    *,
    request: str,
    response: str,
    expected_facts: list[str],
    expected_response: Optional[str] = None,
    name: Optional[str] = None,
) -> Feedback:
    """
    LLM judge determines whether the given response is correct for the input request.

    Args:
        request: Input to the application to evaluate, user's question or query.
        response: The response from the application to evaluate.
        expected_facts: A list of expected facts that should be present in the response.
        expected_response: The expected response from the application. Optional.
        name: Optional name for overriding the default name of the returned feedback.

    Returns:
        A :py:class:`mlflow.entities.assessment.Feedback~` object with a "yes" or "no"
        value indicating whether the response is correct for the request.
    """
    from databricks.agents.evals.judges import correctness

    return correctness(
        request=request,
        response=response,
        expected_facts=expected_facts,
        expected_response=expected_response,
        assessment_name=name,
    )


@requires_databricks_agents
def is_grounded(
    *, request: str, response: str, context: list[dict[str, str]], name: Optional[str] = None
) -> Feedback:
    """
    LLM judge determines whether the given response is grounded in the given context.

    Args:
        request: Input to the application to evaluate, user's question or query.
        response: The response from the application to evaluate.
        context: Context to evaluate the response against. It must be a list of dictionaries,
            each containing a "content" key with a string value.
        name: Optional name for overriding the default name of the returned feedback.

    Returns:
        A :py:class:`mlflow.entities.assessment.Feedback~` object with a "yes" or "no"
        value indicating whether the response is grounded in the context.
    """
    from databricks.agents.evals.judges import groundedness

    return groundedness(
        request=request,
        response=response,
        retrieved_context=context,
        assessment_name=name,
    )


@requires_databricks_agents
def is_relevant_to_query(*, request: str, response: str, name: Optional[str] = None) -> Feedback:
    """
    LLM judge determines whether the given response is relevant to the input request.

    Args:
        request: Input to the application to evaluate, user's question or query.
        response: The response from the application to evaluate.
        name: Optional name for overriding the default name of the returned feedback.

    Returns:
        A :py:class:`mlflow.entities.assessment.Feedback~` object with a "yes" or "no"
        value indicating whether the response is relevant to the request.
    """
    from databricks.agents.evals.judges import relevance_to_query

    return relevance_to_query(
        request=request,
        response=response,
        assessment_name=name,
    )


@requires_databricks_agents
def is_safe(*, request: str, response: str, name: Optional[str] = None) -> Feedback:
    """
    LLM judge determines whether the given response is safe.

    Args:
        request: Input to the application to evaluate, user's question or query.
        response: The response text from the application to evaluate.
        name: Optional name for overriding the default name of the returned feedback.

    Returns:
        A :py:class:`mlflow.entities.assessment.Feedback~` object with a "yes" or "no"
        value indicating whether the response is safe.
    """
    from databricks.agents.evals.judges import safety

    return safety(
        request=request,
        response=response,
        assessment_name=name,
    )


@requires_databricks_agents
def meets_guidelines(
    *,
    request: str,
    response: str,
    guidelines: Union[str, list[str]],
    name: Optional[str] = None,
) -> Feedback:
    """
    LLM judge determines whether the given response meets the given guideline(s).

    Args:
        request: Input to the application to evaluate, user's question or query.
        response: The response from the application to evaluate.
        guidelines: A single guideline or a list of guidelines.
        name: Optional name for overriding the default name of the returned feedback.

    Returns:
        A :py:class:`mlflow.entities.assessment.Feedback~` object with a "yes" or "no"
        value indicating whether the response meets the guideline(s).
    """
    from databricks.agents.evals.judges import guideline_adherence

    return guideline_adherence(
        request=request,
        response=response,
        guidelines=guidelines,
        assessment_name=name,
    )
