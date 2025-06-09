from functools import wraps
from typing import Any, Callable, Optional, Union

from mlflow.entities.assessment import Feedback
from mlflow.genai.utils.enum_utils import StrEnum
from mlflow.utils.annotations import experimental

# NB: User-facing name for the is_context_relevant assessment.
_IS_CONTEXT_RELEVANT_ASSESSMENT_NAME = "relevance_to_context"


class CategoricalRating(StrEnum):
    """
    A categorical rating for an assessment.

    Example:
        .. code-block:: python

            from mlflow.genai.judges import CategoricalRating
            from mlflow.entities import Feedback

            # Create feedback with categorical rating
            feedback = Feedback(
                name="my_metric", value=CategoricalRating.YES, rationale="The metric is passing."
            )
    """

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member == value:
                return member
        return cls.UNKNOWN


def _sanitize_feedback(feedback: Feedback) -> Feedback:
    """Sanitize the feedback object from the databricks judges.

    The judge returns a CategoricalRating class defined in the databricks-agents package.
    This function converts it to our CategoricalRating definition above.

    Args:
        feedback: The Feedback object to convert.

    Returns:
        A new Feedback object with our CategoricalRating.
    """
    feedback.value = CategoricalRating(feedback.value) if feedback.value else feedback.value
    return feedback


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
def is_context_relevant(*, request: str, context: Any, name: Optional[str] = None) -> Feedback:
    """
    LLM judge determines whether the given context is relevant to the input request.

    Args:
        request: Input to the application to evaluate, user's question or query.
        context: Context to evaluate the relevance to the request.
            Supports any JSON-serializable object.
        name: Optional name for overriding the default name of the returned feedback.

    Returns:
        A :py:class:`mlflow.entities.assessment.Feedback~` object with a "yes" or "no" value
        indicating whether the context is relevant to the request.

    Example:

        The following example shows how to evaluate whether a document retrieved by a
        retriever is relevant to the user's question.

        .. code-block:: python

            from mlflow.genai.judges import is_context_relevant

            feedback = is_context_relevant(
                request="What is the capital of France?",
                context="Paris is the capital of France.",
            )
            print(feedback.value)  # "yes"

            feedback = is_context_relevant(
                request="What is the capital of France?",
                context="Paris is known for its Eiffel Tower.",
            )
            print(feedback.value)  # "no"

    """
    from databricks.agents.evals.judges import relevance_to_query

    return _sanitize_feedback(
        relevance_to_query(
            request=request,
            response=str(context),
            # NB: User-facing name for the is_context_relevant assessment. This is required since
            #     the existing databricks judge is called `relevance_to_query`
            assessment_name=name or _IS_CONTEXT_RELEVANT_ASSESSMENT_NAME,
        )
    )


@requires_databricks_agents
def is_context_sufficient(
    *,
    request: str,
    context: Any,
    expected_facts: list[str],
    expected_response: Optional[str] = None,
    name: Optional[str] = None,
) -> Feedback:
    """
    LLM judge determines whether the given context is sufficient to answer the input request.

    Args:
        request: Input to the application to evaluate, user's question or query.
        context: Context to evaluate the sufficiency of. Supports any JSON-serializable object.
        expected_facts: A list of expected facts that should be present in the context.
        expected_response: The expected response from the application. Optional.
        name: Optional name for overriding the default name of the returned feedback.

    Returns:
        A :py:class:`mlflow.entities.assessment.Feedback~` object with a "yes" or "no"
        value indicating whether the context is sufficient to answer the request.

    Example:

        The following example shows how to evaluate whether the documents returned by a
        retriever gives sufficient context to answer the user's question.

        .. code-block:: python

            from mlflow.genai.judges import is_context_sufficient

            feedback = is_context_sufficient(
                request="What is the capital of France?",
                context=[
                    {"content": "Paris is the capital of France."},
                    {"content": "Paris is known for its Eiffel Tower."},
                ],
                expected_facts=["Paris is the capital of France."],
            )
            print(feedback.value)  # "yes"
    """
    from databricks.agents.evals.judges import context_sufficiency

    return _sanitize_feedback(
        context_sufficiency(
            request=request,
            retrieved_context=context,
            expected_facts=expected_facts,
            expected_response=expected_response,
            assessment_name=name,
        )
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

    return _sanitize_feedback(
        correctness(
            request=request,
            response=response,
            expected_facts=expected_facts,
            expected_response=expected_response,
            assessment_name=name,
        )
    )


@requires_databricks_agents
def is_grounded(
    *, request: str, response: str, context: Any, name: Optional[str] = None
) -> Feedback:
    """
    LLM judge determines whether the given response is grounded in the given context.

    Args:
        request: Input to the application to evaluate, user's question or query.
        response: The response from the application to evaluate.
        context: Context to evaluate the response against. Supports any JSON-serializable object.
        name: Optional name for overriding the default name of the returned feedback.

    Returns:
        A :py:class:`mlflow.entities.assessment.Feedback~` object with a "yes" or "no"
        value indicating whether the response is grounded in the context.

    Example:

        The following example shows how to evaluate whether the response is grounded in
        the context.

        .. code-block:: python

            from mlflow.genai.judges import is_grounded

            feedback = is_grounded(
                request="What is the capital of France?",
                response="Paris",
                context=[
                    {"content": "Paris is the capital of France."},
                    {"content": "Paris is known for its Eiffel Tower."},
                ],
            )
            print(feedback.value)  # "yes"
    """
    from databricks.agents.evals.judges import groundedness

    return _sanitize_feedback(
        groundedness(
            request=request,
            response=response,
            retrieved_context=context,
            assessment_name=name,
        )
    )


@requires_databricks_agents
def is_safe(*, content: str, name: Optional[str] = None) -> Feedback:
    """
    LLM judge determines whether the given response is safe.

    Args:
        content: Text content to evaluate for safety.
        name: Optional name for overriding the default name of the returned feedback.

    Returns:
        A :py:class:`mlflow.entities.assessment.Feedback~` object with a "yes" or "no"
        value indicating whether the response is safe.

    Example:

        .. code-block:: python

            from mlflow.genai.judges import is_safe

            feedback = is_safe(content="I am a happy person.")
            print(feedback.value)  # "yes"
    """
    from databricks.agents.evals.judges import safety

    return _sanitize_feedback(safety(response=content, assessment_name=name))


@requires_databricks_agents
def meets_guidelines(
    *,
    guidelines: Union[str, list[str]],
    context: dict[str, Any],
    name: Optional[str] = None,
) -> Feedback:
    """
    LLM judge determines whether the given response meets the given guideline(s).

    Args:
        guidelines: A single guideline or a list of guidelines.
        context: Mapping of context to be evaluated against the guidelines. For example,
            pass {"response": "<response text>"} to evaluate whether the response meets
            the given guidelines.
        name: Optional name for overriding the default name of the returned feedback.

    Returns:
        A :py:class:`mlflow.entities.assessment.Feedback~` object with a "yes" or "no"
        value indicating whether the response meets the guideline(s).

    Example:

        The following example shows how to evaluate whether the response meets the given
        guideline(s).

        .. code-block:: python

            from mlflow.genai.judges import meets_guidelines

            feedback = meets_guidelines(
                guidelines="Be polite and respectful.",
                context={"response": "Hello, how are you?"},
            )
            print(feedback.value)  # "yes"

            feedback = meets_guidelines(
                guidelines=["Be polite and respectful.", "Must be in English."],
                context={"response": "Hola, ¿cómo estás?"},
            )
            print(feedback.value)  # "no"
    """
    from databricks.agents.evals.judges import guideline_adherence

    # Ensure guidelines is a list, as the underlying databricks judge only accepts lists
    if isinstance(guidelines, str):
        guidelines = [guidelines]

    return _sanitize_feedback(
        guideline_adherence(
            guidelines=guidelines,
            guidelines_context=context,
            assessment_name=name,
        )
    )


@experimental
@requires_databricks_agents
def custom_prompt_judge(
    *,
    name: str,
    prompt_template: str,
    numeric_values: Optional[dict[str, Union[int, float]]] = None,
) -> Callable[..., Feedback]:
    """
    Create a custom prompt judge that evaluates inputs using a template.

    Example prompt template:

    .. code-block::

        You will look at the response and determine the formality of the response.

        <request>{{request}}</request>
        <response>{{response}}</response>

        You must choose one of the following categories.

        [[formal]]: The response is very formal.
        [[semi_formal]]: The response is somewhat formal. The response is somewhat formal if the
        response mentions friendship, etc.
        [[not_formal]]: The response is not formal.

    Variable names in the template should be enclosed in double curly
    braces, e.g., `{{request}}`, `{{response}}`. They should be alphanumeric and can include
    underscores, but should not contain spaces or special characters.

    It is required for the prompt template to request choices as outputs, with each choice
    enclosed in square brackets. Choice names should be alphanumeric and can include
    underscores and spaces.

    Args:
        name: Name of the judge, used as the name of returned
            :py:class`mlflow.entities.Feedback~` object.
        prompt_template: Template string with {{var_name}} placeholders for variable substitution.
            Should be prompted with choices as outputs.
        numeric_values: Optional mapping from categorical values to numeric scores.
            Useful if you want to create a custom judge that returns continuous valued outputs.
            Defaults to None.

    Returns:
        A callable that takes keyword arguments mapping to the template variables
        and returns an mlflow :py:class`mlflow.entities.Feedback~`.
    """
    from databricks.agents.evals.judges import custom_prompt_judge

    return custom_prompt_judge(
        name=name,
        prompt_template=prompt_template,
        numeric_values=numeric_values,
    )
