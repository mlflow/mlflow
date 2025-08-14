from functools import wraps
from typing import Any

from mlflow.entities.assessment import Feedback
from mlflow.genai.judges.prompts.relevance_to_query import RELEVANCE_TO_QUERY_ASSESSMENT_NAME
from mlflow.genai.judges.utils import CategoricalRating, get_default_model, invoke_judge_model
from mlflow.utils.docstring_utils import format_docstring

_MODEL_API_DOC = {
    "model": """Judge model to use. Must be either `"databricks"` or a form of
`<provider>:/<model-name>`, such as `"openai:/gpt-4.1-mini"`,
`"anthropic:/claude-3.5-sonnet-20240620"`. MLflow natively supports
`["openai", "anthropic", "bedrock", "mistral"]`, and more providers are supported
through `LiteLLM <https://docs.litellm.ai/docs/providers>`_.
Default model depends on the tracking URI setup:

* Databricks: `databricks`
* Otherwise: `openai:/gpt-4.1-mini`.
""",
}


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


@format_docstring(_MODEL_API_DOC)
def is_context_relevant(
    *, request: str, context: Any, name: str | None = None, model: str | None = None
) -> Feedback:
    """
    LLM judge determines whether the given context is relevant to the input request.

    Args:
        request: Input to the application to evaluate, user's question or query.
        context: Context to evaluate the relevance to the request.
            Supports any JSON-serializable object.
        name: Optional name for overriding the default name of the returned feedback.
        model: {{ model }}

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
    from mlflow.genai.judges.prompts.relevance_to_query import get_prompt

    model = model or get_default_model()

    # NB: User-facing name for the is_context_relevant assessment. This is required
    #     since the existing databricks judge is called `relevance_to_query`
    assessment_name = name or RELEVANCE_TO_QUERY_ASSESSMENT_NAME

    if model == "databricks":
        from databricks.agents.evals.judges import relevance_to_query

        feedback = relevance_to_query(
            request=request,
            response=str(context),
            assessment_name=assessment_name,
        )
    else:
        prompt = get_prompt(request, str(context))
        feedback = invoke_judge_model(model, prompt, assessment_name=assessment_name)

    return _sanitize_feedback(feedback)


@format_docstring(_MODEL_API_DOC)
def is_context_sufficient(
    *,
    request: str,
    context: Any,
    expected_facts: list[str],
    expected_response: str | None = None,
    name: str | None = None,
    model: str | None = None,
) -> Feedback:
    """
    LLM judge determines whether the given context is sufficient to answer the input request.

    Args:
        request: Input to the application to evaluate, user's question or query.
        context: Context to evaluate the sufficiency of. Supports any JSON-serializable object.
        expected_facts: A list of expected facts that should be present in the context. Optional.
        expected_response: The expected response from the application. Optional.
        name: Optional name for overriding the default name of the returned feedback.
        model: {{ model }}

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

            feedback = is_context_sufficient(
                request="What is the capital of France?",
                context={"content": "France is a country in Europe."},
                expected_response="Paris is the capital of France.",
            )
            print(feedback.value)  # "no"
    """
    from mlflow.genai.judges.prompts.context_sufficiency import (
        CONTEXT_SUFFICIENCY_FEEDBACK_NAME,
        get_prompt,
    )

    model = model or get_default_model()
    assessment_name = name or CONTEXT_SUFFICIENCY_FEEDBACK_NAME

    if model == "databricks":
        from databricks.agents.evals.judges import context_sufficiency

        feedback = context_sufficiency(
            request=request,
            retrieved_context=context,
            expected_facts=expected_facts,
            expected_response=expected_response,
            assessment_name=assessment_name,
        )
    else:
        prompt = get_prompt(
            request=request,
            context=context,
            expected_response=expected_response,
            expected_facts=expected_facts,
        )
        feedback = invoke_judge_model(model, prompt, assessment_name=assessment_name)

    return _sanitize_feedback(feedback)


@format_docstring(_MODEL_API_DOC)
def is_correct(
    *,
    request: str,
    response: str,
    expected_facts: list[str] | None = None,
    expected_response: str | None = None,
    name: str | None = None,
    model: str | None = None,
) -> Feedback:
    """
    LLM judge determines whether the given response is correct for the input request.

    Args:
        request: Input to the application to evaluate, user's question or query.
        response: The response from the application to evaluate.
        expected_facts: A list of expected facts that should be present in the response. Optional.
        expected_response: The expected response from the application. Optional.
        name: Optional name for overriding the default name of the returned feedback.
        model: {{ model }}

    Returns:
        A :py:class:`mlflow.entities.assessment.Feedback~` object with a "yes" or "no"
        value indicating whether the response is correct for the request.

    Example:

        The following example shows how to evaluate whether the response is correct.

        .. code-block:: python

            from mlflow.genai.judges import is_correct

            feedback = is_correct(
                request="What is the capital of France?",
                response="Paris is the capital of France.",
                expected_response="Paris",
            )
            print(feedback.value)  # "yes"

            feedback = is_correct(
                request="What is the capital of France?",
                response="London is the capital of France.",
                expected_facts=["Paris is the capital of France"],
            )
            print(feedback.value)  # "no"
    """
    from mlflow.genai.judges.prompts.correctness import CORRECTNESS_FEEDBACK_NAME, get_prompt

    model = model or get_default_model()
    assessment_name = name or CORRECTNESS_FEEDBACK_NAME

    if model == "databricks":
        from databricks.agents.evals.judges import correctness

        feedback = correctness(
            request=request,
            response=response,
            expected_facts=expected_facts,
            expected_response=expected_response,
            assessment_name=assessment_name,
        )
    else:
        prompt = get_prompt(
            request=request,
            response=response,
            expected_response=expected_response,
            expected_facts=expected_facts,
        )
        feedback = invoke_judge_model(model, prompt, assessment_name=assessment_name)

    return _sanitize_feedback(feedback)


@format_docstring(_MODEL_API_DOC)
def is_grounded(
    *,
    request: str,
    response: str,
    context: Any,
    name: str | None = None,
    model: str | None = None,
) -> Feedback:
    """
    LLM judge determines whether the given response is grounded in the given context.

    Args:
        request: Input to the application to evaluate, user's question or query.
        response: The response from the application to evaluate.
        context: Context to evaluate the response against. Supports any JSON-serializable object.
        name: Optional name for overriding the default name of the returned feedback.
        model: {{ model }}

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

            feedback = is_grounded(
                request="What is the capital of France?",
                response="London is the capital of France.",
                context=[
                    {"content": "Paris is the capital of France."},
                    {"content": "Paris is known for its Eiffel Tower."},
                ],
            )
            print(feedback.value)  # "no"
    """
    from mlflow.genai.judges.prompts.groundedness import GROUNDEDNESS_FEEDBACK_NAME, get_prompt

    model = model or get_default_model()
    assessment_name = name or GROUNDEDNESS_FEEDBACK_NAME

    if model == "databricks":
        from databricks.agents.evals.judges import groundedness

        feedback = groundedness(
            request=request,
            response=response,
            retrieved_context=context,
            assessment_name=assessment_name,
        )
    else:
        prompt = get_prompt(
            request=request,
            response=response,
            context=context,
        )
        feedback = invoke_judge_model(model, prompt, assessment_name=assessment_name)

    return _sanitize_feedback(feedback)


@requires_databricks_agents
def is_safe(*, content: str, name: str | None = None) -> Feedback:
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


@format_docstring(_MODEL_API_DOC)
def meets_guidelines(
    *,
    guidelines: str | list[str],
    context: dict[str, Any],
    name: str | None = None,
    model: str | None = None,
) -> Feedback:
    """
    LLM judge determines whether the given response meets the given guideline(s).

    Args:
        guidelines: A single guideline or a list of guidelines.
        context: Mapping of context to be evaluated against the guidelines. For example,
            pass {"response": "<response text>"} to evaluate whether the response meets
            the given guidelines.
        name: Optional name for overriding the default name of the returned feedback.
        model: {{ model }}

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
    from mlflow.genai.judges.prompts.guidelines import GUIDELINES_FEEDBACK_NAME, get_prompt

    model = model or get_default_model()

    if model == "databricks":
        from databricks.agents.evals.judges import guidelines as guidelines_judge

        feedback = guidelines_judge(
            guidelines=guidelines,
            context=context,
            assessment_name=name,
        )
    else:
        prompt = get_prompt(guidelines, context)
        feedback = invoke_judge_model(
            model, prompt, assessment_name=name or GUIDELINES_FEEDBACK_NAME
        )

    return _sanitize_feedback(feedback)
