from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.genai_metric import make_genai_metric
from mlflow.metrics.genai.utils import _MIGRATION_GUIDE, _get_latest_metric_version
from mlflow.models import EvaluationMetric
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import deprecated
from mlflow.utils.class_utils import _get_class_from_string


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def answer_similarity(
    model: str | None = None,
    metric_version: str | None = None,
    examples: list[EvaluationExample] | None = None,
    metric_metadata: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
    max_workers: int = 10,
) -> EvaluationMetric:
    """
    This function will create a genai metric used to evaluate the answer similarity of an LLM
    using the model provided. Answer similarity will be assessed by the semantic similarity of the
    output to the ``ground_truth``, which should be specified in the ``targets`` column. High
    scores mean that your model outputs contain similar information as the ground_truth, while
    low scores mean that outputs may disagree with the ground_truth.

    The ``targets`` eval_arg must be provided as part of the input dataset or output
    predictions. This can be mapped to a column of a different name using ``col_mapping``
    in the ``evaluator_config`` parameter, or using the ``targets`` parameter in mlflow.evaluate().

    An MlflowException will be raised if the specified version for this metric does not exist.

    Args:
        model: (Optional) Model uri of the judge model that will be used to compute the metric,
            e.g., ``openai:/gpt-4``. Refer to the `LLM-as-a-Judge Metrics <https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#selecting-the-llm-as-judge-model>`_
            documentation for the supported model types and their URI format.
        metric_version: (Optional) The version of the answer similarity metric to use.
            Defaults to the latest version.
        examples: (Optional) Provide a list of examples to help the judge model evaluate the
            answer similarity. It is highly recommended to add examples to be used as a reference to
            evaluate the new results.
        metric_metadata: (Optional) Dictionary of metadata to be attached to the
            EvaluationMetric object. Useful for model evaluators that require additional
            information to determine how to evaluate this metric.
        parameters: (Optional) Dictionary of parameters to be passed to the judge model,
            e.g., {"temperature": 0.5}. When specified, these parameters will override
            the default parameters defined in the metric implementation.
        extra_headers: (Optional) Dictionary of extra headers to be passed to the judge model.
        proxy_url: (Optional) Proxy URL to be used for the judge model. This is useful when the
            judge model is served via a proxy endpoint, not directly via LLM provider services.
            If not specified, the default URL for the LLM provider will be used
            (e.g., https://api.openai.com/v1/chat/completions for OpenAI chat models).
        max_workers: (Optional) The maximum number of workers to use for judge scoring.
            Defaults to 10 workers.

    Returns:
        A metric object
    """
    if metric_version is None:
        metric_version = _get_latest_metric_version()
    class_name = f"mlflow.metrics.genai.prompts.{metric_version}.AnswerSimilarityMetric"
    try:
        answer_similarity_class_module = _get_class_from_string(class_name)
    except ModuleNotFoundError:
        raise MlflowException(
            f"Failed to find answer similarity metric for version {metric_version}."
            f" Please check the version",
            error_code=INVALID_PARAMETER_VALUE,
        ) from None
    except Exception as e:
        raise MlflowException(
            f"Failed to construct answer similarity metric {metric_version}. Error: {e!r}",
            error_code=INTERNAL_ERROR,
        ) from None

    if examples is None:
        examples = answer_similarity_class_module.default_examples
    if model is None:
        model = answer_similarity_class_module.default_model

    return make_genai_metric(
        name="answer_similarity",
        definition=answer_similarity_class_module.definition,
        grading_prompt=answer_similarity_class_module.grading_prompt,
        include_input=False,
        examples=examples,
        version=metric_version,
        model=model,
        grading_context_columns=answer_similarity_class_module.grading_context_columns,
        parameters=parameters or answer_similarity_class_module.parameters,
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
        metric_metadata=metric_metadata,
        extra_headers=extra_headers,
        proxy_url=proxy_url,
        max_workers=max_workers,
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def answer_correctness(
    model: str | None = None,
    metric_version: str | None = None,
    examples: list[EvaluationExample] | None = None,
    metric_metadata: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
    max_workers: int = 10,
) -> EvaluationMetric:
    """
    This function will create a genai metric used to evaluate the answer correctness of an LLM
    using the model provided. Answer correctness will be assessed by the accuracy of the provided
    output based on the ``ground_truth``, which should be specified in the ``targets`` column.
    High scores mean that your model outputs contain similar information as the ground_truth and
    that this information is correct, while low scores mean that outputs may disagree with the
    ground_truth or that the information in the output is incorrect. Note that this builds onto
    answer_similarity.

    The ``targets`` eval_arg must be provided as part of the input dataset or output
    predictions. This can be mapped to a column of a different name using ``col_mapping``
    in the ``evaluator_config`` parameter, or using the ``targets`` parameter in mlflow.evaluate().

    An MlflowException will be raised if the specified version for this metric does not exist.

    Args:
        model: (Optional) Model uri of the judge model that will be used to compute the metric,
            e.g., ``openai:/gpt-4``. Refer to the `LLM-as-a-Judge Metrics <https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#selecting-the-llm-as-judge-model>`_
            documentation for the supported model types and their URI format.
        metric_version: The version of the answer correctness metric to use.
            Defaults to the latest version.
        examples: Provide a list of examples to help the judge model evaluate the
            answer correctness. It is highly recommended to add examples to be used as a reference
            to evaluate the new results.
        metric_metadata: (Optional) Dictionary of metadata to be attached to the
            EvaluationMetric object. Useful for model evaluators that require additional
            information to determine how to evaluate this metric.
        parameters: (Optional) Dictionary of parameters to be passed to the judge model,
            e.g., {"temperature": 0.5}. When specified, these parameters will override
            the default parameters defined in the metric implementation.
        extra_headers: (Optional) Dictionary of extra headers to be passed to the judge model.
        proxy_url: (Optional) Proxy URL to be used for the judge model. This is useful when the
            judge model is served via a proxy endpoint, not directly via LLM provider services.
            If not specified, the default URL for the LLM provider will be used
            (e.g., https://api.openai.com/v1/chat/completions for OpenAI chat models).
        max_workers: (Optional) The maximum number of workers to use for judge scoring.
            Defaults to 10 workers.

    Returns:
        A metric object
    """
    if metric_version is None:
        metric_version = _get_latest_metric_version()
    class_name = f"mlflow.metrics.genai.prompts.{metric_version}.AnswerCorrectnessMetric"
    try:
        answer_correctness_class_module = _get_class_from_string(class_name)
    except ModuleNotFoundError:
        raise MlflowException(
            f"Failed to find answer correctness metric for version {metric_version}."
            f"Please check the version",
            error_code=INVALID_PARAMETER_VALUE,
        ) from None
    except Exception as e:
        raise MlflowException(
            f"Failed to construct answer correctness metric {metric_version}. Error: {e!r}",
            error_code=INTERNAL_ERROR,
        ) from None

    if examples is None:
        examples = answer_correctness_class_module.default_examples
    if model is None:
        model = answer_correctness_class_module.default_model

    return make_genai_metric(
        name="answer_correctness",
        definition=answer_correctness_class_module.definition,
        grading_prompt=answer_correctness_class_module.grading_prompt,
        examples=examples,
        version=metric_version,
        model=model,
        grading_context_columns=answer_correctness_class_module.grading_context_columns,
        parameters=parameters or answer_correctness_class_module.parameters,
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
        metric_metadata=metric_metadata,
        extra_headers=extra_headers,
        proxy_url=proxy_url,
        max_workers=max_workers,
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def faithfulness(
    model: str | None = None,
    metric_version: str | None = _get_latest_metric_version(),
    examples: list[EvaluationExample] | None = None,
    metric_metadata: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
    max_workers: int = 10,
) -> EvaluationMetric:
    """
    This function will create a genai metric used to evaluate the faithfullness of an LLM using the
    model provided. Faithfulness will be assessed based on how factually consistent the output
    is to the ``context``. High scores mean that the outputs contain information that is in
    line with the context, while low scores mean that outputs may disagree with the context
    (input is ignored).

    The ``context`` eval_arg must be provided as part of the input dataset or output
    predictions. This can be mapped to a column of a different name using ``col_mapping``
    in the ``evaluator_config`` parameter.

    An MlflowException will be raised if the specified version for this metric does not exist.

    Args:
        model: (Optional) Model uri of the judge model that will be used to compute the metric,
            e.g., ``openai:/gpt-4``. Refer to the `LLM-as-a-Judge Metrics <https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#selecting-the-llm-as-judge-model>`_
            documentation for the supported model types and their URI format.
        metric_version: The version of the faithfulness metric to use.
            Defaults to the latest version.
        examples: Provide a list of examples to help the judge model evaluate the
            faithfulness. It is highly recommended to add examples to be used as a reference to
            evaluate the new results.
        metric_metadata: (Optional) Dictionary of metadata to be attached to the
            EvaluationMetric object. Useful for model evaluators that require additional
            information to determine how to evaluate this metric.
        parameters: (Optional) Dictionary of parameters to be passed to the judge model,
            e.g., {"temperature": 0.5}. When specified, these parameters will override
            the default parameters defined in the metric implementation.
        extra_headers: (Optional) Dictionary of extra headers to be passed to the judge model.
        proxy_url: (Optional) Proxy URL to be used for the judge model. This is useful when the
            judge model is served via a proxy endpoint, not directly via LLM provider services.
            If not specified, the default URL for the LLM provider will be used
            (e.g., https://api.openai.com/v1/chat/completions for OpenAI chat models).
        max_workers: (Optional) The maximum number of workers to use for judge scoring.
            Defaults to 10 workers.

    Returns:
        A metric object
    """
    class_name = f"mlflow.metrics.genai.prompts.{metric_version}.FaithfulnessMetric"
    try:
        faithfulness_class_module = _get_class_from_string(class_name)
    except ModuleNotFoundError:
        raise MlflowException(
            f"Failed to find faithfulness metric for version {metric_version}."
            f" Please check the version",
            error_code=INVALID_PARAMETER_VALUE,
        ) from None
    except Exception as e:
        raise MlflowException(
            f"Failed to construct faithfulness metric {metric_version}. Error: {e!r}",
            error_code=INTERNAL_ERROR,
        ) from None

    if examples is None:
        examples = faithfulness_class_module.default_examples
    if model is None:
        model = faithfulness_class_module.default_model

    return make_genai_metric(
        name="faithfulness",
        definition=faithfulness_class_module.definition,
        grading_prompt=faithfulness_class_module.grading_prompt,
        include_input=False,
        examples=examples,
        version=metric_version,
        model=model,
        grading_context_columns=faithfulness_class_module.grading_context_columns,
        parameters=parameters or faithfulness_class_module.parameters,
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
        metric_metadata=metric_metadata,
        extra_headers=extra_headers,
        proxy_url=proxy_url,
        max_workers=max_workers,
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def answer_relevance(
    model: str | None = None,
    metric_version: str | None = _get_latest_metric_version(),
    examples: list[EvaluationExample] | None = None,
    metric_metadata: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
    max_workers: int = 10,
) -> EvaluationMetric:
    """
    This function will create a genai metric used to evaluate the answer relevance of an LLM
    using the model provided. Answer relevance will be assessed based on the appropriateness and
    applicability of the output with respect to the input. High scores mean that your model
    outputs are about the same subject as the input, while low scores mean that outputs may
    be non-topical.

    An MlflowException will be raised if the specified version for this metric does not exist.

    Args:
        model: (Optional) Model uri of the judge model that will be used to compute the metric,
            e.g., ``openai:/gpt-4``. Refer to the `LLM-as-a-Judge Metrics <https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#selecting-the-llm-as-judge-model>`_
            documentation for the supported model types and their URI format.
        metric_version: The version of the answer relevance metric to use.
            Defaults to the latest version.
        examples: Provide a list of examples to help the judge model evaluate the
            answer relevance. It is highly recommended to add examples to be used as a reference to
            evaluate the new results.
        metric_metadata: (Optional) Dictionary of metadata to be attached to the
            EvaluationMetric object. Useful for model evaluators that require additional
            information to determine how to evaluate this metric.
        parameters: (Optional) Dictionary of parameters to be passed to the judge model,
            e.g., {"temperature": 0.5}. When specified, these parameters will override
            the default parameters defined in the metric implementation.
        extra_headers: (Optional) Dictionary of extra headers to be passed to the judge model.
        proxy_url: (Optional) Proxy URL to be used for the judge model. This is useful when the
            judge model is served via a proxy endpoint, not directly via LLM provider services.
            If not specified, the default URL for the LLM provider will be used
            (e.g., https://api.openai.com/v1/chat/completions for OpenAI chat models).
        max_workers: (Optional) The maximum number of workers to use for judge scoring.
            Defaults to 10 workers.

    Returns:
        A metric object
    """
    class_name = f"mlflow.metrics.genai.prompts.{metric_version}.AnswerRelevanceMetric"
    try:
        answer_relevance_class_module = _get_class_from_string(class_name)
    except ModuleNotFoundError:
        raise MlflowException(
            f"Failed to find answer relevance metric for version {metric_version}."
            f" Please check the version",
            error_code=INVALID_PARAMETER_VALUE,
        ) from None
    except Exception as e:
        raise MlflowException(
            f"Failed to construct answer relevance metric {metric_version}. Error: {e!r}",
            error_code=INTERNAL_ERROR,
        ) from None

    if examples is None:
        examples = answer_relevance_class_module.default_examples
    if model is None:
        model = answer_relevance_class_module.default_model

    return make_genai_metric(
        name="answer_relevance",
        definition=answer_relevance_class_module.definition,
        grading_prompt=answer_relevance_class_module.grading_prompt,
        examples=examples,
        version=metric_version,
        model=model,
        parameters=parameters or answer_relevance_class_module.parameters,
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
        metric_metadata=metric_metadata,
        extra_headers=extra_headers,
        proxy_url=proxy_url,
        max_workers=max_workers,
    )


@deprecated(since="3.4.0", impact=_MIGRATION_GUIDE)
def relevance(
    model: str | None = None,
    metric_version: str | None = None,
    examples: list[EvaluationExample] | None = None,
    metric_metadata: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
    max_workers: int = 10,
) -> EvaluationMetric:
    """
    This function will create a genai metric used to evaluate the evaluate the relevance of an
    LLM using the model provided. Relevance will be assessed by the appropriateness, significance,
    and applicability of the output with respect to the input and ``context``. High scores mean
    that the model has understood the context and correct extracted relevant information from
    the context, while low score mean that output has completely ignored the question and the
    context and could be hallucinating.

    The ``context`` eval_arg must be provided as part of the input dataset or output
    predictions. This can be mapped to a column of a different name using ``col_mapping``
    in the ``evaluator_config`` parameter.

    An MlflowException will be raised if the specified version for this metric does not exist.

    Args:
        model: (Optional) Model uri of the judge model that will be used to compute the metric,
            e.g., ``openai:/gpt-4``. Refer to the `LLM-as-a-Judge Metrics <https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#selecting-the-llm-as-judge-model>`_
            documentation for the supported model types and their URI format.
        metric_version: (Optional) The version of the relevance metric to use.
            Defaults to the latest version.
        examples: (Optional) Provide a list of examples to help the judge model evaluate the
            relevance. It is highly recommended to add examples to be used as a reference to
            evaluate the new results.
        metric_metadata: (Optional) Dictionary of metadata to be attached to the
            EvaluationMetric object. Useful for model evaluators that require additional
            information to determine how to evaluate this metric.
        parameters: (Optional) Dictionary of parameters to be passed to the judge model,
            e.g., {"temperature": 0.5}. When specified, these parameters will override
            the default parameters defined in the metric implementation.
        extra_headers: (Optional) Dictionary of extra headers to be passed to the judge model.
        proxy_url: (Optional) Proxy URL to be used for the judge model. This is useful when the
            judge model is served via a proxy endpoint, not directly via LLM provider services.
            If not specified, the default URL for the LLM provider will be used
            (e.g., https://api.openai.com/v1/chat/completions for OpenAI chat models).
        max_workers: (Optional) The maximum number of workers to use for judge scoring.
            Defaults to 10 workers.

    Returns:
        A metric object
    """
    if metric_version is None:
        metric_version = _get_latest_metric_version()
    class_name = f"mlflow.metrics.genai.prompts.{metric_version}.RelevanceMetric"
    try:
        relevance_class_module = _get_class_from_string(class_name)
    except ModuleNotFoundError:
        raise MlflowException(
            f"Failed to find relevance metric for version {metric_version}."
            f"Please check the version",
            error_code=INVALID_PARAMETER_VALUE,
        ) from None
    except Exception as e:
        raise MlflowException(
            f"Failed to construct relevance metric {metric_version}. Error: {e!r}",
            error_code=INTERNAL_ERROR,
        ) from None

    if examples is None:
        examples = relevance_class_module.default_examples
    if model is None:
        model = relevance_class_module.default_model

    return make_genai_metric(
        name="relevance",
        definition=relevance_class_module.definition,
        grading_prompt=relevance_class_module.grading_prompt,
        examples=examples,
        version=metric_version,
        model=model,
        grading_context_columns=relevance_class_module.grading_context_columns,
        parameters=parameters or relevance_class_module.parameters,
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
        metric_metadata=metric_metadata,
        extra_headers=extra_headers,
        proxy_url=proxy_url,
        max_workers=max_workers,
    )
