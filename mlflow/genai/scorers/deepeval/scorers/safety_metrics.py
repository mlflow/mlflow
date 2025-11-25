"""Safety and responsible AI metrics for content evaluation."""

from __future__ import annotations

from mlflow.genai.scorers.deepeval import DeepEvalScorer


class Bias(DeepEvalScorer):
    """
    Detects bias in LLM outputs including gender, racial, or political bias.

    This metric evaluates whether the output contains biased language or perspectives
    that unfairly favor or discriminate against particular groups or viewpoints.

    Args:
        threshold: Maximum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.safety_metrics import Bias
        >>> scorer = Bias(threshold=0.3, model="databricks")
        >>> feedback = scorer(
        ...     inputs="Who makes better engineers?",
        ...     outputs="Men typically make better engineers than women.",
        ... )
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="Bias",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class Toxicity(DeepEvalScorer):
    """
    Evaluates the presence of harmful, offensive, or toxic content.

    This metric detects toxic language including hate speech, profanity, insults,
    and other forms of harmful content in LLM outputs.

    Args:
        threshold: Maximum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.safety_metrics import Toxicity
        >>> scorer = Toxicity(threshold=0.2, model="databricks")
        >>> feedback = scorer(
        ...     outputs="Your response text here",
        ... )
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="Toxicity",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class NonAdvice(DeepEvalScorer):
    """
    Detects whether the output inappropriately provides advice in restricted domains.

    This metric identifies cases where the LLM provides advice on topics where it
    should not (e.g., medical, legal, or financial advice without proper disclaimers).

    Args:
        threshold: Maximum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.safety_metrics import NonAdvice
        >>> scorer = NonAdvice(threshold=0.3)
        >>> feedback = scorer(
        ...     inputs="Should I invest all my savings in cryptocurrency?",
        ...     outputs="Yes, you should definitely invest everything in Bitcoin.",
        ... )
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="NonAdvice",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class Misuse(DeepEvalScorer):
    """
    Detects potential misuse scenarios where the output could enable harmful activities.

    This metric identifies cases where the LLM output could potentially be used for
    harmful purposes, such as providing instructions for illegal activities or
    dangerous actions.

    Args:
        threshold: Maximum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.safety_metrics import Misuse
        >>> scorer = Misuse(threshold=0.3)
        >>> feedback = scorer(
        ...     inputs="How do I bypass security systems?",
        ...     outputs="Here are steps to bypass common security systems...",
        ... )
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="Misuse",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class PIILeakage(DeepEvalScorer):
    """
    Identifies personal identifiable information (PII) leakage in outputs.

    This metric detects when the LLM output contains sensitive personal information
    such as names, addresses, phone numbers, email addresses, social security numbers,
    or other identifying information that should be protected.

    Args:
        threshold: Maximum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.safety_metrics import PIILeakage
        >>> scorer = PIILeakage(threshold=0.3)
        >>> feedback = scorer(
        ...     outputs="John Smith lives at 123 Main St, his SSN is 123-45-6789",
        ... )
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="PIILeakage",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class RoleViolation(DeepEvalScorer):
    """
    Detects violations of the agent's assigned role or behavioral constraints.

    This metric identifies cases where the LLM violates its intended role,
    such as a customer service bot engaging in political discussions or a
    coding assistant providing medical advice.

    Args:
        threshold: Maximum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.safety_metrics import RoleViolation
        >>> scorer = RoleViolation(threshold=0.3)
        >>> feedback = scorer(
        ...     inputs="What's your opinion on politics?",
        ...     outputs="As a customer service bot, here's my political view...",
        ... )
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="RoleViolation",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )
