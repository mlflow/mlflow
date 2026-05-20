"""
Deterministic NLP scorers for model evaluation.

This module provides built-in deterministic NLP metrics that don't require LLM calls,
making them fast, reproducible, and suitable for offline evaluation.
"""

import logging
from typing import Any, Literal

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import Judge, JudgeField
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


def _ensure_string(value: Any, field_name: str) -> str:
    """Convert value to string, handling None and other types."""
    if value is None:
        raise MlflowException.invalid_parameter_value(
            f"{field_name} cannot be None for NLP scoring"
        )
    if isinstance(value, str):
        return value
    return str(value)


@experimental(version="3.13.0")
class BERTScore(Judge):
    """
    BERTScore measures semantic similarity between outputs and expectations using BERT embeddings.

    BERTScore computes precision, recall, and F1 scores by matching tokens based on their
    contextual embeddings from BERT models. This provides a more nuanced similarity measure
    than simple string matching, as it captures semantic meaning.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Args:
        name: The name of the scorer. Defaults to "bert_score".
        model_type: The BERT model to use for embeddings. Defaults to "bert-base-uncased".
            Common options include "bert-base-uncased", "roberta-base", "distilbert-base-uncased".
        lang: Language of the text. Defaults to "en" (English).
        return_hash: If True, returns the model hash for reproducibility. Defaults to False.

    Example (direct usage):

    .. code-block:: python

        from mlflow.genai.scorers import BERTScore

        scorer = BERTScore()
        feedback = scorer(
            outputs="The cat sat on the mat",
            expectations={"expected_output": "A cat was sitting on a mat"}
        )
        print(f"F1 Score: {feedback.value}")

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=[BERTScore()])

    Note:
        Requires the `bert-score` package: `pip install bert-score`
    """

    name: str = "bert_score"
    model_type: str = "bert-base-uncased"
    lang: str = "en"
    return_hash: bool = False
    description: str = "Semantic similarity using BERT embeddings"

    @property
    def instructions(self) -> str:
        return "Evaluates semantic similarity between outputs and expected outputs using BERT embeddings."

    @property
    def feedback_value_type(self) -> Any:
        return float

    def get_input_fields(self) -> list[JudgeField]:
        return [
            JudgeField(
                name="outputs",
                description="The generated text to evaluate",
            ),
            JudgeField(
                name="expectations",
                description="Dictionary containing 'expected_output' key with reference text",
            ),
        ]

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.BUILTIN

    def __call__(
        self, *, outputs: Any, expectations: dict[str, Any] | None = None
    ) -> Feedback:
        """
        Compute BERTScore between outputs and expected outputs.

        Args:
            outputs: The generated text to evaluate.
            expectations: Dictionary containing 'expected_output' key with reference text.

        Returns:
            Feedback object with F1 score as the value (range: 0.0 to 1.0).
        """
        try:
            from bert_score import score as bert_score_fn
        except ImportError as e:
            raise MlflowException(
                "BERTScore requires the 'bert-score' package. "
                "Install it with: pip install bert-score"
            ) from e

        if not expectations or "expected_output" not in expectations:
            raise MlflowException.invalid_parameter_value(
                "BERTScore requires 'expected_output' in expectations dictionary"
            )

        candidate = _ensure_string(outputs, "outputs")
        reference = _ensure_string(expectations["expected_output"], "expected_output")

        # Compute BERTScore
        P, R, F1 = bert_score_fn(
            [candidate],
            [reference],
            model_type=self.model_type,
            lang=self.lang,
            return_hash=self.return_hash,
            verbose=False,
        )

        # Return F1 score as the primary metric
        f1_value = float(F1[0].item())

        return Feedback(
            name=self.name,
            value=f1_value,
            metadata={
                "precision": float(P[0].item()),
                "recall": float(R[0].item()),
                "model_type": self.model_type,
            },
        )


@experimental(version="3.13.0")
class METEOR(Judge):
    """
    METEOR (Metric for Evaluation of Translation with Explicit ORdering) score.

    METEOR is a metric originally designed for machine translation evaluation that aligns
    hypothesis and reference translations based on exact, stem, synonym, and paraphrase matches.
    It computes a score based on the harmonic mean of precision and recall, with recall
    weighted more heavily than precision.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Args:
        name: The name of the scorer. Defaults to "meteor".
        alpha: Parameter for controlling weight of precision vs recall. Defaults to 0.9.
        beta: Parameter for controlling weight of fragmentation penalty. Defaults to 3.0.
        gamma: Parameter for controlling fragmentation penalty. Defaults to 0.5.

    Example (direct usage):

    .. code-block:: python

        from mlflow.genai.scorers import METEOR

        scorer = METEOR()
        feedback = scorer(
            outputs="The cat sat on the mat",
            expectations={"expected_output": "A cat was sitting on a mat"}
        )
        print(f"METEOR Score: {feedback.value}")

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=[METEOR()])

    Note:
        Requires NLTK with WordNet data: `pip install nltk` and download 'wordnet' corpus.
    """

    name: str = "meteor"
    alpha: float = 0.9
    beta: float = 3.0
    gamma: float = 0.5
    description: str = "Machine translation evaluation metric"

    @property
    def instructions(self) -> str:
        return "Evaluates text quality using METEOR metric with stem, synonym, and paraphrase matching."

    @property
    def feedback_value_type(self) -> Any:
        return float

    def get_input_fields(self) -> list[JudgeField]:
        return [
            JudgeField(
                name="outputs",
                description="The generated text to evaluate",
            ),
            JudgeField(
                name="expectations",
                description="Dictionary containing 'expected_output' key with reference text",
            ),
        ]

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.BUILTIN

    def __call__(
        self, *, outputs: Any, expectations: dict[str, Any] | None = None
    ) -> Feedback:
        """
        Compute METEOR score between outputs and expected outputs.

        Args:
            outputs: The generated text to evaluate.
            expectations: Dictionary containing 'expected_output' key with reference text.

        Returns:
            Feedback object with METEOR score as the value (range: 0.0 to 1.0).
        """
        try:
            from nltk.translate.meteor_score import meteor_score
            import nltk
        except ImportError as e:
            raise MlflowException(
                "METEOR requires the 'nltk' package. Install it with: pip install nltk"
            ) from e

        # Download required NLTK data if not present
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            _logger.info("Downloading NLTK wordnet data for METEOR scorer...")
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)

        if not expectations or "expected_output" not in expectations:
            raise MlflowException.invalid_parameter_value(
                "METEOR requires 'expected_output' in expectations dictionary"
            )

        candidate = _ensure_string(outputs, "outputs")
        reference = _ensure_string(expectations["expected_output"], "expected_output")

        # Tokenize
        candidate_tokens = candidate.split()
        reference_tokens = reference.split()

        # Compute METEOR score
        score = meteor_score(
            [reference_tokens],
            candidate_tokens,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

        return Feedback(
            name=self.name,
            value=float(score),
            metadata={
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
        )


@experimental(version="3.13.0")
class Readability(Judge):
    """
    Readability scorer that computes text complexity metrics.

    This scorer computes multiple readability metrics including:
    - Flesch Reading Ease: Higher scores indicate easier readability (0-100)
    - Flesch-Kincaid Grade Level: U.S. school grade level required to understand the text
    - Automated Readability Index (ARI): Similar to Flesch-Kincaid
    - Coleman-Liau Index: Based on characters rather than syllables

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Args:
        name: The name of the scorer. Defaults to "readability".
        metric: The specific readability metric to return. Options: "flesch_reading_ease",
            "flesch_kincaid_grade", "automated_readability_index", "coleman_liau_index".
            Defaults to "flesch_reading_ease".

    Example (direct usage):

    .. code-block:: python

        from mlflow.genai.scorers import Readability

        scorer = Readability(metric="flesch_reading_ease")
        feedback = scorer(outputs="The cat sat on the mat.")
        print(f"Readability Score: {feedback.value}")

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=[Readability()])

    Note:
        Requires the `textstat` package: `pip install textstat`
    """

    name: str = "readability"
    metric: Literal[
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "automated_readability_index",
        "coleman_liau_index",
    ] = "flesch_reading_ease"
    description: str = "Text complexity and readability metrics"

    @property
    def instructions(self) -> str:
        return f"Evaluates text readability using {self.metric} metric."

    @property
    def feedback_value_type(self) -> Any:
        return float

    def get_input_fields(self) -> list[JudgeField]:
        return [
            JudgeField(
                name="outputs",
                description="The text to evaluate for readability",
            ),
        ]

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.BUILTIN

    def __call__(self, *, outputs: Any) -> Feedback:
        """
        Compute readability score for the given text.

        Args:
            outputs: The text to evaluate.

        Returns:
            Feedback object with the selected readability metric as the value.
        """
        try:
            import textstat
        except ImportError as e:
            raise MlflowException(
                "Readability scorer requires the 'textstat' package. "
                "Install it with: pip install textstat"
            ) from e

        text = _ensure_string(outputs, "outputs")

        # Compute the selected metric
        if self.metric == "flesch_reading_ease":
            score = textstat.flesch_reading_ease(text)
        elif self.metric == "flesch_kincaid_grade":
            score = textstat.flesch_kincaid_grade(text)
        elif self.metric == "automated_readability_index":
            score = textstat.automated_readability_index(text)
        elif self.metric == "coleman_liau_index":
            score = textstat.coleman_liau_index(text)
        else:
            raise MlflowException.invalid_parameter_value(
                f"Unknown readability metric: {self.metric}"
            )

        # Compute all metrics for metadata
        metadata = {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "coleman_liau_index": textstat.coleman_liau_index(text),
            "selected_metric": self.metric,
        }

        return Feedback(
            name=self.name,
            value=float(score),
            metadata=metadata,
        )


@experimental(version="3.13.0")
class Sentiment(Judge):
    """
    Sentiment analysis scorer using VADER (Valence Aware Dictionary and sEntiment Reasoner).

    VADER is a lexicon and rule-based sentiment analysis tool specifically attuned to
    sentiments expressed in social media. It provides compound, positive, negative, and
    neutral sentiment scores.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Args:
        name: The name of the scorer. Defaults to "sentiment".
        return_compound: If True, returns the compound score (normalized between -1 and 1).
            If False, returns individual pos/neg/neu scores. Defaults to True.

    Example (direct usage):

    .. code-block:: python

        from mlflow.genai.scorers import Sentiment

        scorer = Sentiment()
        feedback = scorer(outputs="This is a great product! I love it.")
        print(f"Sentiment Score: {feedback.value}")

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=[Sentiment()])

    Note:
        Requires the `vaderSentiment` package: `pip install vaderSentiment`
    """

    name: str = "sentiment"
    return_compound: bool = True
    description: str = "Sentiment analysis using VADER"

    @property
    def instructions(self) -> str:
        return "Evaluates sentiment polarity of text using VADER sentiment analyzer."

    @property
    def feedback_value_type(self) -> Any:
        return float

    def get_input_fields(self) -> list[JudgeField]:
        return [
            JudgeField(
                name="outputs",
                description="The text to analyze for sentiment",
            ),
        ]

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.BUILTIN

    def __call__(self, *, outputs: Any) -> Feedback:
        """
        Compute sentiment score for the given text.

        Args:
            outputs: The text to analyze.

        Returns:
            Feedback object with sentiment score. If return_compound=True, returns compound
            score (-1 to 1). Otherwise, returns positive score (0 to 1).
        """
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as e:
            raise MlflowException(
                "Sentiment scorer requires the 'vaderSentiment' package. "
                "Install it with: pip install vaderSentiment"
            ) from e

        text = _ensure_string(outputs, "outputs")

        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)

        # Return compound score or positive score
        value = scores["compound"] if self.return_compound else scores["pos"]

        return Feedback(
            name=self.name,
            value=float(value),
            metadata={
                "compound": scores["compound"],
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
            },
        )


@experimental(version="3.13.0")
class LevenshteinRatio(Judge):
    """
    Levenshtein ratio scorer that measures edit distance similarity.

    The Levenshtein ratio is computed as:
        ratio = (len(s1) + len(s2) - distance) / (len(s1) + len(s2))

    This gives a normalized similarity score between 0 and 1, where 1 means identical strings
    and 0 means completely different strings.

    You can invoke the scorer directly with a single input for testing, or pass it to
    `mlflow.genai.evaluate` for running full evaluation on a dataset.

    Args:
        name: The name of the scorer. Defaults to "levenshtein_ratio".
        case_sensitive: If True, comparison is case-sensitive. Defaults to False.

    Example (direct usage):

    .. code-block:: python

        from mlflow.genai.scorers import LevenshteinRatio

        scorer = LevenshteinRatio()
        feedback = scorer(
            outputs="The cat sat on the mat",
            expectations={"expected_output": "The cat sat on a mat"}
        )
        print(f"Similarity: {feedback.value}")

    Example (with evaluate):

    .. code-block:: python

        import mlflow

        data = mlflow.search_traces(...)
        result = mlflow.genai.evaluate(data=data, scorers=[LevenshteinRatio()])

    Note:
        Requires the `python-Levenshtein` package: `pip install python-Levenshtein`
    """

    name: str = "levenshtein_ratio"
    case_sensitive: bool = False
    description: str = "Edit distance similarity between texts"

    @property
    def instructions(self) -> str:
        return "Evaluates string similarity using Levenshtein edit distance ratio."

    @property
    def feedback_value_type(self) -> Any:
        return float

    def get_input_fields(self) -> list[JudgeField]:
        return [
            JudgeField(
                name="outputs",
                description="The generated text to evaluate",
            ),
            JudgeField(
                name="expectations",
                description="Dictionary containing 'expected_output' key with reference text",
            ),
        ]

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.BUILTIN

    def __call__(
        self, *, outputs: Any, expectations: dict[str, Any] | None = None
    ) -> Feedback:
        """
        Compute Levenshtein ratio between outputs and expected outputs.

        Args:
            outputs: The generated text to evaluate.
            expectations: Dictionary containing 'expected_output' key with reference text.

        Returns:
            Feedback object with Levenshtein ratio as the value (range: 0.0 to 1.0).
        """
        try:
            import Levenshtein
        except ImportError as e:
            raise MlflowException(
                "LevenshteinRatio scorer requires the 'python-Levenshtein' package. "
                "Install it with: pip install python-Levenshtein"
            ) from e

        if not expectations or "expected_output" not in expectations:
            raise MlflowException.invalid_parameter_value(
                "LevenshteinRatio requires 'expected_output' in expectations dictionary"
            )

        candidate = _ensure_string(outputs, "outputs")
        reference = _ensure_string(expectations["expected_output"], "expected_output")

        # Apply case normalization if needed
        if not self.case_sensitive:
            candidate = candidate.lower()
            reference = reference.lower()

        # Compute Levenshtein ratio
        ratio = Levenshtein.ratio(candidate, reference)

        # Also compute distance for metadata
        distance = Levenshtein.distance(candidate, reference)

        return Feedback(
            name=self.name,
            value=float(ratio),
            metadata={
                "edit_distance": distance,
                "case_sensitive": self.case_sensitive,
            },
        )
