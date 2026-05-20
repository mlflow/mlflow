"""
Tests for deterministic NLP scorers.
"""

import pytest

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException


class TestBERTScore:
    """Tests for BERTScore scorer."""

    def test_bert_score_basic(self):
        """Test basic BERTScore functionality."""
        pytest.importorskip("bert_score")
        from mlflow.genai.scorers import BERTScore

        scorer = BERTScore()
        feedback = scorer(
            outputs="The cat sat on the mat",
            expectations={"expected_output": "A cat was sitting on a mat"},
        )

        assert isinstance(feedback, Feedback)
        assert feedback.name == "bert_score"
        assert 0.0 <= feedback.value <= 1.0
        assert "precision" in feedback.metadata
        assert "recall" in feedback.metadata
        assert "model_type" in feedback.metadata

    def test_bert_score_custom_model(self):
        """Test BERTScore with custom model."""
        pytest.importorskip("bert_score")
        from mlflow.genai.scorers import BERTScore

        scorer = BERTScore(model_type="distilbert-base-uncased")
        feedback = scorer(
            outputs="Hello world",
            expectations={"expected_output": "Hi world"},
        )

        assert isinstance(feedback, Feedback)
        assert feedback.metadata["model_type"] == "distilbert-base-uncased"

    def test_bert_score_missing_expectations(self):
        """Test BERTScore with missing expectations."""
        pytest.importorskip("bert_score")
        from mlflow.genai.scorers import BERTScore

        scorer = BERTScore()
        with pytest.raises(MlflowException, match="expected_output"):
            scorer(outputs="Some text", expectations={})

    def test_bert_score_none_expectations(self):
        """Test BERTScore with None expectations."""
        pytest.importorskip("bert_score")
        from mlflow.genai.scorers import BERTScore

        scorer = BERTScore()
        with pytest.raises(MlflowException, match="expected_output"):
            scorer(outputs="Some text", expectations=None)

    def test_bert_score_identical_texts(self):
        """Test BERTScore with identical texts."""
        pytest.importorskip("bert_score")
        from mlflow.genai.scorers import BERTScore

        scorer = BERTScore()
        text = "The quick brown fox jumps over the lazy dog"
        feedback = scorer(
            outputs=text,
            expectations={"expected_output": text},
        )

        # Identical texts should have high F1 score
        assert feedback.value > 0.95

    def test_bert_score_import_error(self, monkeypatch):
        """Test BERTScore raises error when bert-score not installed."""
        from mlflow.genai.scorers import BERTScore

        # Mock the import to raise ImportError
        import sys

        monkeypatch.setitem(sys.modules, "bert_score", None)

        scorer = BERTScore()
        with pytest.raises(MlflowException, match="bert-score"):
            scorer(
                outputs="test",
                expectations={"expected_output": "test"},
            )


class TestMETEOR:
    """Tests for METEOR scorer."""

    def test_meteor_basic(self):
        """Test basic METEOR functionality."""
        pytest.importorskip("nltk")
        from mlflow.genai.scorers import METEOR

        scorer = METEOR()
        feedback = scorer(
            outputs="The cat sat on the mat",
            expectations={"expected_output": "A cat was sitting on a mat"},
        )

        assert isinstance(feedback, Feedback)
        assert feedback.name == "meteor"
        assert 0.0 <= feedback.value <= 1.0
        assert "alpha" in feedback.metadata
        assert "beta" in feedback.metadata
        assert "gamma" in feedback.metadata

    def test_meteor_custom_params(self):
        """Test METEOR with custom parameters."""
        pytest.importorskip("nltk")
        from mlflow.genai.scorers import METEOR

        scorer = METEOR(alpha=0.8, beta=2.0, gamma=0.3)
        feedback = scorer(
            outputs="Hello world",
            expectations={"expected_output": "Hi world"},
        )

        assert isinstance(feedback, Feedback)
        assert feedback.metadata["alpha"] == 0.8
        assert feedback.metadata["beta"] == 2.0
        assert feedback.metadata["gamma"] == 0.3

    def test_meteor_missing_expectations(self):
        """Test METEOR with missing expectations."""
        pytest.importorskip("nltk")
        from mlflow.genai.scorers import METEOR

        scorer = METEOR()
        with pytest.raises(MlflowException, match="expected_output"):
            scorer(outputs="Some text", expectations={})

    def test_meteor_identical_texts(self):
        """Test METEOR with identical texts."""
        pytest.importorskip("nltk")
        from mlflow.genai.scorers import METEOR

        scorer = METEOR()
        text = "The quick brown fox"
        feedback = scorer(
            outputs=text,
            expectations={"expected_output": text},
        )

        # Identical texts should have perfect score
        assert feedback.value == 1.0

    def test_meteor_import_error(self, monkeypatch):
        """Test METEOR raises error when nltk not installed."""
        from mlflow.genai.scorers import METEOR

        import sys

        monkeypatch.setitem(sys.modules, "nltk", None)

        scorer = METEOR()
        with pytest.raises(MlflowException, match="nltk"):
            scorer(
                outputs="test",
                expectations={"expected_output": "test"},
            )


class TestReadability:
    """Tests for Readability scorer."""

    def test_readability_flesch_reading_ease(self):
        """Test Readability with Flesch Reading Ease metric."""
        pytest.importorskip("textstat")
        from mlflow.genai.scorers import Readability

        scorer = Readability(metric="flesch_reading_ease")
        feedback = scorer(outputs="The cat sat on the mat. It was a sunny day.")

        assert isinstance(feedback, Feedback)
        assert feedback.name == "readability"
        assert isinstance(feedback.value, float)
        assert "flesch_reading_ease" in feedback.metadata
        assert "flesch_kincaid_grade" in feedback.metadata
        assert feedback.metadata["selected_metric"] == "flesch_reading_ease"

    def test_readability_flesch_kincaid_grade(self):
        """Test Readability with Flesch-Kincaid Grade metric."""
        pytest.importorskip("textstat")
        from mlflow.genai.scorers import Readability

        scorer = Readability(metric="flesch_kincaid_grade")
        feedback = scorer(outputs="The cat sat on the mat.")

        assert isinstance(feedback, Feedback)
        assert feedback.metadata["selected_metric"] == "flesch_kincaid_grade"

    def test_readability_ari(self):
        """Test Readability with Automated Readability Index."""
        pytest.importorskip("textstat")
        from mlflow.genai.scorers import Readability

        scorer = Readability(metric="automated_readability_index")
        feedback = scorer(outputs="Simple text here.")

        assert isinstance(feedback, Feedback)
        assert feedback.metadata["selected_metric"] == "automated_readability_index"

    def test_readability_coleman_liau(self):
        """Test Readability with Coleman-Liau Index."""
        pytest.importorskip("textstat")
        from mlflow.genai.scorers import Readability

        scorer = Readability(metric="coleman_liau_index")
        feedback = scorer(outputs="Another simple sentence.")

        assert isinstance(feedback, Feedback)
        assert feedback.metadata["selected_metric"] == "coleman_liau_index"

    def test_readability_complex_text(self):
        """Test Readability with complex text."""
        pytest.importorskip("textstat")
        from mlflow.genai.scorers import Readability

        scorer = Readability()
        complex_text = (
            "The implementation of sophisticated algorithms necessitates "
            "comprehensive understanding of computational complexity theory."
        )
        feedback = scorer(outputs=complex_text)

        assert isinstance(feedback, Feedback)
        # Complex text should have lower readability score
        assert feedback.value < 60  # Flesch Reading Ease

    def test_readability_import_error(self, monkeypatch):
        """Test Readability raises error when textstat not installed."""
        from mlflow.genai.scorers import Readability

        import sys

        monkeypatch.setitem(sys.modules, "textstat", None)

        scorer = Readability()
        with pytest.raises(MlflowException, match="textstat"):
            scorer(outputs="test")


class TestSentiment:
    """Tests for Sentiment scorer."""

    def test_sentiment_positive(self):
        """Test Sentiment with positive text."""
        pytest.importorskip("vaderSentiment")
        from mlflow.genai.scorers import Sentiment

        scorer = Sentiment()
        feedback = scorer(outputs="This is amazing! I love it so much!")

        assert isinstance(feedback, Feedback)
        assert feedback.name == "sentiment"
        assert feedback.value > 0  # Positive sentiment
        assert "compound" in feedback.metadata
        assert "positive" in feedback.metadata
        assert "negative" in feedback.metadata
        assert "neutral" in feedback.metadata

    def test_sentiment_negative(self):
        """Test Sentiment with negative text."""
        pytest.importorskip("vaderSentiment")
        from mlflow.genai.scorers import Sentiment

        scorer = Sentiment()
        feedback = scorer(outputs="This is terrible! I hate it!")

        assert isinstance(feedback, Feedback)
        assert feedback.value < 0  # Negative sentiment

    def test_sentiment_neutral(self):
        """Test Sentiment with neutral text."""
        pytest.importorskip("vaderSentiment")
        from mlflow.genai.scorers import Sentiment

        scorer = Sentiment()
        feedback = scorer(outputs="The cat sat on the mat.")

        assert isinstance(feedback, Feedback)
        # Neutral text should have compound score close to 0
        assert -0.5 < feedback.value < 0.5

    def test_sentiment_return_positive_score(self):
        """Test Sentiment returning positive score instead of compound."""
        pytest.importorskip("vaderSentiment")
        from mlflow.genai.scorers import Sentiment

        scorer = Sentiment(return_compound=False)
        feedback = scorer(outputs="This is great!")

        assert isinstance(feedback, Feedback)
        assert 0.0 <= feedback.value <= 1.0  # Positive score is 0-1

    def test_sentiment_import_error(self, monkeypatch):
        """Test Sentiment raises error when vaderSentiment not installed."""
        from mlflow.genai.scorers import Sentiment

        import sys

        monkeypatch.setitem(sys.modules, "vaderSentiment", None)

        scorer = Sentiment()
        with pytest.raises(MlflowException, match="vaderSentiment"):
            scorer(outputs="test")


class TestLevenshteinRatio:
    """Tests for LevenshteinRatio scorer."""

    def test_levenshtein_identical(self):
        """Test LevenshteinRatio with identical strings."""
        pytest.importorskip("Levenshtein")
        from mlflow.genai.scorers import LevenshteinRatio

        scorer = LevenshteinRatio()
        text = "The cat sat on the mat"
        feedback = scorer(
            outputs=text,
            expectations={"expected_output": text},
        )

        assert isinstance(feedback, Feedback)
        assert feedback.name == "levenshtein_ratio"
        assert feedback.value == 1.0  # Identical strings
        assert feedback.metadata["edit_distance"] == 0

    def test_levenshtein_similar(self):
        """Test LevenshteinRatio with similar strings."""
        pytest.importorskip("Levenshtein")
        from mlflow.genai.scorers import LevenshteinRatio

        scorer = LevenshteinRatio()
        feedback = scorer(
            outputs="The cat sat on the mat",
            expectations={"expected_output": "The cat sat on a mat"},
        )

        assert isinstance(feedback, Feedback)
        assert 0.8 < feedback.value < 1.0  # Very similar
        assert feedback.metadata["edit_distance"] > 0

    def test_levenshtein_different(self):
        """Test LevenshteinRatio with different strings."""
        pytest.importorskip("Levenshtein")
        from mlflow.genai.scorers import LevenshteinRatio

        scorer = LevenshteinRatio()
        feedback = scorer(
            outputs="Hello world",
            expectations={"expected_output": "Goodbye universe"},
        )

        assert isinstance(feedback, Feedback)
        assert feedback.value < 0.5  # Very different

    def test_levenshtein_case_sensitive(self):
        """Test LevenshteinRatio with case sensitivity."""
        pytest.importorskip("Levenshtein")
        from mlflow.genai.scorers import LevenshteinRatio

        scorer_insensitive = LevenshteinRatio(case_sensitive=False)
        scorer_sensitive = LevenshteinRatio(case_sensitive=True)

        feedback_insensitive = scorer_insensitive(
            outputs="Hello World",
            expectations={"expected_output": "hello world"},
        )
        feedback_sensitive = scorer_sensitive(
            outputs="Hello World",
            expectations={"expected_output": "hello world"},
        )

        # Case-insensitive should be perfect match
        assert feedback_insensitive.value == 1.0
        # Case-sensitive should not be perfect match
        assert feedback_sensitive.value < 1.0

    def test_levenshtein_missing_expectations(self):
        """Test LevenshteinRatio with missing expectations."""
        pytest.importorskip("Levenshtein")
        from mlflow.genai.scorers import LevenshteinRatio

        scorer = LevenshteinRatio()
        with pytest.raises(MlflowException, match="expected_output"):
            scorer(outputs="Some text", expectations={})

    def test_levenshtein_import_error(self, monkeypatch):
        """Test LevenshteinRatio raises error when python-Levenshtein not installed."""
        from mlflow.genai.scorers import LevenshteinRatio

        import sys

        monkeypatch.setitem(sys.modules, "Levenshtein", None)

        scorer = LevenshteinRatio()
        with pytest.raises(MlflowException, match="python-Levenshtein"):
            scorer(
                outputs="test",
                expectations={"expected_output": "test"},
            )


class TestScorerIntegration:
    """Integration tests for deterministic NLP scorers."""

    def test_scorer_names(self):
        """Test that all scorers have correct default names."""
        pytest.importorskip("bert_score")
        pytest.importorskip("nltk")
        pytest.importorskip("textstat")
        pytest.importorskip("vaderSentiment")
        pytest.importorskip("Levenshtein")

        from mlflow.genai.scorers import (
            BERTScore,
            METEOR,
            Readability,
            Sentiment,
            LevenshteinRatio,
        )

        assert BERTScore().name == "bert_score"
        assert METEOR().name == "meteor"
        assert Readability().name == "readability"
        assert Sentiment().name == "sentiment"
        assert LevenshteinRatio().name == "levenshtein_ratio"

    def test_scorer_custom_names(self):
        """Test that scorers accept custom names."""
        pytest.importorskip("textstat")
        from mlflow.genai.scorers import Readability

        scorer = Readability(name="my_custom_readability")
        assert scorer.name == "my_custom_readability"

    def test_scorer_descriptions(self):
        """Test that all scorers have descriptions."""
        pytest.importorskip("bert_score")
        pytest.importorskip("nltk")
        pytest.importorskip("textstat")
        pytest.importorskip("vaderSentiment")
        pytest.importorskip("Levenshtein")

        from mlflow.genai.scorers import (
            BERTScore,
            METEOR,
            Readability,
            Sentiment,
            LevenshteinRatio,
        )

        assert BERTScore().description
        assert METEOR().description
        assert Readability().description
        assert Sentiment().description
        assert LevenshteinRatio().description

    def test_scorer_kind(self):
        """Test that all scorers have correct kind."""
        pytest.importorskip("textstat")
        from mlflow.genai.scorers import Readability
        from mlflow.genai.scorers.base import ScorerKind

        scorer = Readability()
        assert scorer.kind == ScorerKind.BUILTIN

    def test_multiple_scorers_together(self):
        """Test using multiple scorers together."""
        pytest.importorskip("textstat")
        pytest.importorskip("vaderSentiment")

        from mlflow.genai.scorers import Readability, Sentiment

        text = "This is a great product! I love it."

        readability_scorer = Readability()
        sentiment_scorer = Sentiment()

        readability_feedback = readability_scorer(outputs=text)
        sentiment_feedback = sentiment_scorer(outputs=text)

        assert isinstance(readability_feedback, Feedback)
        assert isinstance(sentiment_feedback, Feedback)
        assert readability_feedback.name == "readability"
        assert sentiment_feedback.name == "sentiment"
