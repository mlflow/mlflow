"""
Smoke tests validating the Guardrails AI documentation examples work end-to-end.

Requires:
    pip install guardrails-ai

Run:
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m pytest tests/genai/scorers/guardrails/test_guardrails_docs.py -v
"""

import pytest

import mlflow
from mlflow.genai.scorers.guardrails import (
    DetectJailbreak,
    DetectPII,
    GibberishText,
    NSFWText,
    SecretsPresent,
    ToxicLanguage,
    get_scorer,
)

MODEL = "openai:/gpt-5-mini"


@pytest.fixture(autouse=True)
def tracking_uri():
    mlflow.set_tracking_uri("http://localhost:5000")


# ── Direct scorer calls ──────────────────────────────────────────────


def test_toxic_language_pass():
    scorer = ToxicLanguage(threshold=0.7)
    feedback = scorer(outputs="This is a professional and helpful response.")
    assert feedback.value == "yes"


def test_toxic_language_fail():
    scorer = ToxicLanguage(threshold=0.7)
    feedback = scorer(outputs="You are an absolute idiot and I hate you.")
    assert feedback.value == "no"


def test_detect_pii_clean():
    scorer = DetectPII()
    feedback = scorer(outputs="MLflow is an open-source platform.")
    assert feedback.value == "yes"


def test_detect_pii_found():
    scorer = DetectPII()
    feedback = scorer(outputs="Contact john@email.com or call 555-867-5309 for details.")
    assert feedback.value == "no"


def test_detect_pii_custom_entities():
    scorer = DetectPII(pii_entities=["EMAIL_ADDRESS"])
    feedback = scorer(outputs="Email me at alice@example.org")
    assert feedback.value == "no"


def test_secrets_present_clean():
    scorer = SecretsPresent()
    feedback = scorer(outputs="Use the MLflow UI to view experiments.")
    assert feedback.value == "yes"


def test_secrets_present_found():
    scorer = SecretsPresent()
    feedback = scorer(outputs="Use this key: sk-1234567890abcdefghijklmnopqrstuvwxyz")
    assert feedback.value == "no"


def test_gibberish_text_clean():
    scorer = GibberishText()
    feedback = scorer(outputs="MLflow provides experiment tracking and model management.")
    assert feedback.value == "yes"


def test_gibberish_text_found():
    scorer = GibberishText()
    feedback = scorer(outputs="asdf jkl; qwerty uiop zxcv bnm")
    assert feedback.value == "no"


def test_nsfw_text_clean():
    scorer = NSFWText()
    feedback = scorer(outputs="MLflow helps manage the ML lifecycle.")
    assert feedback.value == "yes"


def test_detect_jailbreak_clean():
    scorer = DetectJailbreak()
    feedback = scorer(inputs="What is MLflow?")
    assert feedback.value == "yes"


# ── get_scorer ────────────────────────────────────────────────────────


def test_get_scorer_toxic_language():
    scorer = get_scorer("ToxicLanguage", threshold=0.7)
    feedback = scorer(outputs="This is a friendly response.")
    assert feedback.value in ("yes", "no")
    assert feedback.error is None


# ── mlflow.genai.evaluate ────────────────────────────────────────────


def test_evaluate_integration():
    eval_dataset = [
        {
            "inputs": {"query": "What is MLflow?"},
            "outputs": "MLflow is an open-source platform for managing ML workflows.",
        },
        {
            "inputs": {"query": "How do I contact support?"},
            "outputs": "You can reach us at support@example.com or call 555-0123.",
        },
    ]

    results = mlflow.genai.evaluate(
        data=eval_dataset,
        scorers=[
            ToxicLanguage(threshold=0.7),
            DetectPII(),
        ],
    )

    assert results is not None
    assert results.metrics is not None
