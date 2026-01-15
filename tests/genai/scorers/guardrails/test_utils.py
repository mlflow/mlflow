import sys
from unittest.mock import patch

import pytest

from mlflow.exceptions import MlflowException


def test_check_guardrails_installed_success():
    with patch.dict("sys.modules", {"guardrails": object()}):
        from mlflow.genai.scorers.guardrails.utils import check_guardrails_installed

        check_guardrails_installed()


def test_check_guardrails_installed_failure():
    from mlflow.genai.scorers.guardrails.utils import check_guardrails_installed

    # Save original guardrails module
    original_guardrails = sys.modules.get("guardrails")

    try:
        # Remove guardrails from sys.modules to simulate it not being installed
        if "guardrails" in sys.modules:
            del sys.modules["guardrails"]

        # Mock import to raise ImportError
        with patch.dict(sys.modules, {"guardrails": None}):
            # The check function tries to import guardrails, which should fail
            with pytest.raises(MlflowException, match="guardrails-ai"):
                check_guardrails_installed()
    finally:
        # Restore original module
        if original_guardrails is not None:
            sys.modules["guardrails"] = original_guardrails


@pytest.mark.parametrize(
    ("inputs", "outputs", "expected"),
    [
        ("input text", "output text", "output text"),
        ("input only", None, "input only"),
        ({"query": "test"}, None, "test"),
    ],
)
def test_map_scorer_inputs_to_text(inputs, outputs, expected):
    from mlflow.genai.scorers.guardrails.utils import map_scorer_inputs_to_text

    result = map_scorer_inputs_to_text(inputs=inputs, outputs=outputs)

    assert expected in result


def test_map_scorer_inputs_to_text_requires_input_or_output():
    from mlflow.genai.scorers.guardrails.utils import map_scorer_inputs_to_text

    with pytest.raises(MlflowException, match="require either 'outputs' or 'inputs'"):
        map_scorer_inputs_to_text(inputs=None, outputs=None)
