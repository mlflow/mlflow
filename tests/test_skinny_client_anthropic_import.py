"""Test that mlflow.types.chat can be imported without numpy.

This verifies the fix for https://github.com/mlflow/mlflow/issues/21779
where mlflow-skinny users couldn't use mlflow.anthropic.autolog() because
importing mlflow.types.chat transitively pulled in numpy via
mlflow.types.__init__ -> mlflow.types.llm -> mlflow.types.schema.
"""

import importlib.util
import os

import pytest


@pytest.fixture(autouse=True)
def is_skinny():
    if "MLFLOW_SKINNY" not in os.environ:
        pytest.skip("This test is only valid for the skinny client")


def test_mlflow_types_chat_importable_without_numpy():
    # Verify numpy is genuinely not installed (not just not yet imported)
    assert importlib.util.find_spec("numpy") is None

    # This import chain was failing before the fix:
    # mlflow.types.chat -> mlflow.types.__init__ -> mlflow.types.llm -> mlflow.types.schema -> numpy
    from mlflow.types.chat import ChatMessage  # noqa: F401
