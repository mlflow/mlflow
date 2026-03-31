"""Test that mlflow.types.chat can be imported without numpy.

This verifies the fix for https://github.com/mlflow/mlflow/issues/21779
where mlflow-skinny users couldn't use mlflow.anthropic.autolog() because
importing mlflow.types.chat transitively pulled in numpy via
mlflow.types.__init__ -> mlflow.types.llm -> mlflow.types.schema.
"""

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def is_skinny():
    if "MLFLOW_SKINNY" not in os.environ:
        pytest.skip("This test is only valid for the skinny client")


def test_mlflow_types_chat_importable_without_numpy():
    # numpy should not be available in skinny environment before explicit install
    assert "numpy" not in sys.modules

    # This import chain was failing before the fix:
    # mlflow.types.chat -> mlflow.types.__init__ -> mlflow.types.llm -> mlflow.types.schema -> numpy
    from mlflow.types.chat import ChatMessage  # noqa: F401
