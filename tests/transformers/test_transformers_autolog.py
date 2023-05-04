import pytest

import mlflow


def test_with_autolog():
    mlflow.autolog()
