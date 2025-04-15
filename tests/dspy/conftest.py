import dspy
import pytest


@pytest.fixture(autouse=True)
def reset_dspy_settings():
    dspy.settings.configure(callbacks=[], lm=None, adapter=None)
