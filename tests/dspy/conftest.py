import dspy
import pytest


@pytest.fixture(autouse=True)
def reset_dspy_settings():
    # DSPy 3.5 changed the default adapter from `ChatAdapter` to `XMLAdapter`. Pin `ChatAdapter`
    # so the existing tests behave the same on every supported version. Tests that cover the
    # default adapter override this with `dspy.configure(adapter=None)`.
    dspy.settings.configure(callbacks=[], lm=None, rm=None, adapter=dspy.ChatAdapter())
