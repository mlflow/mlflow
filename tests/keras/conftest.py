import keras
import pytest


@pytest.fixture(autouse=True)
def clear_keras_session():
    # Reset Keras global state before each test so optimizer names aren't uniquified
    # across tests (a second Adam in the same session becomes "adam_1", which breaks
    # the optimizer_name assertions in test_autolog.py / test_callback.py).
    keras.backend.clear_session()
    yield
    # Also clear on teardown so a failing test doesn't leave global state behind.
    keras.backend.clear_session()
