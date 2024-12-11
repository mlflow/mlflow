import random
import time
from threading import Thread

import pytest

from mlflow.pyfunc.context import (
    Context,
    get_prediction_context,
    set_prediction_context,
)


def test_prediction_context_thread_safe():
    def set_context(context):
        with set_prediction_context(context):
            time.sleep(0.2 * random.random())
            assert get_prediction_context() == context
            context.update(is_evaluate=not context.is_evaluate)
            assert get_prediction_context() == context

    threads = []
    for i in range(10):
        context = Context(request_id=f"request_{i}", is_evaluate=random.choice([True, False]))
        thread = Thread(target=set_context, args=(context,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    assert get_prediction_context() is None


def test_set_prediction_context_raise_on_invalid_context():
    with pytest.raises(TypeError, match="Expected context to be an instance of Context"):
        with set_prediction_context("invalid"):
            pass
