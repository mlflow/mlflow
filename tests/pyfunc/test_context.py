import random
import time
from threading import Thread

import pytest

from mlflow.exceptions import MlflowException
from mlflow.pyfunc.context import (
    Context,
    get_prediction_context,
    maybe_get_evaluation_request_id,
    set_prediction_context,
)


def test_prediction_context_thread_safe():
    def set_context(context):
        with set_prediction_context(context):
            time.sleep(0.2 * random.random())
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


def test_maybe_get_evaluation_request_id():
    assert maybe_get_evaluation_request_id() is None

    with set_prediction_context(Context(request_id="eval", is_evaluate=True)):
        assert maybe_get_evaluation_request_id() == "eval"

    with set_prediction_context(Context(request_id="non_eval", is_evaluate=False)):
        assert maybe_get_evaluation_request_id() is None

    with pytest.raises(MlflowException, match="When prediction request context"):
        with set_prediction_context(Context(request_id=None, is_evaluate=True)):
            maybe_get_evaluation_request_id()
