import random
import time
from threading import Thread

import pytest

import mlflow
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


def test_prediction_context_pyfunc_predict():
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, model_inputs):
            context = get_prediction_context()
            return context.request_id

        def predict_stream(self, model_inputs):
            for _ in range(3):
                context = get_prediction_context()
                yield context.request_id

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(name="model", python_model=MyModel())
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)

    with set_prediction_context(Context(request_id="request_1")):
        assert pyfunc_model.predict(None) == "request_1"

    with set_prediction_context(Context(request_id="request_2")):
        generator = pyfunc_model.predict_stream(None)

    # When a prediction context is set for predict_stream call, it should also
    # be effective during the iteration of the generator.
    for output in generator:
        assert output == "request_2"
