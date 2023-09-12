import time

from mlflow.utils.time import Timer


def test_timer():
    with Timer() as t:
        time.sleep(1)

    assert f"{t:.1f}" == "1.0"
    assert f"{t:.2f}" == "1.00"
