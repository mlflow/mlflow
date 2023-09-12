import time

from mlflow.utils.time import Timer


def test_timer():
    with Timer() as t:
        time.sleep(1)

    assert 1.0 < t.time < 1.1
    assert f"{t:.1f}" == "1.0"
