import time

from mlflow.utils.time import Timer


def test_timer():
    with Timer() as t:
        time.sleep(0.1)

    assert f"{t}" == f"{t.elapsed}"
    assert f"{t:.3f}" == f"{t.elapsed:.3f}"
