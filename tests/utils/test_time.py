from mlflow.utils.time import Timer


def test_timer():
    with Timer() as t:
        pass

    assert t.time >= 0
