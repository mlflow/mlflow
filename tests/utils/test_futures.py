import time
from concurrent.futures import ThreadPoolExecutor

from mlflow.utils.futures import complete_futures


def sleep(sec):
    time.sleep(sec)
    return sec


def test_complete_futures():
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(sleep, 0.0),
            executor.submit(sleep, 0.2),
            executor.submit(sleep, 0.1),
        ]
        results = list(complete_futures(futures))
        assert [r.value for r in results] == [0.0, 0.1, 0.2]
        assert [r.value for r in sorted(results)] == [0.0, 0.2, 0.1]
        assert [r.value for r in sorted(results, reverse=True)] == [0.1, 0.2, 0.0]


def throw():
    raise Exception("error")


def test_complete_futures_error():
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(throw),
            executor.submit(sleep, 0.1),
        ]
        results = complete_futures(futures)
        first = next(results)
        assert first.is_err()
        assert type(first.err) is Exception
        assert first.err.args == ("error",)
        second = next(results)
        assert second.is_ok()
        assert second.value == 0.1
