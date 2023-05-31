import time
from concurrent.futures import ThreadPoolExecutor

from mlflow.utils.futures import complete_futures


def sleep(sec):
    time.sleep(sec)
    return sec


def test_complete_futures():
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(sleep, 0),
            executor.submit(sleep, 2),
            executor.submit(sleep, 1),
        ]
        results = list(complete_futures(futures))
        assert [r.value for r in results] == [0, 1, 2]
        assert [r.value for r in sorted(results)] == [0, 2, 1]
        assert [r.value for r in sorted(results, reverse=True)] == [1, 2, 0]
