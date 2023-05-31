from functools import total_ordering
from concurrent.futures import as_completed


@total_ordering
class SortableResult:
    def __init__(self, key, value=None, err=None):
        if (value, err).count(None) != 1:
            raise ValueError("Exactly one of `value` and `error` must be set.")
        self.key = key
        self.value = value
        self.err = err

    def is_err(self):
        return self.err is not None

    def is_ok(self):
        return self.value is not None

    def __lt__(self, other):
        return self.key < other.key


def complete_futures(futures):
    """
    Completes the specified futures, yielding a `SortableResult` for each future as it completes.
    Note that this function returns an iterator and does not preserve ordering of results.
    """
    for index, fut in as_completed(enumerate(futures)):
        try:
            yield SortableResult(key=index, value=fut.result())
        except Exception as e:
            yield SortableResult(key=index, err=e)
