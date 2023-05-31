from functools import total_ordering
from concurrent.futures import as_completed


@total_ordering
class SortableResult:
    def __init__(self, key, value, err):
        """
        Should not be called directly. Use `SortableResult.ok` or `SortableResult.err` instead.
        """
        self.key = key
        self.value = value
        self.err = err

    @classmethod
    def ok(cls, key, value):
        return cls(key=key, value=value, err=None)

    @classmethod
    def error(cls, key, err):
        return cls(key=key, value=None, err=err)

    def is_err(self):
        return self.err is not None

    def is_ok(self):
        return self.err is None

    def __lt__(self, other):
        return self.key < other.key


def complete_futures(futures):
    """
    Completes the specified futures, yielding a `SortableResult` for each future as it completes.
    Note that this function returns a generator and does not preserve ordering of results.

    :param futures: A dict of futures to complete.
    :return: An iterator over `SortableResult` objects.
    """
    for fut in as_completed(futures):
        key = futures[fut]
        try:
            result = fut.result()
        except Exception as e:
            yield SortableResult.error(key=key, err=e)
        else:
            yield SortableResult.ok(key=key, value=result)
