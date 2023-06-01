from functools import total_ordering
from concurrent.futures import as_completed


@total_ordering
class SortableResult:
    def __init__(self, key, value, error):
        self.key = key
        self.value = value
        self.error = error
        self.is_err = error is not None
        self.is_ok = not self.is_err

    @classmethod
    def ok(cls, key, value):
        return cls(key, value=value, error=None)

    @classmethod
    def err(cls, key, error):
        return cls(key, value=None, error=error)

    def __repr__(self):
        if self.is_ok:
            return "Ok({})".format(repr(self.value))
        else:
            return "Err({})".format(repr(self.error))

    def __lt__(self, other):
        return self.key < other.key


def complete_futures(futures):
    """
    Completes the specified futures, yielding a `SortableResult` for each future as it completes.
    Note the caller is responsible for consuming the iterator until completion and sorting the
    results if necessary.

    :param futures: A dict of futures to complete.
    :return: An iterator over `SortableResult` objects.
    """
    for fut in as_completed(futures):
        key = futures[fut]
        try:
            result = fut.result()
        except Exception as e:
            yield SortableResult.err(key=key, error=e)
        else:
            yield SortableResult.ok(key=key, value=result)
