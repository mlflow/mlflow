from concurrent.futures import as_completed


class Result:
    def __init__(self, value=None, err=None):
        if (value, err).count(None) != 1:
            raise ValueError("Exactly one of `value` and `error` must be set.")
        self.value = value
        self.err = err

    def is_err(self):
        return self.err is not None

    def is_ok(self):
        return self.value is not None

    @classmethod
    def from_future(cls, future):
        try:
            return cls(value=future.result())
        except Exception as e:
            return cls(err=e)


def complete_futures(futures):
    """
    Completes the specified futures, yielding a `_Result` for each future as it completes.
    Note that this function returns an iterator and does not preserve ordering of results.
    """
    return map(Result.from_future, as_completed(futures))
