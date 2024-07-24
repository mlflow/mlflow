# Customized from https://github.com/open-telemetry/opentelemetry-python/blob/754fc36a408dd45e86d4a0f820f84e692f14b4c1/opentelemetry-api/src/opentelemetry/util/_once.py
from threading import Lock
from typing import Callable


class Once:
    """Execute a function exactly once and block all callers until the function returns"""

    def __init__(self) -> None:
        self.__lock = Lock()
        self.__done = False

    @property
    def done(self):
        with self.__lock:
            return self.__done

    @done.setter
    def done(self, value):
        with self.__lock:
            self.__done = value

    def do_once(self, func: Callable[[], None]):
        """
        Execute ``func`` if it hasn't been executed or return.
        Will block until ``func`` has been called by one thread.
        """
        if self.__done:
            return

        with self.__lock:
            if not self.__done:
                func()
                self.__done = True
                return
