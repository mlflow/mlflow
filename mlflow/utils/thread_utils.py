import os
import threading
from typing import Any


class ThreadLocalVariable:
    """
    Class for creating a thread local variable.

    Args:
        default_factory: A function used to create the default value
        reset_in_subprocess: Indicating whether the variable is reset in subprocess.
    """

    def __init__(self, default_factory, reset_in_subprocess=True):
        self.reset_in_subprocess = reset_in_subprocess
        self.default_factory = default_factory
        self.thread_local = threading.local()
        # The `__global_thread_values` attribute saves all thread-local values,
        # the key is thread ID.
        self.__global_thread_values: dict[int, Any] = {}

    def get(self):
        """
        Get the thread-local variable value.
        If the thread-local variable is not set, return the provided `init_value` value.
        If `get` is called in a forked subprocess and `reset_in_subprocess` is True,
        return the provided `init_value` value
        """
        if hasattr(self.thread_local, "value"):
            value, pid = self.thread_local.value
            if self.reset_in_subprocess and pid != os.getpid():
                # `get` is called in a forked subprocess, reset it.
                init_value = self.default_factory()
                self.set(init_value)
                return init_value
            else:
                return value
        else:
            init_value = self.default_factory()
            self.set(init_value)
            return init_value

    def set(self, value):
        """
        Set a value for the thread-local variable.
        """
        self.thread_local.value = (value, os.getpid())
        self.__global_thread_values[threading.get_ident()] = value

    def get_all_thread_values(self) -> dict[int, Any]:
        """
        Return all thread values as a dict, dict key is the thread ID.
        """
        return self.__global_thread_values.copy()
