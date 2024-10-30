import os
import threading


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
        # The `_value_dict` attribute saves all thread-local values, the key is thread ID.
        # It is used by MLflow Spark datasource autologging.
        self._value_dict = {}

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
        self._value_dict[threading.currentThread().ident] = value
