import os
import threading


class ThreadLocalVariable:
    """
    Class for creating a thread local variable.
    """

    def __init__(self, init_value, reset_in_subprocess=True):
        self.reset_in_subprocess = reset_in_subprocess
        self.init_value = init_value
        self.thread_local = threading.local()

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
                self.set(self.init_value)
                return self.init_value
            else:
                return value
        else:
            self.set(self.init_value)
            return self.init_value

    def set(self, value):
        """
        Set a value for the thread-local variable.
        """
        self.thread_local.value = (value, os.getpid())
