import os
import threading


class ThreadLocalVariable:
    """
    Class for creating a thread local variable.
    """

    def __init__(self, init_value_creator, reset_in_subprocess=True):
        self.reset_in_subprocess = reset_in_subprocess
        self.init_value_creator = init_value_creator
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
                init_value = self.init_value_creator()
                self.set(init_value)
                return init_value
            else:
                return value
        else:
            init_value = self.init_value_creator()
            self.set(init_value)
            return init_value

    def set(self, value):
        """
        Set a value for the thread-local variable.
        """
        self.thread_local.value = (value, os.getpid())
