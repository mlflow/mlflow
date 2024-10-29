import os
import threading

_thread_locals = threading.local()
_globals = {}
_globals_lock = threading.Lock()


def set_thread_local_var(key, value):
    """
    Set a thread-local variable.
    """
    global _thread_locals
    setattr(_thread_locals, key, (value, os.getpid()))


def get_thread_local_var(key, init_value, reset_in_subprocess=True):
    """
    Get a thread-local variable.
    If the thread-local variable is not set, return the provided `init_value` value.
    If `get_thread_local_var` is called in a forked subprocess and `reset_in_subprocess` is True,
    return the provided `init_value` value
    """
    global _thread_locals

    if hasattr(_thread_locals, key):
        value, pid = getattr(_thread_locals, key)
        if reset_in_subprocess and pid != os.getpid():
            # `get_thread_local_var` is called in a forked subprocess, reset it.
            set_thread_local_var(key, init_value)
            return init_value
        else:
            return value
    else:
        set_thread_local_var(key, init_value)
        return init_value
