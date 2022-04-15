import os
import subprocess
import functools

_IS_UNIX = os.name != "nt"


class ShellCommandException(Exception):
    @classmethod
    def from_completed_process(cls, process):
        lines = [
            f"Non-zero exit code: {process.returncode}",
            f"Command: {process.args}",
        ]
        if process.stdout:
            lines += [
                "",
                "STDOUT:",
                process.stdout,
            ]
        if process.stderr:
            lines += [
                "",
                "STDERR:",
                process.stderr,
            ]
        return cls("\n".join(lines))


def _exec_cmd(
    cmd,
    *,
    throw_on_error=True,
    extra_env=None,
    capture_output=True,
    synchronous=True,
    **kwargs,
):
    """
    A convenience wrapper of `subprocess.run` for running a command from a Python script.

    :param cmd: The command to run, as a list of strings.
    :param throw_on_error: If True, raises an Exception if the exit code of the program is nonzero.
    :param extra_env: Extra environment variables to be defined when running the child process.
                      If this argument is specified, `kwargs` cannot contain `env`.
    :param: capture_output: If True, stdout and stderr will be captured and included in an exception
                            message on failure; if False, these streams won't be captured.
    :param: synchronous: If True, wait process complete and return a CompletedProcess instance,
                         If False, does not wait process complete and return a Popen instance,
                         and ignore the `throw_on_error`, `check`, `capture_output` argument.
    :param kwargs: Keyword arguments (except `check` and `text`) passed to `subprocess.run` or
                   `subproces.Popen`.
    :return:  If synchronous is True, return a `subprocess.CompletedProcess` instance,
              otherwise return a Popen instance.
    """
    illegal_kwargs = set(kwargs.keys()).intersection(("check", "text"))
    if illegal_kwargs:
        raise ValueError(f"`kwargs` cannot contain {list(illegal_kwargs)}")

    env = kwargs.pop("env", None)
    if extra_env is not None and env is not None:
        raise ValueError("`extra_env` and `env` cannot be used at the same time")

    env = env if extra_env is None else {**os.environ, **extra_env}

    # In Python < 3.8, `subprocess.Popen` doesn't accpet a command containing path-like
    # objects (e.g. `["ls", pathlib.Path("abc")]`) on Windows. To avoid this issue,
    # stringify all elements in `cmd`. Note `str(pathlib.Path("abc"))` returns 'abc'.
    cmd = list(map(str, cmd))

    if synchronous:
        prc = subprocess.run(
            cmd,
            env=env,
            check=False,
            capture_output=capture_output,
            text=True,
            **kwargs,
        )
        if throw_on_error and prc.returncode != 0:
            raise ShellCommandException.from_completed_process(prc)
        return prc
    else:
        return subprocess.Popen(
            cmd,
            env=env,
            text=True,
            **kwargs,
        )


def _join_commands(*commands):
    entry_point = ["bash", "-c"] if _IS_UNIX else ["cmd", "/c"]
    sep = " && " if _IS_UNIX else " & "
    return [*entry_point, sep.join(map(str, commands))]


# A global map storing (function, args_tuple) --> (value, pid)
_per_process_value_cache_map = {}


def cache_return_value_per_process(fn):
    """
    A decorator which globally caches the return value of the decorated function.
    But if current process forked out a new child process, in child process,
    old cache values are invalidated.

    Restrictions: The decorated function must be called with only positional arguments,
    and all the argument values must be hashable.
    """

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if len(kwargs) > 0:
            raise ValueError(
                "The function decorated by `cache_return_value_per_process` is not allowed to be"
                "called with key-word style arguments."
            )
        if (fn, args) in _per_process_value_cache_map:
            prev_value, prev_pid = _per_process_value_cache_map.get((fn, args))
            if os.getpid() == prev_pid:
                return prev_value

        new_value = fn(*args)
        new_pid = os.getpid()
        _per_process_value_cache_map[(fn, args)] = (new_value, new_pid)
        return new_value

    return wrapped_fn
