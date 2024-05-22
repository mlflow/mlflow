import functools
import os
import subprocess
import sys

from mlflow.utils.os import is_windows


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
    stream_output=False,
    **kwargs,
):
    """A convenience wrapper of `subprocess.Popen` for running a command from a Python script.

    Args:
        cmd: The command to run, as a string or a list of strings.
        throw_on_error: If True, raises an Exception if the exit code of the program is nonzero.
        extra_env: Extra environment variables to be defined when running the child process.
            If this argument is specified, `kwargs` cannot contain `env`.
        capture_output: If True, stdout and stderr will be captured and included in an exception
            message on failure; if False, these streams won't be captured.
        synchronous: If True, wait for the command to complete and return a CompletedProcess
            instance, If False, does not wait for the command to complete and return
            a Popen instance, and ignore the `throw_on_error` argument.
        stream_output: If True, stream the command's stdout and stderr to `sys.stdout`
            as a unified stream during execution.
            If False, do not stream the command's stdout and stderr to `sys.stdout`.
        kwargs: Keyword arguments (except `text`) passed to `subprocess.Popen`.

    Returns:
        If synchronous is True, return a `subprocess.CompletedProcess` instance,
        otherwise return a Popen instance.

    """
    illegal_kwargs = set(kwargs.keys()).intersection({"text"})
    if illegal_kwargs:
        raise ValueError(f"`kwargs` cannot contain {list(illegal_kwargs)}")

    env = kwargs.pop("env", None)
    if extra_env is not None and env is not None:
        raise ValueError("`extra_env` and `env` cannot be used at the same time")

    if capture_output and stream_output:
        raise ValueError(
            "`capture_output=True` and `stream_output=True` cannot be specified at the same time"
        )

    env = env if extra_env is None else {**os.environ, **extra_env}

    # In Python < 3.8, `subprocess.Popen` doesn't accept a command containing path-like
    # objects (e.g. `["ls", pathlib.Path("abc")]`) on Windows. To avoid this issue,
    # stringify all elements in `cmd`. Note `str(pathlib.Path("abc"))` returns 'abc'.
    if isinstance(cmd, list):
        cmd = list(map(str, cmd))

    if capture_output or stream_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError(
                "stdout and stderr arguments may not be used with capture_output or stream_output"
            )
        kwargs["stdout"] = subprocess.PIPE
        if capture_output:
            kwargs["stderr"] = subprocess.PIPE
        elif stream_output:
            # Redirect stderr to stdout in order to combine the streams for unified printing to
            # `sys.stdout`, as documented in
            # https://docs.python.org/3/library/subprocess.html#subprocess.run
            kwargs["stderr"] = subprocess.STDOUT

    process = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        **kwargs,
    )
    if not synchronous:
        return process

    if stream_output:
        for output_char in iter(lambda: process.stdout.read(1), ""):
            sys.stdout.write(output_char)

    stdout, stderr = process.communicate()
    returncode = process.poll()
    comp_process = subprocess.CompletedProcess(
        process.args,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
    if throw_on_error and returncode != 0:
        raise ShellCommandException.from_completed_process(comp_process)
    return comp_process


def _join_commands(*commands):
    entry_point = ["bash", "-c"] if not is_windows() else ["cmd", "/c"]
    sep = " && " if not is_windows() else " & "
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
                "The function decorated by `cache_return_value_per_process` is not allowed to be "
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
