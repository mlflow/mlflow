import os
import subprocess


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
    **kwargs,
):
    """
    A convenience wrapper of `subprocess.run` for running a command from a Python script.

    :param cmd: The command to run, as a list of strings.
    :param throw_on_error: If True, raises an Exception if the exit code of the program is nonzero.
    :param extra_env: Extra environment variables to be defined when running the child process.
    :param: capture_output: If True, stdout and stderr will be captured and included in an exception
                            message on failure; if False, these streams won't be captured.
    :param kwargs: Keyword arguments (except `check`, `capture_output`, and `text`) passed to
                   `subprocess.run`.
    :return: A `subprocess.CompletedProcess` instance.
    """
    illegal_kwargs = set(kwargs.keys()).intersection(("check", "capture_output", "text"))
    if illegal_kwargs:
        raise ShellCommandException(f"`kwargs` cannot contain {illegal_kwargs}")

    env = None if extra_env is None else {**os.environ, **extra_env}
    prc = subprocess.run(
        # In Python < 3.8, `subprocess.Popen` doesn't accpet a command containing path-like
        # objects (e.g. `["ls", pathlib.Path("mlflow")]`) on Windows. To avoid this issue,
        # stringify all elements in `cmd`. Note `str(pathlib.Path("mlflow"))` returns 'mlflow'.
        map(str, cmd),
        env=env,
        check=False,
        capture_output=capture_output,
        text=True,
        **kwargs,
    )

    if throw_on_error and prc.returncode != 0:
        raise ShellCommandException.from_completed_process(prc)
    return prc
