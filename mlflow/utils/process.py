import multiprocessing
import os
import subprocess


class ShellCommandException(Exception):
    pass


def _wait_polling(pid, poll_interval=1):
    import time
    while True:
        syscall_pid, exit_code = os.waitpid(pid, os.WNOHANG)
        if syscall_pid != 0:
            return exit_code
        time.sleep(poll_interval)


def exec_cmd(cmd, throw_on_error=True, env=None, stream_output=False, cwd=None, cmd_stdin=None,
             **kwargs):
    """
    Runs a command as a child process.

    A convenience wrapper for running a command from a Python script.
    Keyword arguments:
    cmd -- the command to run, as a list of strings
    throw_on_error -- if true, raises an Exception if the exit code of the program is nonzero
    env -- additional environment variables to be defined when running the child process
    cwd -- working directory for child process
    stream_output -- if true, does not capture standard output and error; if false, captures these
      streams and returns them
    cmd_stdin -- if specified, passes the specified string as stdin to the child process.

    Note on the return value: If stream_output is true, then only the exit code is returned. If
    stream_output is false, then a tuple of the exit code, standard output and standard error is
    returned.
    """
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)

    if stream_output:
        child = subprocess.Popen(cmd, env=cmd_env, cwd=cwd, universal_newlines=True,
                                 stdin=subprocess.PIPE, **kwargs)
        child.communicate(cmd_stdin)
        exit_code = child.wait()
        if throw_on_error and exit_code is not 0:
            raise ShellCommandException("Non-zero exitcode: %s" % (exit_code))
        return exit_code
    else:
        child = subprocess.Popen(
            cmd, env=cmd_env, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=cwd, universal_newlines=True, **kwargs)
        (stdout, stderr) = child.communicate(cmd_stdin)
        exit_code = child.wait()
        if throw_on_error and exit_code is not 0:
            raise ShellCommandException("Non-zero exit code: %s\n\nSTDOUT:\n%s\n\nSTDERR:%s" %
                                        (exit_code, stdout, stderr))
        return exit_code, stdout, stderr


def exec_fn(target, args, **kwargs):
    """
    Runs a Python function as a child process. The function's output will be streamed
    to the current process's stdout/stderr
    :param target: Function to run
    :param args: Iterable of arguments to pass to the function
    :param kwargs: Additional arguments to pass to the `multiprocessing.Process` launched to run the
                   function.
    :return: The `multiprocessing.Process` used to run the function
    """
    def wrapper():
        # Run function in a subprocess in its own process group so that it doesn't receive signals
        # sent to the parent - this allows us to consistently handle interrupting/terminating the
        # subprocess from the parent (i.e. we don't need to distinguish between the case where the
        # process group of the parent is signalled [CTRL+C in a POSIX shell] vs just the parent is
        # signalled [cancel in an IPython notebook])
        os.setsid()
        target(*args)
    p = multiprocessing.Process(target=wrapper, args=[], **kwargs)
    p.start()
    return p
