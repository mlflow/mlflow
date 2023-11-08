import functools
import logging
import os
import platform
import signal
import subprocess
import sys
import time
from collections import namedtuple

from mlflow.exceptions import MlflowException

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
    stream_output=False,
    **kwargs,
):
    """
    A convenience wrapper of `subprocess.Popen` for running a command from a Python script.

    :param cmd: The command to run, as a string or a list of strings
    :param throw_on_error: If True, raises an Exception if the exit code of the program is nonzero.
    :param extra_env: Extra environment variables to be defined when running the child process.
                      If this argument is specified, `kwargs` cannot contain `env`.
    :param capture_output: If True, stdout and stderr will be captured and included in an exception
                           message on failure; if False, these streams won't be captured.
    :param synchronous: If True, wait for the command to complete and return a CompletedProcess
                        instance, If False, does not wait for the command to complete and return
                        a Popen instance, and ignore the `throw_on_error` argument.
    :param stream_output: If True, stream the command's stdout and stderr to `sys.stdout`
                          as a unified stream during execution.
                          If False, do not stream the command's stdout and stderr to `sys.stdout`.
    :param kwargs: Keyword arguments (except `text`) passed to `subprocess.Popen`.
    :return:  If synchronous is True, return a `subprocess.CompletedProcess` instance,
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


def kill_child_processes(parent_pid):
    """
    Gracefully terminate or kill child processes from a main process
    """

    # Terminate parent and child processes.
    os.kill(parent_pid, signal.SIGTERM)

    # Wait for 3 seconds to let the child processes exit gracefully.
    time.sleep(3)

    # If the parent process is still running after 3 seconds, send SIGKILL
    try:
        os.kill(parent_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    # Kill the children processes still alive.
    still_alive = os.popen("pgrep -P %d" % parent_pid).read().split()
    for p in still_alive:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            logging.warning("Failed to kill child process %s", p)


# As of 2023 Nov psutil doesn't have binary for Arm64 Linux, requireing building from CPython source. To
# avoid headache of building with GCC, we only use it on other platforms and use custom implementation on arm64.
def _check_and_install_psutil():
    if _is_arm64_linux():
        raise MlflowException(
            "We don't use psutil on Arm64 Linux hardware as it doesn't provide binary"
            "for it. mlflow/utils/process.py should provide custom implementation for it."
        )
    try:
        import psutil
    except ImportError:
        logging.warning(
            "Installing `psutil` package as it is required to use system metrics in MLflow. "
        )
        # install psutil from pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil<6"])


def _is_arm64_linux():
    return platform.system() == "Linux" and platform.machine() == "aarch64"


def cpu_count():
    if _is_arm64_linux():
        return Arm64ProcessUtilAdaptor.cpu_count()

    _check_and_install_psutil()
    import psutil

    return psutil.cpu_count()


def cpu_percent():
    if _is_arm64_linux():
        return Arm64ProcessUtilAdaptor.cpu_percent()

    _check_and_install_psutil()
    import psutil

    return psutil.cpu_percent()


def virtual_memory():
    if _is_arm64_linux():
        return Arm64ProcessUtilAdaptor.virtual_memory()

    _check_and_install_psutil()
    import psutil

    return psutil.virtual_memory()


def disk_usage(path: str):
    if _is_arm64_linux():
        return Arm64ProcessUtilAdaptor.disk_usage(path)

    _check_and_install_psutil()
    import psutil

    return psutil.disk_usage(path)


def net_io_counters():
    if _is_arm64_linux():
        return Arm64ProcessUtilAdaptor.net_io_counter()

    _check_and_install_psutil()
    import psutil

    return psutil.net_io_counters()


_last_cpu_times = {}


class Arm64ProcessUtilAdaptor:
    @staticmethod
    def cpu_count():
        # Get cpu count by reading /proc/cpuinfo
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        return cpuinfo.count("processor")

    @staticmethod
    def cpu_percent():
        # Get cpu usage from /proc/stat
        cpu_time = Arm64ProcessUtilAdaptor._get_cpu_times()
        last_cpu_time = _last_cpu_times.get("cpu_time", cpu_time)
        _last_cpu_times["cpu_time"] = cpu_time
        total = sum(cpu_time) - sum(last_cpu_time)
        idle = cpu_time.idle - last_cpu_time.idle
        try:
            busy_pct = 100 * (total - idle) / total
        except ZeroDivisionError:
            return 0.0
        return round(busy_pct, 1)

    @staticmethod
    def _get_cpu_times():
        cpu_time = namedtuple("scputime", ["user", "system", "idle"], defaults=(0, 0, 0))
        with open("/proc/stat") as f:
            stat = f.readline()
        fields = (float(x) for x in stat.split(" ")[2:5])
        return cpu_time(*fields)

    @staticmethod
    def virtual_memory():
        # Mimic psutil.virtual_memory return type (omitting a few unused fields)
        svmem = namedtuple("svmem", ["total", "available", "used"])
        # Get memory info from /proc/meminfo
        with open("/proc/meminfo") as f:
            meminfo = {
                line.split(":")[0]: int(line.split(":")[1].strip().split(" ")[0])
                for line in f.readlines()
            }

        # This is how psutil calculates mem_used
        # https://github.com/giampaolo/psutil/blob/902fada98ef1899d86717e7b34be46485d55e016/psutil/_pslinux.py#L477
        mem_used = (
            meminfo["MemTotal"]
            - meminfo["MemFree"]
            - meminfo.get("Cached", 0)
            - meminfo.get("Buffers", 0)
        )
        mem_used -= meminfo.get("SReclaimable", 0)
        if mem_used < 0:
            mem_used = meminfo["MemTotal"] - meminfo["MemFree"]

        return svmem(total=meminfo["MemTotal"], available=meminfo["MemAvailable"], used=mem_used)

    @staticmethod
    def disk_usage(path: str):
        # Mimic psutil.disk_usage return type
        sdiskusage = namedtuple("sdiskusage", ["total", "used", "free", "percent"])

        # Get disk usage from shutil.
        import shutil

        disk_usage = shutil.disk_usage(path)
        # shutil.disk_usage().used doesn't counts reserved space unlike psutil, so we need to calculate it.
        used_plus_reserved = disk_usage.total - disk_usage.free
        return sdiskusage(
            total=disk_usage.total,
            used=used_plus_reserved,
            free=disk_usage.free,
            percent=used_plus_reserved / disk_usage.total * 100,
        )

    @staticmethod
    def net_io_counter():
        # Mimic psutil.net_io_counters return type (omitting a few unused fields)
        snetio = namedtuple("snetio", ["bytes_sent", "bytes_recv"])
        # Get net io from /proc/net/dev
        with open("/proc/net/dev") as f:
            netinfo = f.readlines()
        netinfo = [x.strip() for x in netinfo]
        netinfo = netinfo[2:]

        net_recv = 0
        net_sent = 0
        for line in netinfo:
            fields = line.strip().split()
            net_recv += int(fields[1])
            net_sent += int(fields[9])
        return snetio(net_recv, net_sent)
