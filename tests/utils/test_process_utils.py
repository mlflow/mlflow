import pytest
import uuid

from mlflow.utils.process import cache_return_value_per_process
import os


@cache_return_value_per_process
def _gen_random_str1(v):
    return str(v) + uuid.uuid4().hex


@cache_return_value_per_process
def _gen_random_str2(v):
    return str(v) + uuid.uuid4().hex


@cache_return_value_per_process
def _gen_random_no_arg():
    return uuid.uuid4().hex


def test_cache_return_value_per_process():

    path1 = _gen_random_str1(True)
    path2 = _gen_random_str1(True)

    assert path1 == path2

    path3 = _gen_random_str1(False)
    assert path3 != path2

    no_arg_path1 = _gen_random_no_arg()
    no_arg_path2 = _gen_random_no_arg()
    assert no_arg_path1 == no_arg_path2

    with pytest.raises(
        ValueError,
        match="The function decorated by `cache_return_value_per_process` is not allowed to be"
        "called with key-word style arguments.",
    ):
        _gen_random_str1(v=True)

    f2_path1 = _gen_random_str2(True)
    f2_path2 = _gen_random_str2(False)

    assert len({path1, path3, f2_path1, f2_path2}) == 4

    # Skip the following block on Windows which doesn't support `os.fork`
    if os.name != "nt":
        # Use `os.fork()` to make parent and child processes share the same global variable
        # `_per_process_value_cache_map`.
        pid = os.fork()
        if pid > 0:
            # in parent process
            child_pid = pid
            # check child process exit with return value 0.
            assert os.waitpid(child_pid, 0)[1] == 0
        else:
            # in forked out child process
            child_path1 = _gen_random_str1(True)
            child_path2 = _gen_random_str1(False)
            test_pass = len({path1, path3, child_path1, child_path2}) == 4
            # exit forked out child process with exit code representing testing pass or fail.
            os._exit(0 if test_pass else 1)
