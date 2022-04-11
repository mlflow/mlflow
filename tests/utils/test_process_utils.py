import pytest
import uuid

from mlflow.utils.process import cache_return_value_per_process
from multiprocessing import Process, Queue


@cache_return_value_per_process
def _gen_random_str1(v):
    return str(v) + uuid.uuid4().hex


@cache_return_value_per_process
def _gen_random_str2(v):
    return str(v) + uuid.uuid4().hex


@cache_return_value_per_process
def _gen_random_no_arg():
    return uuid.uuid4().hex


def _test_cache_return_value_per_process_child_proc_target(path1, path3, queue):
    # in forked out child process
    child_path1 = _gen_random_str1(True)
    child_path2 = _gen_random_str1(False)
    result = len({path1, path3, child_path1, child_path2}) == 4
    queue.put(result)


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

    queue = Queue()
    child_proc = Process(
        target=_test_cache_return_value_per_process_child_proc_target, args=(path1, path3, queue)
    )
    child_proc.start()
    child_proc.join()
    assert queue.get(), "Testing inside child process failed."
