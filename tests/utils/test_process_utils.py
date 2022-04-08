import os
import tempfile
import pytest

from mlflow.utils.process import cache_return_value_per_process
from multiprocessing import Process, Queue


@cache_return_value_per_process
def _gen_temp_dir1(_):
    return tempfile.mkdtemp()


@cache_return_value_per_process
def _gen_temp_dir2(_):
    return tempfile.mkdtemp()


def _test_cache_return_value_per_process_child_proc_target(path1, path3, queue):
    # in forked out child process
    child_path1 = _gen_temp_dir1(True)
    child_path2 = _gen_temp_dir1(False)
    result = len({path1, path3, child_path1, child_path2}) == 4
    queue.put(result)


def test_cache_return_value_per_process():

    path1 = _gen_temp_dir1(True)
    path2 = _gen_temp_dir1(True)

    assert path1 == path2

    path3 = _gen_temp_dir1(False)
    assert path3 != path2

    f2_path1 = _gen_temp_dir2(True)
    f2_path2 = _gen_temp_dir2(False)

    assert len({path1, path3, f2_path1, f2_path2}) == 4

    queue = Queue()
    child_proc = Process(
        target=_test_cache_return_value_per_process_child_proc_target, args=(path1, path3, queue)
    )
    child_proc.start()
    child_proc.join()
    assert queue.get(), "Testing inside child process failed."
