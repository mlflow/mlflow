import os
import tempfile
import pytest

from mlflow.utils.process import cache_return_value_per_process


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows does not support fork",
)
def test_cache_return_value_per_process():
    @cache_return_value_per_process
    def f1(_):
        return tempfile.mkdtemp()

    path1 = f1(True)
    path2 = f1(True)

    assert path1 == path2

    path3 = f1(False)
    assert path3 != path2

    @cache_return_value_per_process
    def f2(_):
        return tempfile.mkdtemp()

    f2_path1 = f2(True)
    f2_path2 = f2(False)

    assert len({path1, path3, f2_path1, f2_path2}) == 4

    pid = os.fork()
    if pid > 0:
        # in parent process
        child_pid = pid
        # check child process exit with return value 0.
        assert os.waitpid(child_pid, 0)[1] == 0
    else:
        # in forked out child process
        child_path1 = f1(True)
        child_path2 = f1(False)
        test_pass = len({path1, path3, child_path1, child_path2}) == 4
        # exit forked out child process with exit code representing testing pass or fail.
        os._exit(0 if test_pass else 1)
