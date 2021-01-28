from contextlib import contextmanager
import os

from click.testing import CliRunner


@contextmanager
def set_temp_env(temp_env):
    old_env = os.environ
    try:
        os.environ = temp_env
        yield
    finally:
        os.environ = old_env


@contextmanager
def update_temp_env(temp_env):
    old_env = os.environ
    new_env = os.environ.copy()
    new_env.update(temp_env)
    try:
        os.environ = new_env
        yield
    finally:
        os.environ = old_env


def invoke_cli_runner(*args, **kwargs):
    """
    Helper method to invoke the CliRunner while asserting that the exit code is actually 0.
    """

    res = CliRunner().invoke(*args, **kwargs)
    assert res.exit_code == 0, "Got non-zero exit code {0}. Output is: {1}".format(
        res.exit_code, res.output
    )
    return res
