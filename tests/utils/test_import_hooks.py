import subprocess
import sys

import pytest

_SCRIPT = """
import importlib
import sys
from threading import Thread

from mlflow.utils import import_hooks

sys.path.insert(0, sys.argv[1])
from hook_state import callback_started, loading

register = getattr(import_hooks, sys.argv[2])
if sys.argv[3] == "True":
    register(lambda module: None, "sys")

worker = Thread(target=lambda: importlib.import_module("hook_target"), daemon=True)
worker.start()
assert loading.wait(10)


def callback(module):
    callback_started.set()
    importlib.import_module("hook_target")


register(callback, "sys")
worker.join(10)
assert not worker.is_alive()
"""


@pytest.mark.parametrize("register", ["register_post_import_hook", "register_import_error_hook"])
@pytest.mark.parametrize("already_registered", [False, True])
def test_immediate_hook_does_not_deadlock_concurrent_import(tmp_path, register, already_registered):
    (tmp_path / "hook_state.py").write_text(
        "from threading import Event\nloading, callback_started = Event(), Event()\n"
    )
    # Holds its module lock while waiting for the hook callback to start, then imports
    # another module, which requires the import hook registry lock in `find_spec`.
    (tmp_path / "hook_target.py").write_text(
        "from hook_state import callback_started, loading\n"
        "loading.set()\n"
        "assert callback_started.wait(10)\n"
        "import hook_dependency\n"
    )
    (tmp_path / "hook_dependency.py").touch()
    # Run in a subprocess so that a deadlock doesn't hang the test session.
    subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp_path), register, str(already_registered)],
        check=True,
        timeout=15,
    )
