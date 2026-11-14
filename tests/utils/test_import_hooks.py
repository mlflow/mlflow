import importlib
import inspect
import subprocess
import sys

import pytest

from mlflow.utils import import_hooks

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


@pytest.fixture
def module_dir(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in ("hooked_module", "broken_module"):
        sys.modules.pop(name, None)
        import_hooks._post_import_hooks.pop(name, None)
        import_hooks._import_error_hooks.pop(name, None)


def test_post_import_hook_fires_on_later_import(module_dir):
    (module_dir / "hooked_module.py").write_text("def func():\n    return 42\n")
    fired = []
    import_hooks.register_post_import_hook(fired.append, "hooked_module")
    assert fired == []

    module = importlib.import_module("hooked_module")

    assert fired == [module]
    assert module.func() == 42
    assert not isinstance(module.__loader__, import_hooks._ImportHookChainedLoader)
    assert module.__spec__.loader is module.__loader__
    assert "return 42" in inspect.getsource(module.func)


def test_import_error_hook_fires_on_failed_import(module_dir):
    (module_dir / "broken_module.py").write_text("import mlflow_nonexistent_module\n")
    failed = []
    import_hooks.register_import_error_hook(failed.append, "broken_module")

    with pytest.raises(ModuleNotFoundError, match="mlflow_nonexistent_module"):
        importlib.import_module("broken_module")

    assert failed == ["broken_module"]
