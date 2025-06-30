import threading

import pytest


@pytest.fixture(autouse=True, scope="module")
def cleanup_zombie_threads():
    for thread in threading.enumerate():
        if thread != threading.main_thread():
            thread.join(timeout=1)
