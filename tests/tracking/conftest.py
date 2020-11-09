import os

import pytest


@pytest.fixture
def tmp_wkdir(tmpdir):
    initial_wkdir = os.getcwd()
    os.chdir(str(tmpdir))
    yield
    os.chdir(initial_wkdir)
