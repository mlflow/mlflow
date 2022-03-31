import subprocess
import re

import pytest
from tests.helper_functions import _is_importable


def skip_if_pylint_unavailable():
    return pytest.mark.skipif(
        not _is_importable("pylint"), reason="pylint is required to run tests in this module"
    )


def get_pylint_msg_ids():
    from pylint.constants import MSG_TYPES

    res = subprocess.run(["pylint", "--list-msgs"], stdout=subprocess.PIPE, check=True)
    stdout = res.stdout.decode("utf-8")
    letters = "".join(MSG_TYPES.keys())
    return set(re.findall(rf"\(([{letters}][0-9]{{4}})\)", stdout))

def create_message(msg_id, node):
    import pylint.testutils

    return pylint.testutils.Message(msg_id=msg_id, node=node)


def extract_node(code):
    import astroid

    return astroid.extract_node(code)
