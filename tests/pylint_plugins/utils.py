import subprocess
import re


def get_pylint_msg_ids():
    from pylint.constants import MSG_TYPES

    res = subprocess.run(["pylint", "--list-msgs"], stdout=subprocess.PIPE, check=True)
    stdout = res.stdout.decode("utf-8")
    letters = "".join(MSG_TYPES.keys())
    return set(re.findall(rf"\(([{letters}][0-9]{{4}})\)", stdout))
