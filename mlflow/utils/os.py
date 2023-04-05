import os


def is_windows():
    """
    :return: Returns true if the local system/OS name is Windows.
    """
    return os.name == "nt"
