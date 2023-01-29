import platform


def is_local_os_windows():
    """
    :return: Returns true if the local system/OS name is Windows.
    """
    return platform.system().lower() == "windows"
