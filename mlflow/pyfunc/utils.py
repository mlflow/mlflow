import os
import sys


def _add_code_to_system_path(code_path):
    sys.path = [code_path] + _get_code_dirs(code_path) + sys.path


def _get_code_dirs(src_code_path, dst_code_path=None):
    """
    Obtains the names of the subdirectories contained under the specified source code
    path and joins them with the specified destination code path.

    :param src_code_path: The path of the source code directory for which to list subdirectories.
    :param dst_code_path: The destination directory path to which subdirectory names should be
                          joined.
    """
    if not dst_code_path:
        dst_code_path = src_code_path
    return [(os.path.join(dst_code_path, x)) for x in os.listdir(src_code_path)
            if os.path.isdir(x) and not x == "__pycache__"]
