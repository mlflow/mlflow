import contextlib
import socket


def get_safe_port():
    """
    Get an available port on localhost, binding an ephemeral socket to it.
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]