from __future__ import print_function
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
